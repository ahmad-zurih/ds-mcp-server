"""Tests for the anti-hang safeguards in the web chat loop and tool calls.

Covers:
  * run_openai_turn stops after DS_MCP_MAX_STEPS instead of looping forever,
  * a normal (tool-free) reply still returns cleanly,
  * call_tool_async times out instead of hanging on a stuck tool.
"""
from __future__ import annotations

import asyncio

import pytest

from ds_mcp_server.client import _base
from ds_mcp_server.web import chat as chat_mod


# ---------------------------------------------------------------------------
# Fakes for the OpenAI-compatible surface used by run_openai_turn.
# ---------------------------------------------------------------------------


class _FakeFunc:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, tid: str, name: str, arguments: str) -> None:
        self.id = tid
        self.function = _FakeFunc(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls, content):
        self.tool_calls = tool_calls
        self.content = content

    def model_dump(self, exclude_unset: bool = False):
        return {"role": "assistant", "content": self.content}


class _FakeDelta:
    """Mimics openai streaming chunk.choices[0].delta."""

    def __init__(self, message):
        # For a tool-call message expose tool_calls; for text expose content.
        self.content = message.content
        # Wrap tool_calls in streaming-delta shape: each has .index, .id,
        # .function.name, .function.arguments
        if message.tool_calls:
            deltas = []
            for i, tc in enumerate(message.tool_calls):
                f = type("F", (), {"name": tc.function.name, "arguments": tc.function.arguments})()
                deltas.append(type("DTC", (), {"index": i, "id": tc.id, "function": f})())
            self.tool_calls = deltas
        else:
            self.tool_calls = None


class _FakeChunk:
    """Mimics a single openai streaming ChatCompletionChunk."""

    def __init__(self, message):
        self.choices = [type("C", (), {"delta": _FakeDelta(message)})()]


class _FakeStream:
    """Mimics openai.Stream — a context manager that yields one chunk."""

    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        yield _FakeChunk(self._message)


class _FakeCompletions:
    def __init__(self, message_factory):
        self._factory = message_factory

    def create(self, **kwargs):
        return _FakeStream(self._factory())


class _FakeChat:
    def __init__(self, message_factory):
        self.completions = _FakeCompletions(message_factory)


class _FakeLLM:
    def __init__(self, message_factory):
        self.chat = _FakeChat(message_factory)


class _FakeToolItem:
    def __init__(self, name):
        self.name = name
        self.description = ""
        self.inputSchema = {"type": "object", "properties": {}}


class _FakeToolsResult:
    def __init__(self, names):
        self.tools = [_FakeToolItem(n) for n in names]


class _FakeSession:
    """Minimal ClientSession stand-in."""

    def __init__(self, tool_names, call_impl=None):
        self._tool_names = tool_names
        self._call_impl = call_impl

    async def list_tools(self):
        return _FakeToolsResult(self._tool_names)

    async def call_tool(self, name, arguments=None):
        if self._call_impl is not None:
            return await self._call_impl(name, arguments)
        content = [type("Blk", (), {"text": "ok"})()]
        return type("R", (), {"content": content})()


async def _collect(agen):
    return [event async for event in agen]


# ---------------------------------------------------------------------------
# Step cap
# ---------------------------------------------------------------------------


def test_openai_turn_stops_when_model_loops(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "3")

    # Model ALWAYS returns a tool call -> would loop forever without the cap.
    def always_tool():
        return _FakeMessage(
            tool_calls=[_FakeToolCall("t1", "do_thing", "{}")],
            content=None,
        )

    llm = _FakeLLM(always_tool)
    session = _FakeSession(["do_thing"])
    conv: list[dict] = []

    events = asyncio.run(
        _collect(chat_mod.run_openai_turn(session, llm, "m", conv))
    )

    # Exactly 3 rounds of tool_call/tool_result, then a final "stopped" text.
    assert sum(1 for e in events if e["type"] == "tool_call") == 3
    text_events = [e for e in events if e["type"] == "text"]
    assert text_events, "expected a terminating text event"
    assert "stopped after 3" in text_events[-1]["text"]


def test_openai_turn_returns_on_final_message(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "5")

    calls = {"n": 0}

    def one_tool_then_answer():
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeMessage([_FakeToolCall("t1", "do_thing", "{}")], None)
        return _FakeMessage(tool_calls=None, content="all done")

    llm = _FakeLLM(one_tool_then_answer)
    session = _FakeSession(["do_thing"])

    events = asyncio.run(
        _collect(chat_mod.run_openai_turn(session, llm, "m", []))
    )

    # Streaming path emits text_delta events (one per token), not a single "text".
    texts = [e["text"] for e in events if e["type"] == "text_delta"]
    assert "".join(texts) == "all done"
    # No "stopped after" notice on a clean finish.
    assert all("stopped after" not in t for t in texts)


def test_system_prompt_prepended_once(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "2")

    llm = _FakeLLM(lambda: _FakeMessage(None, "hi"))
    session = _FakeSession(["do_thing"])
    conv: list[dict] = []

    asyncio.run(
        _collect(chat_mod.run_openai_turn(session, llm, "m", conv, "SYSTEM"))
    )
    assert conv[0] == {"role": "system", "content": "SYSTEM"}


# ---------------------------------------------------------------------------
# Tool-call timeout
# ---------------------------------------------------------------------------


def test_call_tool_async_times_out(monkeypatch):
    monkeypatch.setenv("DS_MCP_TOOL_TIMEOUT", "0.2")

    async def _hang(name, arguments):
        await asyncio.sleep(5)
        raise AssertionError("should have timed out")

    session = _FakeSession(["slow_tool"], call_impl=_hang)
    result = asyncio.run(_base.call_tool_async(session, "slow_tool", {}))
    assert "[tool timeout]" in result
    assert "slow_tool" in result


def test_call_tool_async_returns_text_normally(monkeypatch):
    monkeypatch.setenv("DS_MCP_TOOL_TIMEOUT", "5")
    session = _FakeSession(["fast_tool"])
    result = asyncio.run(_base.call_tool_async(session, "fast_tool", {}))
    assert result == "ok"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def test_max_agent_steps_env(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "7")
    assert _base.max_agent_steps() == 7
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "bogus")
    assert _base.max_agent_steps() == _base._DEFAULT_MAX_STEPS
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "0")
    assert _base.max_agent_steps() == 1  # floored to >= 1


def test_llm_request_timeout_env(monkeypatch):
    monkeypatch.setenv("DS_MCP_LLM_TIMEOUT", "42")
    assert _base.llm_request_timeout() == 42.0
    monkeypatch.setenv("DS_MCP_LLM_TIMEOUT", "0")
    assert _base.llm_request_timeout() is None
