"""Tests for the agent-loop hardening: history memory, wrap-up step, repeat
guard, invalid tool arguments, truncation handling and multi-agent retries."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

from ds_mcp_server.agents.llm import OpenAIClient, _messages_to_anthropic
from ds_mcp_server.agents.protocol import LLMResponse, ToolCall
from ds_mcp_server.agents.supervisor import Supervisor
from ds_mcp_server.agents.worker import Worker, _looks_like_error
from ds_mcp_server.client import _base
from ds_mcp_server.web import chat as chat_mod
from tests.test_agents import FakeLLMClient, make_tool_runner
from tests.test_chat_safeguards import _FakeSession, _collect


# ---------------------------------------------------------------------------
# OpenAI streaming fakes that record request kwargs
# ---------------------------------------------------------------------------


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish_reason)])


def _tc_delta(idx, tid, name, args):
    return NS(index=idx, id=tid, function=NS(name=name, arguments=args))


class _Stream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __iter__(self):
        return iter(self.chunks)


class _RecordingLLM:
    """``responder(kwargs) -> list[chunk]`` decides each streamed reply."""

    def __init__(self, responder):
        self.requests: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                return _Stream(responder(kwargs))

        self.chat = NS(completions=_Completions())


def _counting_session(names):
    calls: list[tuple[str, dict]] = []

    async def impl(name, arguments):
        calls.append((name, arguments))
        return NS(content=[NS(text=f"result of {name}")])

    return _FakeSession(names, call_impl=impl), calls


# ---------------------------------------------------------------------------
# Single-agent OpenAI loop
# ---------------------------------------------------------------------------


def test_wrap_up_call_returns_answer_after_step_budget(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "2")

    def responder(kw):
        if kw["tool_choice"] == "none":
            return [_chunk(content="summary from partial work")]
        n = len(kw["messages"])
        return [_chunk(tool_calls=[_tc_delta(0, f"t{n}", "do_thing", f'{{"n": {n}}}')])]

    llm = _RecordingLLM(responder)
    session, calls = _counting_session(["do_thing"])
    conv: list[dict] = [{"role": "user", "content": "go"}]
    events = asyncio.run(_collect(chat_mod.run_openai_turn(session, llm, "m", conv)))

    assert len(calls) == 2
    assert llm.requests[-1]["tool_choice"] == "none"
    # The wrap-up instruction is sent but not persisted.
    assert "budget" in llm.requests[-1]["messages"][-1]["content"]
    assert conv[-1] == {"role": "assistant", "content": "summary from partial work"}
    assert not any("stopped after" in e.get("text", "") for e in events)


def test_invalid_json_arguments_are_not_executed(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "4")
    replies = [
        [_chunk(tool_calls=[_tc_delta(0, "t1", "do_thing", '{"x": 1')], finish_reason="length")],
        [_chunk(content="ok")],
    ]
    llm = _RecordingLLM(lambda kw: replies.pop(0))
    session, calls = _counting_session(["do_thing"])
    conv: list[dict] = [{"role": "user", "content": "go"}]
    asyncio.run(_collect(chat_mod.run_openai_turn(session, llm, "m", conv)))

    assert calls == []
    tool_msg = next(m for m in conv if m["role"] == "tool")
    assert "not valid JSON" in tool_msg["content"]
    assert tool_msg["tool_call_id"] == "t1"


def test_repeated_identical_calls_are_blocked(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "5")
    llm = _RecordingLLM(
        lambda kw: [_chunk(content="done")]
        if kw["tool_choice"] == "none"
        else [_chunk(tool_calls=[_tc_delta(0, "", "do_thing", "{}")])]
    )
    session, calls = _counting_session(["do_thing"])
    conv: list[dict] = [{"role": "user", "content": "go"}]
    asyncio.run(_collect(chat_mod.run_openai_turn(session, llm, "m", conv)))

    assert len(calls) == _base.MAX_IDENTICAL_TOOL_CALLS
    tool_msgs = [m for m in conv if m["role"] == "tool"]
    assert "already called" in tool_msgs[-1]["content"]
    # Missing ids from the provider are filled in.
    assert all(m["tool_call_id"] for m in tool_msgs)


def test_tool_results_are_capped(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_TOOL_RESULT_CHARS", "100")
    replies = [
        [_chunk(tool_calls=[_tc_delta(0, "t1", "big", "{}")])],
        [_chunk(content="ok")],
    ]
    llm = _RecordingLLM(lambda kw: replies.pop(0))

    async def impl(name, arguments):
        return NS(content=[NS(text="x" * 5000)])

    conv: list[dict] = [{"role": "user", "content": "go"}]
    asyncio.run(
        _collect(chat_mod.run_openai_turn(_FakeSession(["big"], impl), llm, "m", conv))
    )
    tool_msg = next(m for m in conv if m["role"] == "tool")
    assert len(tool_msg["content"]) < 200
    assert "truncated" in tool_msg["content"]


def test_system_prompt_is_refreshed():
    llm = _RecordingLLM(lambda kw: [_chunk(content="hi")])
    conv = [{"role": "system", "content": "OLD"}, {"role": "user", "content": "x"}]
    asyncio.run(
        _collect(chat_mod.run_openai_turn(_FakeSession([]), llm, "m", conv, "NEW"))
    )
    assert conv[0]["content"] == "NEW"
    assert sum(1 for m in conv if m["role"] == "system") == 1


# ---------------------------------------------------------------------------
# Single-agent Anthropic loop
# ---------------------------------------------------------------------------


class _AnthStream:
    def __init__(self, final):
        self.final = final
        self.text_stream = [b.text for b in final.content if b.type == "text"]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def get_final_message(self):
        return self.final


class _FakeAnthropic:
    def __init__(self, finals):
        self.finals = list(finals)
        self.requests: list[dict] = []
        outer = self

        class _Messages:
            def stream(self, **kw):
                outer.requests.append(kw)
                return _AnthStream(outer.finals.pop(0))

        self.messages = _Messages()


def _text(t):
    return NS(type="text", text=t)


def _tool_use(tid, name, inp):
    return NS(type="tool_use", id=tid, name=name, input=inp)


def test_anthropic_truncated_tool_call_keeps_history_valid():
    client = _FakeAnthropic(
        [
            NS(content=[_text("let me"), _tool_use("tu1", "do_thing", {"code": "par"})], stop_reason="max_tokens"),
            NS(content=[_text("final")], stop_reason="end_turn"),
        ]
    )
    session, calls = _counting_session(["do_thing"])
    msgs: list[dict] = [{"role": "user", "content": "go"}]
    asyncio.run(_collect(chat_mod.run_anthropic_turn(session, client, "m", "sys", msgs)))

    assert calls == []  # incomplete call never executed
    results = msgs[2]["content"]
    assert results[0]["tool_use_id"] == "tu1"
    assert "NOT run" in results[0]["content"]
    assert msgs[-1]["content"][0].text == "final"


def test_anthropic_wrap_up_uses_tool_choice_none(monkeypatch):
    monkeypatch.setenv("DS_MCP_MAX_STEPS", "1")
    client = _FakeAnthropic(
        [
            NS(content=[_tool_use("tu1", "do_thing", {})], stop_reason="tool_use"),
            NS(content=[_tool_use("tu2", "do_thing", {})], stop_reason="tool_use"),
        ]
    )
    session, _ = _counting_session(["do_thing"])
    msgs: list[dict] = [{"role": "user", "content": "go"}]
    events = asyncio.run(
        _collect(chat_mod.run_anthropic_turn(session, client, "m", "sys", msgs))
    )
    assert client.requests[-1]["tool_choice"] == {"type": "none"}
    # Ignored tool_use in the wrap-up is not persisted; history ends on assistant text.
    assert msgs[-1]["role"] == "assistant"
    assert isinstance(msgs[-1]["content"], str)
    assert any("stopped after 1" in e.get("text", "") for e in events)


# ---------------------------------------------------------------------------
# History compaction / repair
# ---------------------------------------------------------------------------


def _turn(i, size):
    return [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "tool_calls": [{"id": f"c{i}", "function": {"arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": "r" * size},
        {"role": "assistant", "content": f"a{i}"},
    ]


def test_compact_history_shrinks_then_drops_old_turns():
    msgs = [{"role": "system", "content": "S"}]
    for i in range(5):
        msgs += _turn(i, 10000)
    _base.compact_history(msgs, budget=8000)

    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"  # starts on a whole turn
    # The current (last) turn is untouched.
    assert msgs[-2]["content"] == "r" * 10000
    assert msgs[-4]["content"] == "q4"
    # Older tool results that survived were clipped.
    for m in msgs[:-4]:
        if m["role"] == "tool":
            assert len(m["content"]) < 2000


def test_compact_history_noop_within_budget():
    msgs = _turn(0, 10) + _turn(1, 10)
    before = [dict(m) for m in msgs]
    _base.compact_history(msgs, budget=100000)
    assert msgs == before


def test_repair_history_openai_drops_orphan_tool_calls():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "x"},
    ]
    _base.repair_history(msgs, "boom")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert "boom" in msgs[-1]["content"]


def test_repair_history_anthropic_keeps_complete_pairs():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [_tool_use("t1", "x", {})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    _base.repair_history(msgs, "rate limited")
    assert len(msgs) == 4
    assert msgs[-1]["role"] == "assistant"


def test_call_tool_async_flags_is_error(monkeypatch):
    async def impl(name, arguments):
        return NS(content=[NS(text="division by zero")], isError=True)

    out = asyncio.run(_base.call_tool_async(_FakeSession(["t"], impl), "t", {}))
    assert out.startswith("[tool error]")


# ---------------------------------------------------------------------------
# Multi-agent
# ---------------------------------------------------------------------------


def test_error_detection_avoids_false_positives():
    assert _looks_like_error("failed_logins: mean=3.2, max=10") is False
    assert _looks_like_error("Column summary — could not find outliers") is False
    assert _looks_like_error("[tool timeout] 'x' did not finish") is True
    assert _looks_like_error("[tool error] boom") is True
    assert _looks_like_error("Error running command: x") is True


def test_worker_retry_receives_failure_reason():
    llm = FakeLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="t", arguments={})]),
            LLMResponse(text="gave up"),
            LLMResponse(text="fixed it"),
        ]
    )
    runner = make_tool_runner({"t": ["Error: column 'foo' not found"]})
    worker = Worker("stats", llm, [{"name": "t"}], runner, max_attempts=2)
    result = asyncio.run(worker.run("analyse foo"))

    assert result.success and result.attempts == 2
    retry_prompt = llm.calls[2]["messages"][0]["content"]
    assert "column 'foo' not found" in retry_prompt
    assert result.tool_calls == ["t"]  # carried over from attempt 1


def test_worker_does_not_run_unparseable_call():
    llm = FakeLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="t", parse_error="[tool error] bad json")]),
            LLMResponse(text="done"),
        ]
    )
    runner = make_tool_runner({"t": ["ok"]})
    log = runner.log
    asyncio.run(Worker("stats", llm, [{"name": "t"}], runner, max_attempts=1).run("x"))
    assert log == []


def test_supervisor_repairs_malformed_plan_json():
    planner = FakeLLMClient(
        [
            LLMResponse(text='{"status": "done", "final_answer": "hi"'),  # broken
            LLMResponse(text='{"status": "done", "tasks": [], "final_answer": "hello"}'),
        ]
    )
    sup = Supervisor(planner, {}, max_rounds=2)
    assert asyncio.run(sup.run("hey")) == "hello"


def test_supervisor_ignores_non_dict_tasks():
    planner = FakeLLMClient(
        [
            LLMResponse(text='{"status": "continue", "tasks": ["oops", 3]}'),
            LLMResponse(text='{"status": "done", "final_answer": "ok"}'),
        ]
    )
    sup = Supervisor(planner, {}, max_rounds=3)
    assert asyncio.run(sup.run("x")) == "ok"


def test_supervisor_remembers_artifact_paths():
    planner = FakeLLMClient(
        [
            LLMResponse(text='{"status": "continue", "tasks": [{"category": "plot", "task": "p"}]}'),
            LLMResponse(text='{"status": "done", "final_answer": "Here is your chart."}'),
        ]
    )
    wllm = FakeLLMClient(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="plot_x", arguments={})]),
            LLMResponse(text="made it"),
        ]
    )
    runner = make_tool_runner({"plot_x": ["/tmp/plots/a.png|||code"]})
    worker = Worker("plot", wllm, [{"name": "plot_x"}], runner)
    sup = Supervisor(planner, {"plot": worker}, max_rounds=3)
    assert asyncio.run(sup.run("chart")) == "Here is your chart."
    assert "/tmp/plots/a.png" in sup.conversation[-1]["content"]


def test_anthropic_translation_never_sends_empty_content():
    out = _messages_to_anthropic(
        [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
    )
    assert all(m["content"] for m in out)


def test_openai_agent_client_omits_max_tokens():
    client = OpenAIClient(model="o3", api_key="x")
    sent: dict = {}

    def create(**kw):
        sent.update(kw)
        msg = NS(content="hi", tool_calls=[NS(id=None, function=NS(name="t", arguments="{bad"))])
        return NS(choices=[NS(message=msg)])

    client._client = NS(chat=NS(completions=NS(create=create)))
    resp = client.complete([{"role": "user", "content": "q"}], tools=[{"name": "t"}])
    assert "max_tokens" not in sent
    assert resp.tool_calls[0].parse_error
    assert resp.tool_calls[0].id
