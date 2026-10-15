"""
Shared MCP stdio connection and tool-execution utilities.
Both openai_compat and anthropic_client import from here.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters


def _tool_timeout() -> float | None:
    """Per-tool-call timeout in seconds (0/negative disables it).

    Configurable via ``DS_MCP_TOOL_TIMEOUT``; defaults to 180s so a hanging
    tool (slow URL fetch, stalled shell command, ...) fails instead of leaving
    the chat turn stuck forever.
    """
    raw = os.environ.get("DS_MCP_TOOL_TIMEOUT", "180")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 180.0
    return val if val > 0 else None


_DEFAULT_MAX_STEPS = 16


def max_agent_steps() -> int:
    """Hard cap on LLM<->tool rounds per turn (configurable via DS_MCP_MAX_STEPS).

    Weak or looping models can otherwise keep calling tools without ever
    emitting a final tool-free message, leaving the client hung forever.
    """
    raw = os.environ.get("DS_MCP_MAX_STEPS", str(_DEFAULT_MAX_STEPS))
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_STEPS
    return max(1, val)


def llm_request_timeout() -> float | None:
    """Per-request LLM timeout in seconds (0/negative disables it).

    Configurable via ``DS_MCP_LLM_TIMEOUT``; defaults to 300s so a stalled
    provider request fails instead of the SDK's ~10-minute default (which the
    UI would show as an endless spinner).
    """
    raw = os.environ.get("DS_MCP_LLM_TIMEOUT", "300")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 300.0
    return val if val > 0 else None


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def max_output_tokens() -> int:
    """Max tokens per Anthropic response (``DS_MCP_MAX_TOKENS``, default 8192).

    Too small a value truncates long answers and, worse, large tool-call
    arguments (e.g. generated plotting code).
    """
    return _int_env("DS_MCP_MAX_TOKENS", 8192, minimum=256)


def max_tool_result_chars() -> int:
    """Cap on a single tool result fed back to the LLM (``DS_MCP_MAX_TOOL_RESULT_CHARS``).

    One oversized result (shell output, HTTP body, ...) can otherwise blow the
    context window in a single step. ``0`` disables the cap.
    """
    return _int_env("DS_MCP_MAX_TOOL_RESULT_CHARS", 20000)


def history_char_budget() -> int:
    """Approximate size budget for the persisted chat history (``DS_MCP_HISTORY_CHARS``).

    Roughly 4 chars per token; the default (~100k chars, ~25k tokens) leaves
    room for tool schemas and the answer on most hosted models. ``0`` disables
    trimming. Lower it for small local models.
    """
    return _int_env("DS_MCP_HISTORY_CHARS", 100000)


# Older turns keep at most this much of each tool result.
_OLD_TOOL_RESULT_CHARS = 1500

# An identical (tool, arguments) call is allowed this many times per turn;
# further repeats are answered by the harness instead of the tool.
MAX_IDENTICAL_TOOL_CALLS = 2


def truncate_text(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` chars with a visible marker (``limit<=0`` = no cap)."""
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n... [truncated {omitted:,} characters]"


def parse_tool_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse a JSON tool-argument string.

    Returns ``(args, error)``. On invalid JSON (often a response cut off by the
    token limit) ``error`` is a message for the model instead of silently
    calling the tool with ``{}``.
    """
    if not raw or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {}, (
            f"[tool error] The arguments for this call were not valid JSON ({exc}); "
            "the tool was NOT run. The response may have been cut off. Re-issue the "
            "call with complete, valid JSON arguments (keep large inputs shorter)."
        )
    if not isinstance(parsed, dict):
        return {}, "[tool error] Tool arguments must be a JSON object; the tool was NOT run."
    return parsed, None


class RepeatGuard:
    """Detects a model calling the same tool with identical arguments over and over."""

    def __init__(self, limit: int = MAX_IDENTICAL_TOOL_CALLS) -> None:
        self.limit = limit
        self._counts: dict[str, int] = {}

    def check(self, name: str, args: dict[str, Any]) -> str | None:
        """Record the call; return a refusal message once it exceeds the limit."""
        try:
            key = name + "|" + json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key = name + "|" + repr(args)
        self._counts[key] = self._counts.get(key, 0) + 1
        if self._counts[key] <= self.limit:
            return None
        return (
            f"[tool error] '{name}' was already called {self.limit} times with these "
            "exact arguments in this turn, so the harness skipped it. Do not repeat it: "
            "change the arguments, try a different approach, or answer with what you have."
        )


# ---------------------------------------------------------------------------
# Chat-history memory management (single-agent loops)
# ---------------------------------------------------------------------------


def _is_turn_start(m: dict[str, Any]) -> bool:
    """A user message carrying the user's own text (not a tool_result list)."""
    return m.get("role") == "user" and isinstance(m.get("content"), str)


def _block_chars(b: Any) -> int:
    if isinstance(b, dict):
        c = b.get("content", b.get("text", b.get("input", "")))
        return len(c) if isinstance(c, str) else len(str(c))
    for attr in ("text", "input"):
        v = getattr(b, attr, None)
        if v is not None:
            return len(v) if isinstance(v, str) else len(str(v))
    return 0


def _message_chars(m: dict[str, Any]) -> int:
    content = m.get("content")
    n = len(content) if isinstance(content, str) else sum(_block_chars(b) for b in content or [])
    for tc in m.get("tool_calls") or []:
        n += len(str(tc.get("function", {}).get("arguments", ""))) if isinstance(tc, dict) else 0
    return n


def _shrink_tool_results(m: dict[str, Any], limit: int) -> None:
    if m.get("role") == "tool" and isinstance(m.get("content"), str):
        m["content"] = truncate_text(m["content"], limit)
    elif m.get("role") == "user" and isinstance(m.get("content"), list):
        for b in m["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result" and isinstance(b.get("content"), str):
                b["content"] = truncate_text(b["content"], limit)


def compact_history(messages: list[dict[str, Any]], budget: int | None = None) -> None:
    """Keep a persisted chat history within ``budget`` chars, in place.

    Works for both OpenAI-style (``role: tool``) and Anthropic-style
    (``tool_result`` blocks) histories. The current (last) turn is never
    touched. Steps, cheapest first:
      1. clip tool results of earlier turns to a short excerpt,
      2. drop the oldest whole turns (a leading system message is kept).
    Whole turns are removed so tool-call / tool-result pairs never split.
    """
    budget = history_char_budget() if budget is None else budget
    if budget <= 0:
        return
    starts = [i for i, m in enumerate(messages) if _is_turn_start(m)]
    if len(starts) < 2:
        return
    for m in messages[: starts[-1]]:
        _shrink_tool_results(m, _OLD_TOOL_RESULT_CHARS)

    total = sum(_message_chars(m) for m in messages)
    while total > budget and len(starts) >= 2:
        lo, hi = starts[0], starts[1]
        total -= sum(_message_chars(m) for m in messages[lo:hi])
        del messages[lo:hi]
        starts = [s - (hi - lo) for s in starts[1:]]


def repair_history(messages: list[dict[str, Any]], error: str) -> None:
    """Make a history valid again after a turn died mid-way, in place.

    Removes a trailing assistant tool-call message whose results are missing
    (providers reject those forever after) and closes a dangling user turn
    with a short assistant note, so the model knows the request failed and
    roles keep alternating.
    """
    # Find the last assistant message that issued tool calls.
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):  # OpenAI style
            wanted = {tc.get("id") for tc in m["tool_calls"] if isinstance(tc, dict)}
            got = {x.get("tool_call_id") for x in messages[i + 1:] if x.get("role") == "tool"}
            if not wanted <= got:
                del messages[i:]
        else:  # Anthropic style
            content = m.get("content")
            ids = {
                getattr(b, "id", None) or (b.get("id") if isinstance(b, dict) else None)
                for b in (content if isinstance(content, list) else [])
                if (getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)) == "tool_use"
            }
            if ids:
                nxt = messages[i + 1] if i + 1 < len(messages) else None
                got = {
                    b.get("tool_use_id")
                    for b in (nxt.get("content") if nxt and isinstance(nxt.get("content"), list) else [])
                    if isinstance(b, dict)
                }
                if not ids <= got:
                    del messages[i:]
        break

    if messages and messages[-1].get("role") in ("user", "tool"):
        messages.append(
            {
                "role": "assistant",
                "content": f"(The previous request was interrupted by an error: {error[:300]})",
            }
        )


def _server_script() -> str:
    """Return the absolute path to the bundled ds-mcp-server CLI entry point."""
    exe = shutil.which("ds-mcp-server")
    if exe:
        return exe
    return sys.executable


async def list_tools_async(session: ClientSession) -> list[dict[str, Any]]:
    """Return MCP tools as a list of dicts with name, description, inputSchema."""
    result = await session.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
        }
        for t in result.tools
    ]


async def call_tool_async(session: ClientSession, name: str, arguments: dict[str, Any]) -> str:
    """Call an MCP tool and return its text result.

    The call is bounded by ``DS_MCP_TOOL_TIMEOUT`` so a tool that never returns
    cannot hang the whole chat turn; on timeout a readable error string is
    returned so the model can recover instead of blocking indefinitely.
    """
    timeout = _tool_timeout()
    try:
        if timeout is None:
            result = await session.call_tool(name, arguments=arguments)
        else:
            result = await asyncio.wait_for(
                session.call_tool(name, arguments=arguments), timeout=timeout
            )
    except asyncio.TimeoutError:
        return (
            f"[tool timeout] '{name}' did not finish within {timeout:.0f}s and was "
            "aborted. Try again with narrower input or a different approach."
        )
    parts: list[str] = []
    for content in result.content:
        if hasattr(content, "text"):
            parts.append(content.text)
    text = "\n".join(parts) if parts else "(no output)"
    # Surface MCP's error flag (tool raised) in a form every loop recognises.
    if getattr(result, "isError", False) is True and not text.lower().startswith(
        ("error", "[tool error]")
    ):
        text = f"[tool error] {text}"
    return text


def get_server_params() -> StdioServerParameters:
    """Build StdioServerParameters that launch the bundled MCP server."""
    exe = shutil.which("ds-mcp-server")
    if exe:
        return StdioServerParameters(command=exe, args=[], env=None)
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", "from ds_mcp_server.cli import serve; serve()"],
        env=None,
    )
