"""
Chat backends for the web UI.

Wraps the two existing provider paths (OpenAI-compatible + Anthropic) with an
async generator API that yields structured events for the WebSocket transport:

    {"type": "tool_call",  "name": str}
    {"type": "tool_result","name": str, "text": str, "plot": {...} | None}
    {"type": "text",       "text": str}
    {"type": "done"}
    {"type": "error",      "message": str}

The websocket layer forwards these as JSON messages to the browser.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
from contextlib import aclosing
from typing import Any, AsyncIterator, Callable

from mcp import ClientSession

from ds_mcp_server.client._base import (
    RepeatGuard,
    call_tool_async,
    compact_history,
    list_tools_async,
    max_agent_steps,
    max_output_tokens,
    max_tool_result_chars,
    parse_tool_arguments,
    truncate_text,
)

# Matches "path|||code" returns from the plot tools.
_PLOT_RETURN_RE = re.compile(
    r"^(?P<path>[^\r\n]+?\.(?:png|jpg|jpeg|svg|html|json))\|\|\|(?P<code>.*)$",
    re.DOTALL | re.IGNORECASE,
)


def parse_tool_output(text: str) -> tuple[str, dict[str, str] | None]:
    """
    Interpret a raw MCP tool text result.

    Returns (visible_text, plot_meta_or_none). If the result looks like the
    ``path|||code`` plot format, ``plot_meta`` is ``{"path": ..., "code": ...,
    "kind": "html"|"image"}``.
    """
    m = _PLOT_RETURN_RE.match(text.strip())
    if not m:
        return text, None
    path = m.group("path").strip()
    code = m.group("code").strip()
    ext = os.path.splitext(path)[1].lower()
    kind = "html" if ext in {".html", ".json"} else "image"
    return f"Generated plot: {os.path.basename(path)}", {
        "path": path,
        "code": code,
        "kind": kind,
    }


# ---------------------------------------------------------------------------
# Shared loop helpers
# ---------------------------------------------------------------------------

# Sent (not persisted) with the final, tool-free call once the step budget is spent.
_WRAP_UP_NOTE = (
    "[harness] The tool-call budget for this request is used up. Do not call any "
    "more tools. Using only the results above, give the user your best final "
    "answer now, and say clearly what could not be completed."
)


def _stop_notice(max_steps: int) -> str:
    return (
        f"[stopped after {max_steps} tool-calling steps without a final answer. "
        "The model may be stuck in a loop — try rephrasing your request, using a "
        "stronger model, or raising DS_MCP_MAX_STEPS.]"
    )


def _post(loop: asyncio.AbstractEventLoop, q: asyncio.Queue, item: Any) -> None:
    """Hand an item to the event loop from a worker thread (no-op if it closed)."""
    try:
        loop.call_soon_threadsafe(q.put_nowait, item)
    except RuntimeError:
        pass


async def _thread_stream(
    open_stream: Callable[[], Any],
    iterate: Callable[[Any], Any] = lambda s: s,
    finalize: Callable[[Any], Any] | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """
    Drive a *sync* SDK streaming context manager from a daemon thread.

    Yields ``("item", x)`` for each streamed element and, if ``finalize`` is
    given, one ``("final", finalize(stream))``. Items cross into the event loop
    via ``call_soon_threadsafe`` so the loop is never blocked. When the consumer
    stops early (error, client disconnect) the thread is told to stop and the
    HTTP stream is closed instead of being read to the end in the background.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def _worker() -> None:
        try:
            with open_stream() as s:
                for item in iterate(s):
                    if stop.is_set():
                        return
                    _post(loop, q, ("item", item))
                if finalize is not None and not stop.is_set():
                    _post(loop, q, ("final", finalize(s)))
        except Exception as exc:  # noqa: BLE001
            _post(loop, q, ("error", exc))
        finally:
            _post(loop, q, ("end", None))

    threading.Thread(target=_worker, daemon=True).start()
    try:
        while True:
            kind, val = await q.get()
            if kind == "end":
                return
            if kind == "error":
                raise val
            yield kind, val
    finally:
        stop.set()


async def _execute_tool(
    session: ClientSession,
    name: str,
    args: dict[str, Any],
    arg_error: str | None,
    guard: RepeatGuard,
) -> str:
    """Run one tool call, unless its arguments are broken or it is a repeat."""
    result = arg_error or guard.check(name, args)
    if result is not None:
        return result
    try:
        return await call_tool_async(session, name, args)
    except Exception as exc:  # noqa: BLE001
        return f"[tool error] {exc}"


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (OpenAI, Ollama, Gemini, LM Studio, GPUStack, …)
# ---------------------------------------------------------------------------


async def run_openai_turn(
    session: ClientSession,
    llm,
    model: str,
    conversation: list[dict[str, Any]],
    system_prompt: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    One user->assistant turn using the (sync) OpenAI SDK with streaming.

    Runs up to ``DS_MCP_MAX_STEPS`` tool rounds, then one last call with tools
    disabled so the user gets an answer built from the work already done
    rather than a bare "stopped" notice. ``conversation`` is the persistent
    history; it is compacted to ``DS_MCP_HISTORY_CHARS`` before each call.
    """
    if system_prompt:
        if conversation and conversation[0].get("role") == "system":
            # Refresh: the tool set may have changed (settings toggles).
            conversation[0]["content"] = system_prompt
        else:
            conversation.insert(0, {"role": "system", "content": system_prompt})

    tools_raw = await list_tools_async(session)
    tools_openai = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"] or {"type": "object", "properties": {}},
            },
        }
        for t in tools_raw
    ]

    max_steps = max_agent_steps()
    result_cap = max_tool_result_chars()
    guard = RepeatGuard()
    for step in range(max_steps + 1):
        wrap_up = step == max_steps
        compact_history(conversation)
        request_messages = conversation
        if wrap_up:
            request_messages = conversation + [{"role": "user", "content": _WRAP_UP_NOTE}]

        content_parts: list[str] = []
        # index -> {id, name, arguments} accumulated across streaming chunks
        tool_calls_map: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None

        def _open(msgs=request_messages, choice="none" if wrap_up else "auto"):
            return llm.chat.completions.create(
                model=model,
                messages=msgs,
                tools=tools_openai,
                tool_choice=choice,
                stream=True,
            )

        async with aclosing(_thread_stream(_open)) as stream:
            async for _kind, chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                delta = choice.delta

                if delta.content:
                    content_parts.append(delta.content)
                    yield {"type": "text_delta", "text": delta.content}

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                        # id only appears in the first chunk for each tool call
                        if tc.id:
                            tool_calls_map[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_map[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                tool_calls_map[idx]["arguments"] += tc.function.arguments

        full_content = "".join(content_parts) or None

        if wrap_up or not tool_calls_map:
            # Final answer. Tool calls in the wrap-up call are ignored (some
            # providers do not honour tool_choice="none").
            if full_content:
                conversation.append({"role": "assistant", "content": full_content})
                if finish_reason == "length":
                    yield {"type": "text", "text": "[answer cut off: the model hit its output-length limit]"}
            elif wrap_up:
                yield {"type": "text", "text": _stop_notice(max_steps)}
            return

        # Reconstruct tool_calls list ordered by streaming index. Some
        # OpenAI-compatible servers omit ids; tool replies need one to match.
        tool_calls_list: list[dict[str, Any]] = [
            {
                "id": tool_calls_map[idx]["id"] or f"call_{step}_{idx}",
                "type": "function",
                "function": {
                    "name": tool_calls_map[idx]["name"],
                    "arguments": tool_calls_map[idx]["arguments"],
                },
            }
            for idx in sorted(tool_calls_map)
        ]
        assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls_list}
        if full_content:
            assistant_msg["content"] = full_content
        conversation.append(assistant_msg)

        for tc in tool_calls_list:
            name = tc["function"]["name"]
            args, arg_error = parse_tool_arguments(tc["function"]["arguments"])
            yield {"type": "tool_call", "name": name, "arguments": args}
            result = await _execute_tool(session, name, args, arg_error, guard)
            visible, plot = parse_tool_output(result)
            yield {"type": "tool_result", "name": name, "text": visible, "plot": plot}
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": truncate_text(result, result_cap),
                }
            )


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


def _with_note(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    """Copy of ``messages`` with ``note`` added to the trailing user message."""
    if not messages or messages[-1].get("role") != "user":
        return messages + [{"role": "user", "content": note}]
    last = messages[-1]
    content = last["content"]
    blocks = list(content) if isinstance(content, list) else [{"type": "text", "text": content}]
    blocks.append({"type": "text", "text": note})
    return messages[:-1] + [{"role": "user", "content": blocks}]


async def run_anthropic_turn(
    session: ClientSession,
    client,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """
    One user->assistant turn using the (sync) Anthropic SDK with streaming.

    Text tokens are yielded as ``text_delta`` events; the complete Message is
    fetched with get_final_message() for tool-call handling. Same step budget,
    wrap-up call and history compaction as ``run_openai_turn``.
    """
    tools_raw = await list_tools_async(session)
    tools_anthropic = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["inputSchema"] or {"type": "object", "properties": {}},
        }
        for t in tools_raw
    ]

    max_steps = max_agent_steps()
    max_tokens = max_output_tokens()
    result_cap = max_tool_result_chars()
    guard = RepeatGuard()
    for step in range(max_steps + 1):
        wrap_up = step == max_steps
        compact_history(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": _with_note(messages, _WRAP_UP_NOTE) if wrap_up else messages,
            # Tools must stay declared while the history holds tool_use blocks.
            "tools": tools_anthropic,
        }
        if wrap_up:
            kwargs["tool_choice"] = {"type": "none"}

        resp = None
        async with aclosing(
            _thread_stream(
                lambda kw=kwargs: client.messages.stream(**kw),
                iterate=lambda s: s.text_stream,
                finalize=lambda s: s.get_final_message(),
            )
        ) as stream:
            async for kind, val in stream:
                if kind == "item":
                    yield {"type": "text_delta", "text": val}
                else:
                    resp = val
        if resp is None:
            raise RuntimeError("Anthropic stream ended without a final message")

        tool_calls = [b for b in resp.content if b.type == "tool_use"]
        truncated = resp.stop_reason == "max_tokens"

        if wrap_up or not tool_calls:
            # Never persist a tool_use block without a matching tool_result:
            # the API would reject every later request in this chat.
            content = [b for b in resp.content if b.type != "tool_use"]
            if content:
                messages.append({"role": "assistant", "content": content})
                if truncated:
                    yield {
                        "type": "text",
                        "text": "[answer cut off at the output-token limit; raise DS_MCP_MAX_TOKENS]",
                    }
            else:
                note = _stop_notice(max_steps) if wrap_up else "(no response)"
                messages.append({"role": "assistant", "content": note})
                if wrap_up:
                    yield {"type": "text", "text": note}
            return

        messages.append({"role": "assistant", "content": resp.content})
        tool_results: list[dict[str, Any]] = []
        for block in tool_calls:
            args = dict(block.input) if isinstance(block.input, dict) else {}
            yield {"type": "tool_call", "name": block.name, "arguments": args}
            if truncated:
                # The response ran out of tokens while writing this call, so
                # its input is incomplete — don't run it, tell the model why.
                result = (
                    "[tool error] Your response hit the output-token limit while "
                    "writing this tool call, so its arguments are incomplete and it "
                    "was NOT run. Retry with a shorter input (e.g. less code per call)."
                )
            else:
                result = await _execute_tool(session, block.name, args, None, guard)
            visible, plot = parse_tool_output(result)
            yield {"type": "tool_result", "name": block.name, "text": visible, "plot": plot}
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": truncate_text(result, result_cap),
                }
            )
        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Multi-agent provider (supervisor + specialist workers)
# ---------------------------------------------------------------------------


def _agent_event_to_ui(event: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Translate a supervisor/worker progress event into websocket UI events.

    The ``final`` event is intentionally dropped here; the caller yields the
    supervisor's return value as the single authoritative ``text`` event.
    """
    etype = event.get("type")
    out: list[dict[str, Any]] = []
    if etype == "plan":
        out.append(
            {
                "type": "plan",
                "round": event.get("round"),
                "reasoning": event.get("reasoning", ""),
                "status": event.get("status", ""),
                "tasks": event.get("tasks") or [],
            }
        )
    elif etype == "worker_start":
        out.append(
            {
                "type": "worker_start",
                "round": event.get("round"),
                "category": event.get("category"),
                "task": event.get("task"),
            }
        )
    elif etype == "worker_result":
        out.append(
            {
                "type": "worker_result",
                "round": event.get("round"),
                "category": event.get("category"),
                "success": bool(event.get("success")),
                "attempts": event.get("attempts"),
                "tool_calls": event.get("tool_calls") or [],
                "error": event.get("error", ""),
            }
        )
        # Surface any plots/artifacts the worker produced so the UI renders them.
        for art in event.get("artifacts") or []:
            path = art.get("path")
            if not path:
                continue
            out.append(
                {
                    "type": "tool_result",
                    "name": event.get("category", "worker"),
                    "text": "",
                    "plot": {"path": path, "kind": art.get("kind", "image"), "code": ""},
                }
            )
    return out


async def run_multi_agent_turn(
    session: ClientSession,
    config: Any,
    user_message: str,
    history: list[dict[str, Any]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Run one request through the supervisor/worker team, yielding UI events.

    A fresh team is built for each request so it picks up any live config
    changes, but the persistent ``history`` list (owned by the caller) is
    injected into the supervisor so the team remembers earlier turns. The
    supervisor appends this turn's user request and final answer to that list
    in place. Progress events are forwarded live; the final synthesised answer
    is yielded as a ``text`` event.
    """
    from ds_mcp_server.agents.runner import build_team

    tools = await list_tools_async(session)

    async def tool_runner(name: str, args: dict) -> str:
        return await call_tool_async(session, name, args)

    supervisor = build_team(tools, tool_runner, config)
    if history is not None:
        supervisor.conversation = history
    queue: asyncio.Queue = asyncio.Queue()
    supervisor.on_event = lambda e: queue.put_nowait(e)

    run_task = asyncio.create_task(supervisor.run(user_message))
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait(
                {getter, run_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if getter in done:
                for ui in _agent_event_to_ui(getter.result()):
                    yield ui
            else:
                getter.cancel()
            if run_task in done:
                # Drain anything the supervisor emitted just before finishing.
                while not queue.empty():
                    for ui in _agent_event_to_ui(queue.get_nowait()):
                        yield ui
                break
        final = run_task.result()
    except BaseException:
        run_task.cancel()
        raise

    if final:
        yield {"type": "text", "text": final}
