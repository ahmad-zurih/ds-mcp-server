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
import json
import os
import re
import threading
from typing import Any, AsyncIterator

from mcp import ClientSession

from ds_mcp_server.client._base import call_tool_async, list_tools_async, max_agent_steps

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

    The sync stream runs in a daemon thread; chunks are pushed into an
    asyncio.Queue via call_soon_threadsafe and consumed here, so the event
    loop is never blocked and text tokens arrive as ``text_delta`` events.
    """
    if system_prompt and not (conversation and conversation[0].get("role") == "system"):
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
    for _step in range(max_steps):
        content_parts: list[str] = []
        # index -> {id, name, arguments} accumulated across streaming chunks
        tool_calls_map: dict[int, dict[str, str]] = {}

        # --- stream in a background thread -----------------------------------
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _stream_thread() -> None:
            try:
                with llm.chat.completions.create(
                    model=model,
                    messages=conversation,
                    tools=tools_openai,
                    tool_choice="auto",
                    stream=True,
                ) as s:
                    for chunk in s:
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

        threading.Thread(target=_stream_thread, daemon=True).start()

        while True:
            item = await q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            chunk = item
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

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
        # --- end of stream ---------------------------------------------------

        full_content = "".join(content_parts) or None

        if not tool_calls_map:
            # Pure text response — text_delta events already sent to the browser
            if full_content:
                conversation.append({"role": "assistant", "content": full_content})
            return

        # Reconstruct tool_calls list ordered by streaming index
        tool_calls_list: list[dict[str, Any]] = [
            {
                "id": tool_calls_map[idx]["id"],
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
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            yield {"type": "tool_call", "name": name, "arguments": args}
            try:
                result = await call_tool_async(session, name, args)
            except Exception as exc:  # noqa: BLE001
                result = f"[tool error] {exc}"
            visible, plot = parse_tool_output(result)
            yield {"type": "tool_result", "name": name, "text": visible, "plot": plot}
            conversation.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result}
            )

    yield {
        "type": "text",
        "text": (
            f"[stopped after {max_steps} tool-calling steps without a final answer. "
            "The model may be stuck in a loop — try rephrasing your request, using a "
            "stronger model, or raising DS_MCP_MAX_STEPS.]"
        ),
    }


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


async def run_anthropic_turn(
    session: ClientSession,
    client,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """
    One user->assistant turn using the (sync) Anthropic SDK with streaming.

    The sync messages.stream() context manager runs in a daemon thread;
    text tokens are pushed into an asyncio.Queue and yielded as ``text_delta``
    events. get_final_message() is called inside the thread after the text
    stream ends, and the complete Message is passed back for tool-call handling.
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
    for _step in range(max_steps):
        # --- stream in a background thread -----------------------------------
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        _TEXT = "t"
        _FINAL = "f"
        _ERROR = "e"

        def _stream_thread() -> None:
            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=tools_anthropic,
                ) as s:
                    for text in s.text_stream:
                        loop.call_soon_threadsafe(q.put_nowait, (_TEXT, text))
                    loop.call_soon_threadsafe(
                        q.put_nowait, (_FINAL, s.get_final_message())
                    )
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, (_ERROR, exc))

        threading.Thread(target=_stream_thread, daemon=True).start()

        resp = None
        while True:
            kind, val = await q.get()
            if kind == _TEXT:
                yield {"type": "text_delta", "text": val}
            elif kind == _FINAL:
                resp = val
                break
            elif kind == _ERROR:
                raise val
        # --- end of stream ---------------------------------------------------

        text_parts: list[str] = []
        tool_calls: list[Any] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        if resp.stop_reason != "tool_use" or not tool_calls:
            messages.append({"role": "assistant", "content": resp.content})
            # text_delta events already sent; nothing more to yield
            return

        messages.append({"role": "assistant", "content": resp.content})
        tool_results: list[dict[str, Any]] = []
        for block in tool_calls:
            yield {"type": "tool_call", "name": block.name, "arguments": dict(block.input)}
            try:
                result = await call_tool_async(session, block.name, dict(block.input))
            except Exception as exc:  # noqa: BLE001
                result = f"[tool error] {exc}"
            visible, plot = parse_tool_output(result)
            yield {"type": "tool_result", "name": block.name, "text": visible, "plot": plot}
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    yield {
        "type": "text",
        "text": (
            f"[stopped after {max_steps} tool-calling steps without a final answer. "
            "The model may be stuck in a loop — try rephrasing your request, using a "
            "stronger model, or raising DS_MCP_MAX_STEPS.]"
        ),
    }


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
