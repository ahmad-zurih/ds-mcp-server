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
from typing import Any, AsyncIterator

from mcp import ClientSession

from ds_mcp_server.client._base import call_tool_async, list_tools_async

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
) -> AsyncIterator[dict[str, Any]]:
    """
    One user->assistant turn using the OpenAI SDK. Mutates ``conversation``
    in place so the caller keeps history across turns.
    """
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

    while True:
        resp = llm.chat.completions.create(
            model=model,
            messages=conversation,
            tools=tools_openai,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        conversation.append(msg.model_dump(exclude_unset=True))

        if not msg.tool_calls:
            if msg.content:
                yield {"type": "text", "text": msg.content}
            return

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
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
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )


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
    """One user->assistant turn using the Anthropic SDK."""
    tools_raw = await list_tools_async(session)
    tools_anthropic = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["inputSchema"] or {"type": "object", "properties": {}},
        }
        for t in tools_raw
    ]

    while True:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=tools_anthropic,
        )

        text_parts: list[str] = []
        tool_calls: list[Any] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        if resp.stop_reason != "tool_use" or not tool_calls:
            messages.append({"role": "assistant", "content": resp.content})
            if text_parts:
                yield {"type": "text", "text": "".join(text_parts)}
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
) -> AsyncIterator[dict[str, Any]]:
    """
    Run one request through the supervisor/worker team, yielding UI events.

    A fresh team is built for each request (the supervisor keeps its own
    internal history for the duration of that request). Progress events are
    forwarded live; the final synthesised answer is yielded as a ``text`` event.
    """
    from ds_mcp_server.agents.runner import build_team

    tools = await list_tools_async(session)

    async def tool_runner(name: str, args: dict) -> str:
        return await call_tool_async(session, name, args)

    supervisor = build_team(tools, tool_runner, config)
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
