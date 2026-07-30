"""
Shared MCP stdio connection and tool-execution utilities.
Both openai_compat and anthropic_client import from here.
"""
from __future__ import annotations

import asyncio
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
    return "\n".join(parts) if parts else "(no output)"


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
