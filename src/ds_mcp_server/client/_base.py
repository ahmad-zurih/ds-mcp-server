"""
Shared MCP stdio connection and tool-execution utilities.
Both openai_compat and anthropic_client import from here.
"""
from __future__ import annotations

import shutil
import sys
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters


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
    """Call an MCP tool and return its text result."""
    result = await session.call_tool(name, arguments=arguments)
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
