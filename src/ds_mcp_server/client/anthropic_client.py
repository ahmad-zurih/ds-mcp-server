"""
Anthropic Claude client for ds-mcp-server.
Uses the native anthropic SDK with proper tool_use content blocks.
"""
from __future__ import annotations

import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from ds_mcp_server.client._base import call_tool_async, get_server_params, list_tools_async


def _get_anthropic_client():
    try:
        from anthropic import Anthropic
    except ImportError:
        print("[Error] anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("API_KEY")
        or os.environ.get("api_key", "")
    )
    if not api_key:
        print("[Error] ANTHROPIC_API_KEY or API_KEY not set in .env")
        sys.exit(1)
    return Anthropic(api_key=api_key)


async def _chat_loop(model_override: str | None) -> None:
    client = _get_anthropic_client()
    model = model_override or os.environ.get("MODEL") or "claude-opus-4-5"
    server_params = get_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_raw = await list_tools_async(session)
            tools_anthropic = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["inputSchema"] or {"type": "object", "properties": {}},
                }
                for t in tools_raw
            ]
            print("\n[ds-mcp-client] provider : anthropic")
            print(f"[ds-mcp-client] model    : {model}")
            print(f"[ds-mcp-client] tools    : {len(tools_anthropic)} loaded")
            print("[ds-mcp-client] Type 'quit' to exit.\n")
            messages: list[dict] = []
            system_prompt = (
                "You are a helpful data science assistant with access to powerful "
                "visualization and analysis tools."
            )
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue
                messages.append({"role": "user", "content": user_input})
                while True:
                    resp = client.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=messages,
                        tools=tools_anthropic,
                    )
                    text_parts: list[str] = []
                    tool_calls = []
                    for block in resp.content:
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tool_calls.append(block)
                    if resp.stop_reason != "tool_use" or not tool_calls:
                        print(f"\nAssistant: {''.join(text_parts)}\n")
                        messages.append({"role": "assistant", "content": resp.content})
                        break
                    messages.append({"role": "assistant", "content": resp.content})
                    tool_results = []
                    for block in tool_calls:
                        print(f"  > {block.name}")
                        result = await call_tool_async(session, block.name, dict(block.input))
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": result}
                        )
                    messages.append({"role": "user", "content": tool_results})


def run_chat(model_override: str | None = None) -> None:
    asyncio.run(_chat_loop(model_override))
