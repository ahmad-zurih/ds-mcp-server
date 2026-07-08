"""
OpenAI-compatible client for ds-mcp-server.
Supports: OpenAI, GPUStack, Ollama, LM Studio, Azure OpenAI, Google Gemini.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import stdio_client
from openai import OpenAI

from ds_mcp_server.client._base import call_tool_async, get_server_params, list_tools_async

_PROVIDER_DEFAULTS: dict[str, dict[str, str | None]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
    },
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3"},
    "openai-compat": {"base_url": None, "model": None},
}


def _build_client(provider: str, model_override: str | None) -> tuple[OpenAI, str, str]:
    api_key = os.environ.get("API_KEY") or os.environ.get("api_key", "")
    base_url = (
        os.environ.get("API_BASE_URL")
        or os.environ.get("api_base_url")
        or _PROVIDER_DEFAULTS.get(provider, {}).get("base_url")
    )
    model = (
        model_override
        or os.environ.get("MODEL")
        or os.environ.get("model")
        or _PROVIDER_DEFAULTS.get(provider, {}).get("model")
        or "gpt-4o"
    )
    if provider == "ollama" and not api_key:
        api_key = "ollama"
    if not api_key:
        print("[Error] API_KEY not set. Add it to .env")
        sys.exit(1)
    client_kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs), model, base_url or "https://api.openai.com/v1"


async def _chat_loop(provider: str, model_override: str | None) -> None:
    llm, model, base_url = _build_client(provider, model_override)
    server_params = get_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
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
            print(f"\n[ds-mcp-client] provider : {provider}")
            print(f"[ds-mcp-client] model    : {model}")
            print(f"[ds-mcp-client] server   : {base_url}")
            print(f"[ds-mcp-client] tools    : {len(tools_openai)} loaded")
            print("[ds-mcp-client] Type 'quit' to exit.\n")
            conversation: list[dict] = []
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue
                conversation.append({"role": "user", "content": user_input})
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
                        print(f"\nAssistant: {msg.content}\n")
                        break
                    for tc in msg.tool_calls:
                        name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except Exception:
                            args = {}
                        print(f"  > {name}")
                        result = await call_tool_async(session, name, args)
                        conversation.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": result}
                        )


def run_chat(provider: str = "openai", model_override: str | None = None) -> None:
    asyncio.run(_chat_loop(provider, model_override))
