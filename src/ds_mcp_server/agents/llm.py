"""
LLM client abstraction for the multi-agent layer.

A single ``LLMClient`` interface hides the differences between OpenAI-compatible
APIs and the Anthropic SDK. Each backend is *stateless*: it receives the full
normalised message history plus tool specs on every call and translates them to
the provider's native format. This keeps the worker/supervisor loops
provider-agnostic and makes them testable with a scripted ``FakeLLMClient``.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from ds_mcp_server.agents.protocol import LLMResponse, ToolCall


class LLMClient(ABC):
    """Provider-agnostic chat interface returning a normalised ``LLMResponse``."""

    model: str

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Run one completion. ``tools`` are normalised MCP tool specs."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Message / tool translation helpers
# ---------------------------------------------------------------------------


def _tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m.get("content", ""),
                }
            )
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def _tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema") or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


def _messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Translate normalised messages to Anthropic's block format. Consecutive
    ``tool`` results are merged into a single ``user`` message as required.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append({"role": "assistant", "content": blocks or ""})
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": m.get("content", ""),
            }
            # Merge into the previous user message if it already holds tool results.
            if (
                out
                and out[-1]["role"] == "user"
                and isinstance(out[-1]["content"], list)
            ):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        native: list[dict[str, Any]] = []
        if system:
            native.append({"role": "system", "content": system})
        native.extend(_messages_to_openai(messages))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": native,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return LLMResponse(text=msg.content or "", tool_calls=calls)


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": _messages_to_anthropic(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        resp = self._client.messages.create(**kwargs)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        return LLMResponse(text="".join(text_parts), tool_calls=calls)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_OPENAI_BASE_DEFAULTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
}


def build_llm_client(provider: str, model: str) -> LLMClient:
    """
    Build an ``LLMClient`` for ``provider`` using the same env vars the existing
    single-agent clients read (API_KEY, API_BASE_URL, ANTHROPIC_API_KEY).
    """
    provider = (provider or "openai").lower()

    if provider == "anthropic":
        api_key = (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get("api_key", "")
        )
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY or API_KEY not set.")
        return AnthropicClient(model=model, api_key=api_key)

    api_key = os.environ.get("API_KEY") or os.environ.get("api_key", "")
    if provider == "ollama" and not api_key:
        api_key = "ollama"
    if not api_key:
        raise RuntimeError("API_KEY not set.")
    base_url = (
        os.environ.get("API_BASE_URL")
        or os.environ.get("api_base_url")
        or _OPENAI_BASE_DEFAULTS.get(provider)
    )
    return OpenAIClient(model=model, api_key=api_key, base_url=base_url)
