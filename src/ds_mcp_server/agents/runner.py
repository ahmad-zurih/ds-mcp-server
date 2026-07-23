"""
Runner: wire an MCP session to the supervisor/worker team and drive one request
or an interactive loop.

This is the integration layer. It connects to the bundled MCP server over stdio,
lists the available tools, groups them into per-category workers, and runs the
supervisor. Building workers from an explicit tool list (``build_team``) is kept
separate from the MCP plumbing so it can be unit-tested without a live server.
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from ds_mcp_server.agents.categories import CATEGORY_DESCRIPTIONS, categorize_tools
from ds_mcp_server.agents.llm import LLMClient, build_llm_client
from ds_mcp_server.agents.supervisor import Supervisor
from ds_mcp_server.agents.worker import Worker
from ds_mcp_server.client._base import (
    call_tool_async,
    get_server_params,
    list_tools_async,
)


@dataclass
class AgentConfig:
    """Configuration for a multi-agent run."""

    provider: str = "openai"
    planner_model: str = "gpt-4o"
    worker_model: str = "gpt-4o"
    max_rounds: int = 3
    max_worker_retries: int = 2  # retries AFTER the first attempt
    max_worker_steps: int = 6
    share_data_tools: bool = True
    # Optional per-category worker model overrides, e.g. {"system": "gpt-4o"}.
    worker_model_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def max_worker_attempts(self) -> int:
        return self.max_worker_retries + 1


def build_team(
    tools: list[dict],
    tool_runner: Callable,
    config: AgentConfig,
    *,
    planner_llm: LLMClient | None = None,
    worker_llm_factory: Callable[[str], LLMClient] | None = None,
) -> Supervisor:
    """
    Build a ``Supervisor`` wired to one ``Worker`` per present category.

    ``planner_llm`` / ``worker_llm_factory`` may be injected (used by tests with a
    fake client); otherwise real clients are built from ``config``.
    """
    if planner_llm is None:
        planner_llm = build_llm_client(config.provider, config.planner_model)
    if worker_llm_factory is None:
        def worker_llm_factory(category: str) -> LLMClient:  # noqa: E306
            model = config.worker_model_overrides.get(category, config.worker_model)
            return build_llm_client(config.provider, model)

    grouped = categorize_tools(tools, share_data_tools=config.share_data_tools)
    workers: dict[str, Worker] = {}
    for category, cat_tools in grouped.items():
        workers[category] = Worker(
            category=category,
            llm=worker_llm_factory(category),
            tools=cat_tools,
            tool_runner=tool_runner,
            description=CATEGORY_DESCRIPTIONS.get(category, ""),
            max_steps=config.max_worker_steps,
            max_attempts=config.max_worker_attempts,
        )
    return Supervisor(
        planner_llm,
        workers,
        max_rounds=config.max_rounds,
    )


# ---------------------------------------------------------------------------
# CLI-facing progress printer
# ---------------------------------------------------------------------------


def _cli_event_printer(event: dict[str, Any]) -> None:
    etype = event.get("type")
    if etype == "plan":
        tasks = event.get("tasks") or []
        print(f"\n[supervisor] round {event.get('round')} · {event.get('status')}")
        if event.get("reasoning"):
            print(f"  reasoning: {event['reasoning']}")
        for t in tasks:
            print(f"    -> [{t.get('category')}] {t.get('task')}")
    elif etype == "worker_start":
        print(f"  [{event.get('category')}] working: {event.get('task')}")
    elif etype == "worker_result":
        status = "ok" if event.get("success") else "FAILED"
        tools = ", ".join(event.get("tool_calls") or [])
        print(
            f"  [{event.get('category')}] {status} "
            f"(attempts={event.get('attempts')}"
            + (f", tools={tools}" if tools else "")
            + ")"
        )
    elif etype == "final":
        pass  # printed by the caller


# ---------------------------------------------------------------------------
# MCP integration
# ---------------------------------------------------------------------------


async def _run_once_async(request: str, config: AgentConfig, on_event) -> str:
    server_params = get_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await list_tools_async(session)

            async def tool_runner(name: str, args: dict) -> str:
                return await call_tool_async(session, name, args)

            supervisor = build_team(tools, tool_runner, config)
            supervisor.on_event = on_event
            return await supervisor.run(request)


async def _interactive_async(config: AgentConfig) -> None:
    server_params = get_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await list_tools_async(session)

            async def tool_runner(name: str, args: dict) -> str:
                return await call_tool_async(session, name, args)

            supervisor = build_team(tools, tool_runner, config)
            supervisor.on_event = _cli_event_printer

            grouped = categorize_tools(tools, share_data_tools=config.share_data_tools)
            print("\n[ds-mcp-client] mode     : multi-agent")
            print(f"[ds-mcp-client] provider : {config.provider}")
            print(f"[ds-mcp-client] planner  : {config.planner_model}")
            print(f"[ds-mcp-client] workers  : {config.worker_model}")
            print(
                f"[ds-mcp-client] team     : "
                + ", ".join(f"{c}({len(t)})" for c, t in grouped.items())
            )
            print(
                f"[ds-mcp-client] rounds={config.max_rounds} "
                f"worker_retries={config.max_worker_retries} "
                f"worker_steps={config.max_worker_steps}"
            )
            print("[ds-mcp-client] Type 'quit' to exit.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue
                try:
                    answer = await supervisor.run(user_input)
                except Exception as exc:  # noqa: BLE001
                    print(f"\n[error] {exc}\n")
                    continue
                print(f"\nAssistant: {answer}\n")


def run_multi_agent(config: AgentConfig, request: str | None = None) -> None:
    """
    Entry point used by the CLI. When ``request`` is given, run one request and
    print the answer; otherwise start an interactive loop.
    """
    try:
        if request:
            answer = asyncio.run(
                _run_once_async(request, config, _cli_event_printer)
            )
            print(f"\nAssistant: {answer}\n")
        else:
            asyncio.run(_interactive_async(config))
    except KeyboardInterrupt:
        print("\n[ds-mcp-client] interrupted.", file=sys.stderr)


def config_from_env(
    provider: str,
    planner_model: str | None,
    worker_model: str | None,
    max_rounds: int | None,
    max_worker_retries: int | None,
    max_worker_steps: int | None,
) -> AgentConfig:
    """Build an ``AgentConfig`` from CLI args with env-var fallbacks."""
    base_model = os.environ.get("MODEL") or os.environ.get("model") or "gpt-4o"
    planner = (
        planner_model
        or os.environ.get("PLANNER_MODEL")
        or base_model
    )
    worker = (
        worker_model
        or os.environ.get("WORKER_MODEL")
        or base_model
    )

    def _int_env(name: str, fallback: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return fallback
        try:
            return int(raw)
        except ValueError:
            return fallback

    return AgentConfig(
        provider=provider,
        planner_model=planner,
        worker_model=worker,
        max_rounds=max_rounds if max_rounds is not None else _int_env("MAX_ROUNDS", 3),
        max_worker_retries=(
            max_worker_retries
            if max_worker_retries is not None
            else _int_env("MAX_WORKER_RETRIES", 2)
        ),
        max_worker_steps=(
            max_worker_steps
            if max_worker_steps is not None
            else _int_env("MAX_WORKER_STEPS", 6)
        ),
    )
