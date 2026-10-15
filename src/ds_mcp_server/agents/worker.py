"""
Worker agent: owns a single tool category and executes one task at a time.

A worker runs a bounded LLM tool-calling loop (at most ``max_steps`` rounds) and,
if the task ends in failure, retries the whole task up to ``max_attempts`` times.
It never sees tools outside its category, so the toolset each worker reasons over
stays small even as the overall catalogue grows.
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Awaitable, Callable

from ds_mcp_server.agents.llm import LLMClient
from ds_mcp_server.agents.protocol import (
    WorkerResult,
    assistant_msg,
    tool_msg,
    user_msg,
)
from ds_mcp_server.client._base import RepeatGuard, max_tool_result_chars, truncate_text
from ds_mcp_server.prompts import CATEGORY_PLAYBOOKS

# Callable that actually invokes an MCP tool: (name, args) -> result text.
ToolRunner = Callable[[str, dict], Awaitable[str]]

# How much of a failure reason is passed to a retry / reported upward.
_RETRY_REASON_CHARS = 800

# A tool result counts as an error only when it *starts* with one of these
# (the convention every tool in this package follows) or reports a missing
# dependency. Substring matches like "failed" misfired on successful output
# (a column named failed_logins, log lines, ...) and triggered full re-runs.
_ERROR_PREFIXES = (
    "error:",
    "error ",
    "[error]",
    "[tool error]",
    "[tool timeout]",
    "traceback (most recent call last)",
)
_ERROR_SUBSTRINGS = ("not installed",)

_PLOT_RE = re.compile(
    r"^(?P<path>[^\r\n]+?\.(?:png|jpg|jpeg|svg|html|json))\|\|\|",
    re.IGNORECASE,
)


def _looks_like_error(text: str) -> bool:
    low = text.strip().lower()
    if not low:
        return False
    return low.startswith(_ERROR_PREFIXES) or any(m in low[:80] for m in _ERROR_SUBSTRINGS)


def _extract_artifact(text: str) -> dict[str, str] | None:
    m = _PLOT_RE.match(text.strip())
    if not m:
        return None
    path = m.group("path").strip()
    ext = os.path.splitext(path)[1].lower()
    kind = "html" if ext in {".html", ".json"} else "image"
    return {"path": path, "kind": kind}


_WORKER_SYSTEM = (
    "You are a specialist worker agent for the '{category}' domain.\n"
    "{description}\n\n"
    "You are given ONE focused task by a supervisor. Use ONLY the tools "
    "available to you to accomplish it. Rules:\n"
    "- Call tools to do real work; do not fabricate results.\n"
    "- If a tool returns an error, read it, fix your arguments, and try again.\n"
    "- When the task is complete, reply with a short plain-text summary of what "
    "you did and any file paths or key findings. Do NOT call a tool in that "
    "final message.\n"
    "- Stay strictly within your task; do not attempt work outside your domain."
    "{playbook}"
)


class Worker:
    def __init__(
        self,
        category: str,
        llm: LLMClient,
        tools: list[dict],
        tool_runner: ToolRunner,
        *,
        description: str = "",
        max_steps: int = 6,
        max_attempts: int = 2,
    ) -> None:
        self.category = category
        self.llm = llm
        self.tools = tools
        self.tool_runner = tool_runner
        self.description = description
        self.max_steps = max(1, max_steps)
        self.max_attempts = max(1, max_attempts)

    @property
    def tool_names(self) -> list[str]:
        return [t.get("name", "") for t in self.tools]

    async def run(self, task: str) -> WorkerResult:
        """Execute ``task``, retrying the whole task on failure.

        Each retry is told why the previous attempt failed so it does not
        repeat the same mistake; tools used and artifacts produced by earlier
        attempts are carried into the returned result.
        """
        last: WorkerResult | None = None
        used_tools: list[str] = []
        artifacts: list[dict[str, str]] = []
        for attempt in range(1, self.max_attempts + 1):
            result = await self._run_once(task, attempt, last)
            result.attempts = attempt
            used_tools.extend(result.tool_calls)
            for art in result.artifacts:
                if art not in artifacts:
                    artifacts.append(art)
            result.tool_calls = list(used_tools)
            result.artifacts = list(artifacts)
            if result.success:
                return result
            last = result
        assert last is not None
        return last

    async def _run_once(
        self, task: str, attempt: int, previous: WorkerResult | None = None
    ) -> WorkerResult:
        playbook = CATEGORY_PLAYBOOKS.get(self.category, "")
        system = _WORKER_SYSTEM.format(
            category=self.category,
            description=self.description,
            playbook=f"\n\nDomain guidance:\n{playbook}" if playbook else "",
        )
        prompt = task
        if attempt > 1:
            reason = truncate_text(
                (previous.error or previous.summary) if previous else "", _RETRY_REASON_CHARS
            )
            done = ", ".join(previous.tool_calls) if previous and previous.tool_calls else "none"
            prompt = (
                f"(retry {attempt}) A previous attempt at this task failed.\n"
                f"Tools it called: {done}\n"
                f"Why it failed: {reason or 'unknown'}\n"
                "Do not repeat the same failing call; fix the arguments or use a "
                f"different approach.\n\nTask: {task}"
            )
        history = [user_msg(prompt)]

        used_tools: list[str] = []
        artifacts: list[dict[str, str]] = []
        last_round_error = False
        last_result = ""
        result_cap = max_tool_result_chars()
        guard = RepeatGuard()

        for step in range(1, self.max_steps + 1):
            # The LLM SDKs are synchronous; run them off the event loop so the
            # web UI (progress events, other requests) stays responsive.
            resp = await asyncio.to_thread(
                self.llm.complete, history, system=system, tools=self.tools
            )

            if not resp.wants_tools:
                # Worker declared completion.
                summary = resp.text.strip() or "(worker returned no summary)"
                return WorkerResult(
                    category=self.category,
                    task=task,
                    success=not last_round_error,
                    summary=summary,
                    steps=step,
                    error=(summary + "\nLast tool error: " + last_result[:500])
                    if last_round_error
                    else "",
                    tool_calls=used_tools,
                    artifacts=artifacts,
                )

            history.append(assistant_msg(resp.text, resp.tool_calls))
            last_round_error = False
            for call in resp.tool_calls:
                used_tools.append(call.name)
                result_text = call.parse_error or guard.check(call.name, call.arguments)
                if result_text is None:
                    try:
                        result_text = await self.tool_runner(call.name, call.arguments)
                    except Exception as exc:  # noqa: BLE001 - surface any tool crash
                        result_text = f"Error: tool '{call.name}' raised {exc!r}"

                if _looks_like_error(result_text):
                    last_round_error = True
                else:
                    art = _extract_artifact(result_text)
                    if art:
                        artifacts.append(art)
                last_result = result_text

                history.append(
                    tool_msg(call.id, call.name, truncate_text(result_text, result_cap))
                )

        # Ran out of steps without a clean finish. Report the latest output so
        # the supervisor can still use partial progress.
        return WorkerResult(
            category=self.category,
            task=task,
            success=False,
            summary="",
            steps=self.max_steps,
            error=(
                f"Worker hit the {self.max_steps}-step limit without completing. "
                f"Last tool output: {truncate_text(last_result, _RETRY_REASON_CHARS)}"
            ),
            tool_calls=used_tools,
            artifacts=artifacts,
        )
