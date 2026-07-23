"""
Supervisor (planner) agent: decomposes a request and coordinates workers.

The supervisor never calls MCP tools itself. Each round it emits a small JSON
plan of category-tagged subtasks, the matching workers execute them, and their
feedback is fed back so the supervisor can re-plan or finish. It stops when it
declares the work ``done`` or after ``max_rounds`` rounds.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from ds_mcp_server.agents.llm import LLMClient
from ds_mcp_server.agents.protocol import (
    WorkerResult,
    assistant_msg,
    user_msg,
)
from ds_mcp_server.agents.worker import Worker

EventHook = Callable[[dict[str, Any]], None]


_SUPERVISOR_SYSTEM = """\
You are the SUPERVISOR of a team of specialist worker agents. You do NOT run any
tools yourself. You break the user's request into small, focused subtasks and
delegate each to the worker whose domain fits best. Workers report back whether
they succeeded and what they produced; you then decide whether to continue with
more subtasks (for example to fix a failure or do the next step) or to finish.

Available worker categories:
{categories}

Respond with a SINGLE JSON object and nothing else. Schema:
{{
  "reasoning": "one sentence on your current thinking",
  "status": "continue" | "done",
  "tasks": [
    {{"category": "<one of the categories above>", "task": "<clear instruction>"}}
  ],
  "final_answer": "<user-facing answer; required when status is 'done'>"
}}

Rules:
- Delegate; never claim you executed tools.
- Prefer the smallest number of subtasks that makes progress. You may issue
  several subtasks in one round when they are independent.
- If a worker FAILED, either retry it with clearer instructions or adjust the plan.
- When the overall request is satisfied, set status to "done" and write a
  complete final_answer that summarises the results for the user (include any
  file paths workers reported).
- When status is "done", "tasks" may be empty.
"""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from an LLM reply."""
    if not text:
        return None
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        # Fall back to the first balanced-looking {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


class Supervisor:
    def __init__(
        self,
        llm: LLMClient,
        workers: dict[str, Worker],
        *,
        max_rounds: int = 3,
        on_event: EventHook | None = None,
        max_history_messages: int = 20,
    ) -> None:
        self.llm = llm
        self.workers = workers
        self.max_rounds = max(1, max_rounds)
        self.on_event = on_event or (lambda e: None)
        # Persistent cross-turn memory: a flat list of clean user / assistant
        # messages (the user's requests and the supervisor's final answers).
        # Seeded into every ``run`` so the team remembers earlier turns. The
        # noisy per-round planning JSON is deliberately kept out of this list.
        self.conversation: list[dict[str, Any]] = []
        self.max_history_messages = max(0, max_history_messages)

    def _system_prompt(self) -> str:
        lines = []
        for cat, worker in self.workers.items():
            desc = worker.description or ""
            tools = ", ".join(worker.tool_names)
            lines.append(f"- {cat}: {desc} (tools: {tools})")
        return _SUPERVISOR_SYSTEM.format(categories="\n".join(lines))

    def reset_history(self) -> None:
        """Forget all remembered turns (used by the UI's 'new chat' button)."""
        self.conversation.clear()

    def _remember(self, user_request: str, answer: str) -> None:
        """Append one clean turn and trim to the configured window."""
        self.conversation.append(user_msg(user_request))
        self.conversation.append(assistant_msg(answer))
        if self.max_history_messages and len(self.conversation) > self.max_history_messages:
            del self.conversation[: len(self.conversation) - self.max_history_messages]

    async def run(self, user_request: str) -> str:
        answer = await self._run_turn(user_request)
        self._remember(user_request, answer)
        return answer

    async def _run_turn(self, user_request: str) -> str:
        system = self._system_prompt()
        # Seed with prior clean turns so the supervisor remembers the conversation.
        messages = list(self.conversation) + [user_msg(user_request)]
        all_results: list[WorkerResult] = []

        for round_no in range(1, self.max_rounds + 1):
            resp = self.llm.complete(messages, system=system)
            plan = _extract_json(resp.text)
            messages.append(assistant_msg(resp.text))

            if plan is None:
                # Could not parse a plan; treat the raw text as the final answer.
                self.on_event({"type": "final", "text": resp.text, "round": round_no})
                return resp.text.strip() or "(supervisor produced no output)"

            status = str(plan.get("status", "continue")).lower()
            tasks = plan.get("tasks") or []
            self.on_event(
                {
                    "type": "plan",
                    "round": round_no,
                    "reasoning": plan.get("reasoning", ""),
                    "status": status,
                    "tasks": tasks,
                }
            )

            if status == "done" or not tasks:
                final = (plan.get("final_answer") or "").strip()
                if not final and status != "done":
                    # No tasks and not explicitly done: nudge once more.
                    final = plan.get("reasoning", "").strip()
                if final:
                    self.on_event({"type": "final", "text": final, "round": round_no})
                    return final
                # else fall through to synthesis below

            round_results = await self._dispatch(tasks, round_no)
            all_results.extend(round_results)

            briefs = "\n\n".join(r.brief() for r in round_results)
            messages.append(
                user_msg(
                    "Worker results for this round:\n\n"
                    + (briefs or "(no tasks were dispatched)")
                    + "\n\nDecide the next step. If the request is fully satisfied, "
                    "respond with status 'done' and a complete final_answer."
                )
            )

        # Ran out of rounds — ask for a final synthesis without more tasks.
        return await self._synthesize(messages, system, all_results)

    async def _dispatch(
        self, tasks: list[dict[str, Any]], round_no: int
    ) -> list[WorkerResult]:
        results: list[WorkerResult] = []
        for task in tasks:
            category = str(task.get("category", "")).strip()
            instruction = str(task.get("task", "")).strip()
            if not instruction:
                continue
            worker = self.workers.get(category)
            if worker is None:
                results.append(
                    WorkerResult(
                        category=category or "unknown",
                        task=instruction,
                        success=False,
                        summary="",
                        error=(
                            f"No worker for category '{category}'. "
                            f"Available: {', '.join(self.workers)}."
                        ),
                    )
                )
                continue

            self.on_event(
                {
                    "type": "worker_start",
                    "round": round_no,
                    "category": category,
                    "task": instruction,
                }
            )
            result = await worker.run(instruction)
            self.on_event(
                {
                    "type": "worker_result",
                    "round": round_no,
                    "category": category,
                    "success": result.success,
                    "summary": result.summary,
                    "error": result.error,
                    "attempts": result.attempts,
                    "tool_calls": result.tool_calls,
                    "artifacts": result.artifacts,
                }
            )
            results.append(result)
        return results

    async def _synthesize(
        self,
        messages: list[dict[str, Any]],
        system: str,
        all_results: list[WorkerResult],
    ) -> str:
        messages.append(
            user_msg(
                "The maximum number of planning rounds has been reached. Do not "
                "request more tasks. Write the final user-facing answer now, "
                "summarising everything that was accomplished and noting anything "
                "that could not be completed."
            )
        )
        resp = self.llm.complete(messages, system=system)
        final = _extract_json(resp.text)
        text = ""
        if final and final.get("final_answer"):
            text = str(final["final_answer"]).strip()
        else:
            text = resp.text.strip()
        if not text:
            # Last-resort deterministic summary.
            ok = sum(1 for r in all_results if r.success)
            text = (
                f"Completed {ok}/{len(all_results)} subtasks.\n\n"
                + "\n\n".join(r.brief() for r in all_results)
            )
        self.on_event({"type": "final", "text": text, "round": self.max_rounds})
        return text
