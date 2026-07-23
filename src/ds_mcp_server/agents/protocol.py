"""
Provider-agnostic data structures shared across the multi-agent layer.

The LLM backends translate their native responses into these types so the
worker and supervisor loops never touch provider-specific SDK objects. This
also makes the whole system testable with a scripted ``FakeLLMClient``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation requested by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalised result of one LLM completion."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class PlanTask:
    """One subtask the supervisor assigns to a worker."""

    category: str
    task: str
    id: str = ""


@dataclass
class WorkerResult:
    """Outcome of a worker executing one task."""

    category: str
    task: str
    success: bool
    summary: str
    steps: int = 0
    attempts: int = 1
    error: str = ""
    tool_calls: list[str] = field(default_factory=list)
    artifacts: list[dict[str, str]] = field(default_factory=list)

    def brief(self) -> str:
        """Compact one-block description fed back to the supervisor."""
        status = "SUCCESS" if self.success else "FAILURE"
        lines = [f"[{status}] ({self.category}) {self.task}"]
        if self.tool_calls:
            lines.append("  tools used: " + ", ".join(self.tool_calls))
        if self.artifacts:
            paths = ", ".join(a.get("path", "") for a in self.artifacts)
            lines.append("  artifacts: " + paths)
        lines.append("  result: " + (self.summary or self.error or "(no output)"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalised conversation messages
# ---------------------------------------------------------------------------
# Messages are plain dicts so they serialise trivially and tests can assert on
# them. Shapes:
#   {"role": "user",      "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool",      "tool_call_id": str, "name": str, "content": str}


def user_msg(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def assistant_msg(content: str, tool_calls: list[ToolCall] | None = None) -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": tool_calls or []}


def tool_msg(tool_call_id: str, name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}
