"""
Multi-agent orchestration for ds-mcp-server.

A supervisor (planner) LLM that runs no tools decomposes a user request into
category-tagged subtasks and dispatches them to worker LLMs. Each worker only
sees the tools of a single category, so the system keeps working as the tool
catalogue grows. See ``runner.run_multi_agent`` for the entry point.
"""
from __future__ import annotations

from ds_mcp_server.agents.categories import (
    TOOL_CATEGORIES,
    categorize_tools,
    category_of,
)
from ds_mcp_server.agents.protocol import (
    LLMResponse,
    PlanTask,
    ToolCall,
    WorkerResult,
)

__all__ = [
    "TOOL_CATEGORIES",
    "categorize_tools",
    "category_of",
    "LLMResponse",
    "PlanTask",
    "ToolCall",
    "WorkerResult",
]
