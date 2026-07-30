"""
Shared configuration constants for the visualization / data tools.

Historically this module also held a set of multi-agent system prompts and tool
scoping tables for an earlier agent design. Those are gone: the live prompts now
live in ``ds_mcp_server.prompts`` (single-agent + worker playbooks) and the
category/tool map lives in ``ds_mcp_server.agents.categories``. Only the shared
constants that other tool modules import remain here.
"""
from __future__ import annotations

# Maximum number of rows a dataframe may have before the data tools refuse to
# load it, to keep memory and response sizes bounded.
MAX_ROWS: int = 300_000
