"""Tests for the central system-prompt builder in ds_mcp_server.prompts."""
from __future__ import annotations

from ds_mcp_server.prompts import CATEGORY_PLAYBOOKS, build_system_prompt

_DATA_SCIENCE_TOOLS = [
    "get_all_columns_summary",
    "get_column_summary",
    "plot_interactive_histogram",
    "generate_custom_plotly",
    "generate_custom_static_plot",
    "run_correlation",
    "run_linear_regression",
]

_SYSTEM_TOOLS = [
    "run_shell_command",
    "read_file",
    "write_file",
    "patch_file",
]


def test_prompt_includes_present_capabilities() -> None:
    prompt = build_system_prompt(_DATA_SCIENCE_TOOLS)
    assert "DATA INSPECTION" in prompt
    assert "PLOTTING" in prompt
    assert "STATISTICS" in prompt
    # `df` is already loaded guidance salvaged from the old prompts.
    assert "already loaded as `df`" in prompt.lower()
    assert "GENERAL" in prompt


def test_prompt_omits_absent_capabilities() -> None:
    # Only data tools -> no plotting/stats/documents/web sections.
    prompt = build_system_prompt(["get_all_columns_summary"])
    assert "DATA INSPECTION" in prompt
    assert "PLOTTING" not in prompt
    assert "STATISTICS" not in prompt
    assert "DOCUMENTS" not in prompt
    assert "WEB & RESEARCH" not in prompt


def test_system_section_only_when_system_tools_present() -> None:
    without = build_system_prompt(_DATA_SCIENCE_TOOLS)
    assert "SYSTEM & FILE TOOLS" not in without

    with_system = build_system_prompt(_DATA_SCIENCE_TOOLS + _SYSTEM_TOOLS)
    assert "SYSTEM & FILE TOOLS" in with_system
    # Practical guardrail wording (no scary "danger zone" framing).
    assert "danger zone" not in with_system.lower()
    assert "destructive" in with_system.lower()


def test_documents_and_web_sections() -> None:
    prompt = build_system_prompt(
        ["read_pdf", "summarize_document", "search_web", "arxiv_search"]
    )
    assert "DOCUMENTS" in prompt
    assert "WEB & RESEARCH" in prompt


def test_empty_tool_list_is_safe() -> None:
    prompt = build_system_prompt([])
    assert prompt  # intro + general still present
    assert "SYSTEM & FILE TOOLS" not in prompt


def test_category_playbooks_cover_agent_categories() -> None:
    for category in ("data", "plot_interactive", "plot_static", "stats",
                     "web", "research", "documents", "system"):
        assert category in CATEGORY_PLAYBOOKS
        assert CATEGORY_PLAYBOOKS[category].strip()
