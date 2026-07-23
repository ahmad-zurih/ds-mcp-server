"""
Tool categorisation for the multi-agent system.

Each MCP tool belongs to exactly one *category*. A worker agent is created per
category that is actually present on the connected server, and only receives
that category's tools. Data-exploration tools are additionally shared into the
plotting and stats workers so they can inspect columns before acting.
"""
from __future__ import annotations

# Explicit membership. Anything not listed here is matched by prefix in
# ``category_of`` (e.g. a future ``plot_static_violin`` still lands in
# ``plot_static``) and otherwise falls back to the ``misc`` bucket.
TOOL_CATEGORIES: dict[str, list[str]] = {
    "data": [
        "get_all_columns_summary",
        "get_column_summary",
    ],
    "plot_interactive": [
        "plot_interactive_histogram",
        "plot_interactive_scatterplot",
        "plot_interactive_boxplot",
        "plot_interactive_lineplot",
        "plot_interactive_barchart",
        "plot_interactive_scatter_matrix",
        "plot_interactive_correlation_heatmap",
        "generate_custom_plotly",
    ],
    "plot_static": [
        "plot_static_histogram",
        "plot_static_scatterplot",
        "plot_static_boxplot",
        "plot_static_lineplot",
        "plot_static_barchart",
        "plot_static_pairplot",
        "plot_static_wordcloud",
        "plot_static_correlation_heatmap",
        "generate_custom_static_plot",
    ],
    "stats": [
        "run_correlation",
        "run_group_comparison",
        "run_linear_regression",
        "rank_target_correlations",
    ],
    "web": [
        "fetch_webpage",
        "search_web",
        "screenshot_webpage",
        "screenshot_webpages",
    ],
    "research": [
        "arxiv_search",
        "github_search",
        "github_read_file",
        "wikipedia",
        "youtube_transcript",
    ],
    "system": [
        "run_shell_command",
        "read_file",
        "write_file",
        "patch_file",
        "list_directory",
        "find_in_files",
        "run_background_process",
        "stop_background_process",
        "list_background_processes",
        "http_request",
    ],
}

# Prefix fallbacks so new tools are auto-categorised even before this table is
# updated. Order matters: the longest / most specific prefixes come first.
_PREFIX_FALLBACKS: list[tuple[str, str]] = [
    ("plot_interactive", "plot_interactive"),
    ("plot_static", "plot_static"),
    ("generate_custom_plotly", "plot_interactive"),
    ("generate_custom_static", "plot_static"),
]

# Categories whose workers also receive the shared data-exploration tools.
_DATA_SHARED_INTO = {"plot_interactive", "plot_static", "stats"}

# Human-readable one-liners the supervisor sees when planning.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "data": "Inspect datasets: column names, dtypes, summary statistics.",
    "plot_interactive": "Create interactive Plotly HTML charts (hover, zoom).",
    "plot_static": "Create static Matplotlib/Seaborn image charts (PNG).",
    "stats": "Statistical analysis: correlation, regression, group comparison.",
    "web": "Fetch webpages, search the web, screenshot pages.",
    "research": "arXiv, GitHub, Wikipedia and YouTube transcript lookups.",
    "system": "Shell/file/HTTP/background-process access (only if server enabled it).",
    "misc": "Uncategorised tools.",
}


def category_of(tool_name: str) -> str:
    """Return the category a tool name belongs to (falls back to prefix / misc)."""
    for category, names in TOOL_CATEGORIES.items():
        if tool_name in names:
            return category
    for prefix, category in _PREFIX_FALLBACKS:
        if tool_name.startswith(prefix):
            return category
    return "misc"


def categorize_tools(
    tools: list[dict],
    *,
    share_data_tools: bool = True,
) -> dict[str, list[dict]]:
    """
    Group MCP tool specs (dicts with a ``name`` key) by category.

    Returns an ordered mapping ``{category: [tool_spec, ...]}`` containing only
    the categories that actually have tools. When ``share_data_tools`` is True,
    the ``data`` tools are also appended to the plotting and stats workers so
    they can inspect the dataset before acting.
    """
    grouped: dict[str, list[dict]] = {}
    for tool in tools:
        name = tool.get("name", "")
        cat = category_of(name)
        grouped.setdefault(cat, []).append(tool)

    if share_data_tools and "data" in grouped:
        data_tools = grouped["data"]
        data_names = {t.get("name") for t in data_tools}
        for cat in _DATA_SHARED_INTO:
            if cat in grouped:
                existing = {t.get("name") for t in grouped[cat]}
                for dt in data_tools:
                    if dt.get("name") not in existing:
                        grouped[cat].append(dt)
        # keep data_names referenced (silences linters on unused var in edge cases)
        _ = data_names

    return grouped
