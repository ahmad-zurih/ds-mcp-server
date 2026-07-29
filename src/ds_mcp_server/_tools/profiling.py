"""
One-shot dataset profiling via ydata-profiling.

Generates a rich, self-contained HTML report (types, distributions, missing
values, correlations, warnings) for a tabular dataset. The report path is
returned in the ``path|||code`` convention shared by the plotting tools so the
web UI renders it in an iframe and multi-agent workers register it as an
artifact. ydata-profiling is an optional, heavy dependency imported lazily.
"""
from __future__ import annotations

import os

from .viz_config import MAX_ROWS
from .viz_utils import (
    get_plot_path,
    load_data_safely,
    was_last_load_truncated,
)


def profile_dataset_impl(
    data_file_path: str,
    title: str = "Dataset Profiling Report",
    minimal: bool = True,
) -> str:
    """
    Build an interactive HTML profiling report for a dataset.

    minimal: when True (default) skips the most expensive computations
        (e.g. dynamic correlations, interactions) for a much faster report on
        wide datasets. Set False for the full report.
    """
    if not data_file_path or not os.path.exists(data_file_path):
        return f"Error: data file not found at: {data_file_path}"

    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        return (
            "ydata-profiling is not installed. Install the profiling extra:\n"
            "  pip install 'ds-mcp-server[profiling]'"
        )

    try:
        df = load_data_safely(data_file_path)
    except Exception as exc:
        return f"Error loading dataset: {exc}"

    if df.empty:
        return "Error: dataset is empty; nothing to profile."

    truncation_note = ""
    if was_last_load_truncated(data_file_path):
        truncation_note = (
            f" (profiled the first {MAX_ROWS:,} rows for memory safety)"
        )

    try:
        profile = ProfileReport(
            df,
            title=title,
            minimal=minimal,
            explorative=not minimal,
            progress_bar=False,
        )
        plot_path = get_plot_path(data_file_path, "profile_report", ext=".html")
        profile.to_file(plot_path)
    except Exception as exc:
        return f"Error generating profiling report: {exc}"

    code = (
        "import pandas as pd\n"
        "from ydata_profiling import ProfileReport\n\n"
        f"df = pd.read_csv('your_data.csv')\n"
        f"profile = ProfileReport(df, title={title!r}, minimal={minimal})\n"
        "profile.to_file('profile_report.html')"
    )

    summary = (
        f"Generated profiling report for {os.path.basename(data_file_path)}: "
        f"{df.shape[0]:,} rows x {df.shape[1]} columns{truncation_note}."
    )
    return f"{plot_path}|||{summary}\n\n{code}"
