"""
Model Context Protocol (MCP) Server for the AI Visualization Engine.
Registers both interactive (Plotly) and static (Matplotlib) tools.

The optional "system tools" group (arbitrary shell execution, file read/write/patch,
background processes, HTTP requests, and file-search) is DISABLED by default because
it exposes remote-code-execution style capabilities to any connected LLM client.
Enable it explicitly by either:

  * setting the environment variable  DS_MCP_ENABLE_SYSTEM_TOOLS=1
  * or passing  --enable-system-tools  to the `ds-mcp-server` CLI (which sets the env
    var before importing this module).

When enabled, a warning listing every registered dangerous tool is printed to stderr
at startup. See the README for the full risk breakdown.
"""

import logging
import os
import sys


def _system_tools_enabled() -> bool:
    """Return True if the opt-in env var enables the dangerous system tools group."""
    val = os.environ.get("DS_MCP_ENABLE_SYSTEM_TOOLS", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _unrestricted_exec_enabled() -> bool:
    """Return True if the user opted out of the custom-plot sandbox."""
    val = os.environ.get("DS_MCP_ALLOW_UNRESTRICTED_EXEC", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _emit_unrestricted_exec_warning() -> None:
    """Loud stderr warning when the sandbox around custom-plot exec is disabled."""
    banner = "!" * 72
    lines = [
        "",
        banner,
        "!! ds-mcp-server: CUSTOM-PLOT SANDBOX IS DISABLED",
        "!!",
        "!! generate_custom_plotly and generate_custom_static_plot will exec the",
        "!! LLM's Python code with FULL privileges — imports, open(), eval(),",
        "!! subprocess, etc. are all allowed.",
        "!!",
        "!! A hostile dataset, webpage, or file that the model reads can inject",
        "!! instructions that trigger arbitrary code execution on this machine.",
        "!!",
        "!! To re-enable the sandbox, unset DS_MCP_ALLOW_UNRESTRICTED_EXEC or",
        "!! omit the --allow-unrestricted-exec flag.",
        banner,
        "",
    ]
    print("\n".join(lines), file=sys.stderr, flush=True)


from mcp.server.fastmcp import FastMCP

from ds_mcp_server._tools.plot_data import get_all_columns_summary_impl, get_column_summary_impl
from ds_mcp_server._tools.plot_interactive import (
    generate_custom_plotly_impl,
    plot_barchart_impl as interactive_barchart,
    plot_boxplot_impl as interactive_boxplot,
    plot_correlation_heatmap_impl as interactive_correlation_heatmap,
    plot_histogram_impl as interactive_histogram,
    plot_lineplot_impl as interactive_lineplot,
    plot_scatter_matrix_impl as interactive_scatter_matrix,
    plot_scatterplot_impl as interactive_scatterplot,
)
from ds_mcp_server._tools.plot_static import (
    generate_custom_static_plot_impl,
    plot_static_barchart_impl,
    plot_static_boxplot_impl,
    plot_static_correlation_heatmap_impl,
    plot_static_histogram_impl,
    plot_static_lineplot_impl,
    plot_static_pairplot_impl,
    plot_static_scatterplot_impl,
    plot_static_wordcloud_impl,
)

from ds_mcp_server._tools.web_tools import (
    fetch_webpage_impl,
    screenshot_webpage_impl,
    search_web_impl,
)

# System tools are imported lazily below only if opt-in is set — see _system_tools_enabled().

from ds_mcp_server._tools.stats_analysis import (
    run_correlation_impl,
    run_group_comparison_impl,
    run_linear_regression_impl,
    rank_target_correlations_impl,
)

# Configure strict logging to prevent interference with stdout/stderr JSON-RPC
logging.basicConfig(level=logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)

mcp = FastMCP("Data Science MCP Server")


# --- INTERACTIVE TOOLS (Plotly) ---

@mcp.tool()
def plot_interactive_histogram(
    data_file_path: str, column: str, title: str, color_column: str | None = None
) -> str:
    """Generates a web-ready interactive Plotly histogram."""
    return interactive_histogram(data_file_path, column, title, color_column)


@mcp.tool()
def plot_interactive_scatterplot(
    data_file_path: str, x_column: str, y_column: str, title: str, color_column: str | None = None
) -> str:
    """Generates a web-ready interactive Plotly scatter plot."""
    return interactive_scatterplot(data_file_path, x_column, y_column, title, color_column)


@mcp.tool()
def plot_interactive_boxplot(
    data_file_path: str, x_column: str, y_column: str, title: str, color_column: str | None = None
) -> str:
    """Generates a web-ready interactive Plotly box plot."""
    return interactive_boxplot(data_file_path, x_column, y_column, title, color_column)


@mcp.tool()
def plot_interactive_lineplot(
    data_file_path: str, x_column: str, y_column: str, title: str, color_column: str | None = None
) -> str:
    """Generates a web-ready interactive Plotly line plot."""
    return interactive_lineplot(data_file_path, x_column, y_column, title, color_column)


@mcp.tool()
def plot_interactive_correlation_heatmap(
    data_file_path: str,
    title: str,
    method: str = "pearson",
    column_filter: str = "",
) -> str:
    """
    Generates an interactive Plotly correlation heatmap.
    Use this to visualize relationships between numeric features.
    method must be 'pearson' or 'spearman'.
    column_filter: optional comma-separated column names or suffix patterns (e.g. '_mean')
        to restrict the heatmap to a subset of columns. Leave empty for all numeric columns.
    """
    return interactive_correlation_heatmap(data_file_path, title, method, column_filter or None)


@mcp.tool()
def plot_interactive_barchart(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
    aggregation: str = "mean",
) -> str:
    """
    Generates an interactive Plotly grouped bar chart.
    x_column: categorical column for the x-axis groups.
    y_column: numeric column to aggregate.
    aggregation: how to aggregate y per group — 'mean' (default), 'sum', 'count', or 'median'.
    color_column: optional column to split bars by colour.
    """
    return interactive_barchart(data_file_path, x_column, y_column, title, color_column, aggregation)


@mcp.tool()
def plot_interactive_scatter_matrix(
    data_file_path: str,
    columns: str,
    title: str,
    color_column: str | None = None,
) -> str:
    """
    Generates an interactive Plotly scatter matrix (pair plot equivalent).
    columns: comma-separated list of numeric column names (e.g. 'radius_mean,texture_mean,area_mean').
    color_column: optional categorical column to colour points by (e.g. 'diagnosis').
    """
    return interactive_scatter_matrix(data_file_path, columns, title, color_column)


@mcp.tool()
def generate_custom_plotly(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """Executes custom Python code (px, pd) to generate complex Plotly charts."""
    return generate_custom_plotly_impl(data_file_path, python_code, plot_filename_keyword)


@mcp.tool()
def get_all_columns_summary(data_file_path: str) -> str:
    """
    Returns a compact schema of ALL columns in one call: column names grouped by type
    (numeric, categorical, datetime). Categorical columns also show their unique values.
    Call this FIRST to understand the dataset structure, then call plot or stats tools.
    """
    return get_all_columns_summary_impl(data_file_path)


@mcp.tool()
def get_column_summary(data_file_path: str, column: str) -> str:
    """
    Analyzes a specific column in the dataset and returns a statistical summary.
    Use this for a deep dive into one column after using get_all_columns_summary.
    """
    return get_column_summary_impl(data_file_path, column)


# --- STATIC TOOLS (Matplotlib/Seaborn) ---

@mcp.tool()
def plot_static_histogram(
    data_file_path: str, column: str, title: str, x_label: str
) -> str:
    """Generates a static Matplotlib/Seaborn histogram (for papers/publications)."""
    return plot_static_histogram_impl(data_file_path, column, title, x_label)


@mcp.tool()
def plot_static_scatterplot(
    data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str, hue_column: str | None = None
) -> str:
    """Generates a static Matplotlib/Seaborn scatter plot (for papers/publications)."""
    return plot_static_scatterplot_impl(data_file_path, x_column, y_column, title, x_label, y_label, hue_column)


@mcp.tool()
def plot_static_boxplot(
    data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str
) -> str:
    """Generates a static Matplotlib/Seaborn box plot (for papers/publications)."""
    return plot_static_boxplot_impl(data_file_path, x_column, y_column, title, x_label, y_label)


@mcp.tool()
def plot_static_lineplot(
    data_file_path: str, x_column: str, y_column: str, title: str, x_label: str, y_label: str, hue_column: str | None = None
) -> str:
    """Generates a static Matplotlib/Seaborn line plot (for papers/publications)."""
    return plot_static_lineplot_impl(data_file_path, x_column, y_column, title, x_label, y_label, hue_column)


@mcp.tool()
def plot_static_barchart(
    data_file_path: str,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    hue_column: str | None = None,
    aggregation: str = "mean",
) -> str:
    """
    Generates a static Seaborn bar chart (for papers/publications).
    x_column: categorical column for the x-axis groups.
    y_column: numeric column to aggregate.
    aggregation: how to aggregate y per group — 'mean' (default), 'sum', 'count', or 'median'.
    hue_column: optional column to split bars by colour.
    """
    return plot_static_barchart_impl(data_file_path, x_column, y_column, title, x_label, y_label, hue_column, aggregation)


@mcp.tool()
def generate_custom_static_plot(
    data_file_path: str, python_code: str, plot_filename_keyword: str
) -> str:
    """Executes custom Python code (plt, sns, pd) to generate complex static charts."""
    return generate_custom_static_plot_impl(data_file_path, python_code, plot_filename_keyword)


@mcp.tool()
def plot_static_pairplot(
    data_file_path: str,
    columns: str,
    title: str = "Pair Plot",
    hue_column: str = "",
) -> str:
    """
    Generates a Seaborn pair plot (scatter matrix) for the specified columns.
    Use this for multi-feature distribution and correlation exploration.
    columns: comma-separated list of numeric column names (e.g. 'radius_mean,texture_mean,area_mean').
    hue_column: optional categorical column name to colour points by (e.g. 'diagnosis'). Leave empty if not needed.
    """
    return plot_static_pairplot_impl(data_file_path, columns, title, hue_column or None)


@mcp.tool()
def plot_static_wordcloud(
    data_file_path: str,
    text_column: str,
    title: str = "Word Cloud",
    extra_stopwords: str | None = None,
) -> str:
    """
    Generates a static Word Cloud image from a column containing text data.
    Use this when the user wants to visualize the most frequent terms in a dataset.
    extra_stopwords: optional comma-separated words to exclude (e.g. "said,also,one").
    """
    return plot_static_wordcloud_impl(data_file_path, text_column, title, extra_stopwords)


@mcp.tool()
def plot_static_correlation_heatmap(
    data_file_path: str,
    title: str,
    method: str = "pearson",
    column_filter: str = "",
) -> str:
    """
    Generates a publication-ready Seaborn correlation heatmap.
    Use this when the user explicitly asks for static or publication figures.
    method must be 'pearson' or 'spearman'.
    column_filter: optional comma-separated column names or suffix patterns (e.g. '_mean')
        to restrict the heatmap to a subset of columns. Leave empty for all numeric columns.
    """
    return plot_static_correlation_heatmap_impl(data_file_path, title, method, column_filter or None)


# --- STATISTICAL TOOLS ---

@mcp.tool()
def run_correlation(
    data_file_path: str, x_column: str, y_column: str, method: str = "pearson"
) -> str:
    """
    Computes statistical correlation (pearson, spearman) between two numeric columns.
    Use this to mathematically verify relationships before plotting scatterplots.
    """
    return run_correlation_impl(data_file_path, x_column, y_column, method)


@mcp.tool()
def run_group_comparison(
    data_file_path: str, target_col: str, group_col: str
) -> str:
    """
    Performs T-tests (2 groups) or ANOVA (>2 groups) to see if a numeric variable 
    (target_col) differs significantly across categories (group_col).
    Use this before generating boxplots.
    """
    return run_group_comparison_impl(data_file_path, target_col, group_col)


@mcp.tool()
def run_linear_regression(
    data_file_path: str, target_col: str, predictor_cols: list[str]
) -> str:
    """
    Runs an OLS Linear Regression. 
    target_col is the dependent variable (Y).
    predictor_cols is a list of independent variables (X). 
    CRITICAL: predictor_cols MUST be a valid JSON array of strings, e.g., ["col1", "col2"].
    """
    return run_linear_regression_impl(data_file_path, target_col, predictor_cols)

@mcp.tool()
def rank_target_correlations(
    data_file_path: str, target_col: str, method: str = "pearson"
) -> str:
    """
    Calculates and ranks the correlation between a single target column and all other 
    numeric columns in the dataset at once. Use this tool when the user wants to rank, 
    sort, or find top features related to a specific outcome column like diagnosis.
    """
    return rank_target_correlations_impl(data_file_path, target_col, method)



# --- SYSTEM / CODER TOOLS (opt-in, DISABLED by default) ---
#
# These tools grant the connected LLM the ability to execute arbitrary shell
# commands, read and overwrite files anywhere the process can reach, spawn
# long-lived background processes, and issue arbitrary HTTP requests. That is
# effectively remote-code-execution equivalent power. They are registered ONLY
# when the user explicitly opts in via the DS_MCP_ENABLE_SYSTEM_TOOLS env var
# (or the --enable-system-tools CLI flag, which sets the same env var).
#
# See README section "Optional system tools" for the full risk breakdown and
# recommended sandboxing practices.

_DANGEROUS_TOOL_NAMES: tuple[str, ...] = (
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
)


def _register_system_tools() -> None:
    """Register the opt-in system/coder tools on the global `mcp` instance."""
    from ds_mcp_server._tools.system_tools import (
        find_in_files_impl,
        http_request_impl,
        list_background_processes_impl,
        list_directory_impl,
        patch_file_impl,
        read_file_impl,
        run_background_process_impl,
        run_shell_command_impl,
        stop_background_process_impl,
        write_file_impl,
    )

    @mcp.tool()
    def patch_file(file_path: str, old_str: str, new_str: str) -> str:
        """
        Find the unique occurrence of old_str in a file and replace it with new_str.
        PREFER this over write_file for edits to existing files — only send the changed part.
        old_str must match EXACTLY, including whitespace and indentation, and must be unique in the file.
        Returns the line number of the edit on success, or an error with a file preview to help you correct old_str.
        """
        return patch_file_impl(file_path, old_str, new_str)

    @mcp.tool()
    def run_background_process(command: str, label: str, working_dir: str | None = None) -> str:
        """
        Start a long-running process (e.g. a web server) in the background and track it by label.
        Use a short descriptive label like "streamlit-app" or "flask-server".
        After starting, use http_request to verify it is responding. Use stop_background_process to shut it down.
        working_dir: optional absolute path to run the command in.
        """
        return run_background_process_impl(command, label, working_dir)

    @mcp.tool()
    def stop_background_process(label: str) -> str:
        """
        Stop a background process previously started with run_background_process.
        label: the label you used when starting the process.
        """
        return stop_background_process_impl(label)

    @mcp.tool()
    def list_background_processes() -> str:
        """
        List all background processes currently tracked (with their status and PIDs).
        """
        return list_background_processes_impl()

    @mcp.tool()
    def http_request(url: str, method: str = "GET", body: str = "") -> str:
        """
        Make an HTTP request and return the status code + response body (up to 4 KB).
        Use this to test running web servers after starting them with run_background_process.
        method: GET, POST, PUT, DELETE, PATCH
        body: optional JSON string for POST/PUT requests.
        """
        return http_request_impl(url, method, body)

    @mcp.tool()
    def find_in_files(pattern: str, directory: str, file_glob: str = "*.py") -> str:
        """
        Search file contents for a regex pattern across a directory tree (like grep -rn).
        Returns matching lines with file:line references (max 60 results).
        Use this to find function definitions, imports, error messages, or any text in a codebase.
        pattern: a regular expression (e.g. "def train", "import pandas", "TODO")
        directory: root directory to search from
        file_glob: filter by filename pattern, e.g. "*.py", "*.ts", "*.json", "*" for all files
        """
        return find_in_files_impl(pattern, directory, file_glob)

    @mcp.tool()
    def run_shell_command(command: str, working_dir: str | None = None) -> str:
        """
        Execute a shell command and return its stdout/stderr output.
        Use this to create directories, install packages, run scripts, check output, etc.
        working_dir: optional absolute path to run the command in (defaults to home directory).
        """
        return run_shell_command_impl(command, working_dir)

    @mcp.tool()
    def read_file(file_path: str) -> str:
        """
        Read and return the full text content of a file.
        Use this to inspect existing code, configs, logs, or any text file before editing.
        """
        return read_file_impl(file_path)

    @mcp.tool()
    def write_file(file_path: str, content: str) -> str:
        """
        Write (or overwrite) a file with the given content.
        Parent directories are created automatically.
        Always write the COMPLETE file content, this replaces the entire file.
        """
        return write_file_impl(file_path, content)

    @mcp.tool()
    def list_directory(dir_path: str) -> str:
        """
        List the contents of a directory (one level deep).
        Use this to explore project structure before reading or writing files.
        """
        return list_directory_impl(dir_path)


def _emit_system_tools_warning() -> None:
    """Print a loud stderr warning listing every dangerous tool that was enabled."""
    banner = "!" * 72
    lines = [
        "",
        banner,
        "!! ds-mcp-server: OPTIONAL SYSTEM TOOLS ARE ENABLED",
        "!!",
        "!! The connected LLM can now execute arbitrary code on this machine via:",
    ]
    for name in _DANGEROUS_TOOL_NAMES:
        lines.append(f"!!   - {name}")
    lines.extend([
        "!!",
        "!! These tools can run shell commands, overwrite files anywhere the",
        "!! process has permission to write, spawn long-lived background processes,",
        "!! and make arbitrary outbound HTTP requests (SSRF risk).",
        "!!",
        "!! Only keep this enabled inside a sandbox you trust (Docker, WSL, VM,",
        "!! or a dedicated user account). To disable, unset DS_MCP_ENABLE_SYSTEM_TOOLS",
        "!! or omit the --enable-system-tools flag.",
        banner,
        "",
    ])
    print("\n".join(lines), file=sys.stderr, flush=True)


def _emit_system_tools_hint() -> None:
    """One-liner shown when system tools are NOT enabled, so users know they exist."""
    print(
        "[ds-mcp-server] System/coder tools disabled. "
        "Set DS_MCP_ENABLE_SYSTEM_TOOLS=1 (or pass --enable-system-tools) to enable "
        "shell/file/HTTP tools. Only enable inside a sandbox.",
        file=sys.stderr,
        flush=True,
    )


if _system_tools_enabled():
    _register_system_tools()
    _emit_system_tools_warning()
else:
    _emit_system_tools_hint()

if _unrestricted_exec_enabled():
    _emit_unrestricted_exec_warning()


# --- WEB / INTERNET TOOLS ---

@mcp.tool()
def fetch_webpage(url: str) -> str:
    """
    Fetch a webpage and return structured content: title, meta description, navigation items,
    page headings, CSS color palette, font families, and main page text (up to 4000 chars).
    Use this to research a site before cloning its design, extract information, or understand
    its structure. Pair with screenshot_webpage to also see how it looks visually.
    """
    return fetch_webpage_impl(url)


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the internet using DuckDuckGo and return titles, URLs, and text snippets.
    No API key required. Use this to find documentation, discover libraries, look up
    best practices, or research any topic before starting a coding task.
    max_results: number of results to return (1-10, default 5).
    """
    return search_web_impl(query, max_results)


@mcp.tool()
def screenshot_webpage(url: str, save_path: str | None = None) -> str:
    """
    Take a 1440x900 screenshot of a webpage using headless Chromium and save it as a PNG.
    Returns the file path of the saved screenshot. Useful for visually inspecting a
    site's appearance and layout before cloning its design.
    save_path: optional absolute path for the PNG; auto-generated if omitted.
    Requires playwright: pip install playwright && playwright install chromium
    """
    return screenshot_webpage_impl(url, save_path)
