"""
Utility functions for the AI Visualization Engine.
Handles file I/O, memory-safe data loading, and path generation.
"""

import os
import re
import sys
from functools import lru_cache

import pandas as pd

from .viz_config import MAX_ROWS

# Tracks whether the most recent load_data_safely call truncated the file at MAX_ROWS.
# Read by tools so they can surface a warning to the agent / UI.
LAST_LOAD_TRUNCATED: dict[str, bool] = {}


def _read_csv_with_fallback(file_path: str, sep: str = ",", nrows: int | None = None) -> pd.DataFrame:
    """Try UTF-8 first (most common), fall back to latin1 to avoid silent mangling."""
    try:
        return pd.read_csv(file_path, sep=sep, nrows=nrows, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, sep=sep, nrows=nrows, encoding="latin1")


def _read_excel_safely(file_path: str, max_rows: int) -> pd.DataFrame:
    """
    Read an Excel file using openpyxl in read-only streaming mode for .xlsx files,
    which avoids loading the entire workbook into memory at once.
    Falls back to standard pd.read_excel for .xls files (xlrd doesn't support streaming).
    """
    if file_path.lower().endswith(".xls"):
        return pd.read_excel(file_path, nrows=max_rows)

    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.active
        row_iter = ws.iter_rows(values_only=True)
        header = next(row_iter, None)
        if header is None:
            wb.close()
            return pd.DataFrame()
        data = []
        for row in row_iter:
            if len(data) >= max_rows:
                break
            data.append(row)
        wb.close()
        return pd.DataFrame(data, columns=header)
    except ImportError:
        return pd.read_excel(file_path, nrows=max_rows)


def save_data_file(file_bytes: bytes, file_name: str, run_dir: str) -> str:
    """
    Save the uploaded file bytes to a temporary workspace directory.

    Args:
        file_bytes: The raw bytes of the uploaded file.
        file_name: The original name of the uploaded file.
        run_dir: The directory where the file should be saved.

    Returns:
        The absolute path to the saved file.
    """
    file_extension = os.path.splitext(file_name)[1].lower()
    data_file_path = os.path.join(run_dir, f"uploaded_data{file_extension}")
    
    with open(data_file_path, "wb") as f:
        f.write(file_bytes)
        
    return data_file_path


def get_fast_data_preview(file_path: str, file_name: str, nrows: int = 5) -> pd.DataFrame | None:
    """
    Read only the first few rows of a dataset directly from disk to save memory.
    This is used by the UI to quickly preview the data before passing it to the AI.

    Args:
        file_path: The path to the saved data file.
        file_name: The original name of the file (used for extension checking).
        nrows: The number of rows to read.

    Returns:
        A pandas DataFrame containing the top rows, or None if reading fails.
    """
    lower_name = file_name.lower()
    try:
        if lower_name.endswith(".csv"):
            return _read_csv_with_fallback(file_path, nrows=nrows)
        if lower_name.endswith(".tsv"):
            return _read_csv_with_fallback(file_path, sep="\t", nrows=nrows)
        if lower_name.endswith((".xls", ".xlsx")):
            return _read_excel_safely(file_path, nrows)
        if lower_name.endswith(".json"):
            try:
                return pd.read_json(file_path, lines=True, nrows=nrows)
            except (ValueError, TypeError):
                return pd.read_json(file_path).head(nrows)
        return None
    except Exception as e:
        print(f"Error reading data preview for {file_name}: {e}", file=sys.stderr)
        return None


@lru_cache(maxsize=8)
def _load_data_cached(file_path: str, mtime: float, size: int) -> pd.DataFrame:
    """Cache-keyed loader. mtime+size invalidate the cache when the file changes."""
    lower_name = file_path.lower()
    if lower_name.endswith(".csv"):
        df = _read_csv_with_fallback(file_path, nrows=MAX_ROWS + 1)
    elif lower_name.endswith(".tsv"):
        df = _read_csv_with_fallback(file_path, sep="\t", nrows=MAX_ROWS + 1)
    elif lower_name.endswith((".xls", ".xlsx")):
        df = _read_excel_safely(file_path, MAX_ROWS + 1)
    elif lower_name.endswith(".json"):
        try:
            df = pd.read_json(file_path, lines=True, nrows=MAX_ROWS + 1)
        except (ValueError, TypeError):
            df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {os.path.basename(file_path)}")

    truncated = len(df) > MAX_ROWS
    LAST_LOAD_TRUNCATED[file_path] = truncated
    if truncated:
        df = df.iloc[:MAX_ROWS].copy()
    return df


def load_data_safely(file_path: str) -> pd.DataFrame:
    """
    Load data from disk with safety limits to prevent Out-Of-Memory (OOM) crashes.
    Cached by (path, mtime, size) so repeated tool calls in the same MCP session
    don't re-parse the file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file type is unsupported.
        RuntimeError: If pandas fails to read the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    try:
        stat = os.stat(file_path)
        return _load_data_cached(file_path, stat.st_mtime, stat.st_size)
    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load data safely: {str(e)}")


def was_last_load_truncated(file_path: str) -> bool:
    """Returns True if the last load_data_safely call for this path hit MAX_ROWS."""
    return LAST_LOAD_TRUNCATED.get(file_path, False)


def get_plot_path(data_file_path: str, plot_name: str, ext: str = ".json") -> str:
    """
    Generate a safe, unique file path for saving a generated plot.

    Args:
        data_file_path: The path to the source data file (used to locate the run directory).
        plot_name: The descriptive name of the plot.
        ext: The file extension for the plot (e.g., '.json', '.png').

    Returns:
        The absolute path where the plot should be saved.
    """
    run_dir = os.path.dirname(data_file_path)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Sanitize the plot name: Keep only alphanumeric characters, underscores, and hyphens
    safe_plot_name = re.sub(r"[^\w\-]", "", plot_name.replace(" ", "_")).rstrip("_")
    if not safe_plot_name:
        safe_plot_name = "plot"

    return os.path.join(plot_dir, f"{safe_plot_name}{ext}")


def _strip_show_calls(code: str) -> str:
    """
    Remove standalone display/save calls from model-generated code before exec.

    Strips:
      - fig.show() / plt.show()   — clear the active figure on non-interactive backends
      - plt.savefig(...)           — model may save to an arbitrary/wrong path;
                                     the tool always saves explicitly afterwards
    """
    # Remove show() calls
    code = re.sub(r"^\s*(fig|plt)\.show\(\)\s*$", "", code, flags=re.MULTILINE)
    # Remove plt.savefig(...) — matches single-line calls (balanced or not)
    code = re.sub(r"^\s*plt\.savefig\([^\n]*\)\s*$", "", code, flags=re.MULTILINE)
    return code.rstrip()


def generate_code_snippet(plot_code: str, data_file_path: str | None = None) -> str:
    """
    Format the raw plot generation code into a complete, runnable script string.

    Args:
        plot_code: The core logic used to generate the plot.
        data_file_path: Optional source file path; its extension is used to choose
            the appropriate pandas reader in the generated snippet.

    Returns:
        A formatted Python script string including imports and data loading.
    """
    loader = "df = pd.read_csv('your_data.csv')"
    if data_file_path:
        ext = os.path.splitext(data_file_path)[1].lower()
        if ext == ".tsv":
            loader = "df = pd.read_csv('your_data.tsv', sep='\\t')"
        elif ext in (".xls", ".xlsx"):
            loader = f"df = pd.read_excel('your_data{ext}')"
        elif ext == ".json":
            loader = "df = pd.read_json('your_data.json')"

    clean = _strip_show_calls(plot_code)
    return (
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n\n"
        "# Load Data\n"
        f"{loader}\n\n"
        "# Generate Plot\n"
        f"{clean}\n"
        "fig.show()"
    )