"""Tests for _tools.viz_utils: file loading, caching, path sanitization."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from ds_mcp_server._tools import viz_utils
from ds_mcp_server._tools.viz_utils import (
    _strip_show_calls,
    generate_code_snippet,
    get_plot_path,
    load_data_safely,
    was_last_load_truncated,
)


class TestLoadDataSafely:
    def test_loads_csv(self, csv_path: str):
        df = load_data_safely(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"x", "y_linear", "category"}
        assert len(df) == 60

    def test_loads_tsv(self, tsv_path: str):
        df = load_data_safely(tsv_path)
        assert len(df) == 60
        assert "category" in df.columns

    def test_loads_json_lines(self, json_path: str):
        df = load_data_safely(json_path)
        assert len(df) == 60

    def test_loads_xlsx_streaming(self, xlsx_path: str):
        df = load_data_safely(xlsx_path)
        assert len(df) == 60
        assert set(df.columns) >= {"x", "y_linear"}

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_data_safely(str(tmp_path / "nope.csv"))

    def test_unsupported_extension_raises(self, tmp_path: Path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"junk")
        with pytest.raises(ValueError):
            load_data_safely(str(p))

    def test_truncation_flag_set_when_over_max_rows(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(viz_utils, "MAX_ROWS", 10)
        df = pd.DataFrame({"a": range(25)})
        p = tmp_path / "big.csv"
        df.to_csv(p, index=False)
        loaded = load_data_safely(str(p))
        assert len(loaded) == 10
        assert was_last_load_truncated(str(p)) is True

    def test_not_truncated_when_under_max_rows(self, csv_path: str):
        load_data_safely(csv_path)
        assert was_last_load_truncated(csv_path) is False

    def test_cache_invalidates_on_mtime_change(self, tmp_path: Path):
        p = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(p, index=False)
        first = load_data_safely(str(p))
        assert len(first) == 3

        time.sleep(0.02)
        pd.DataFrame({"a": [1, 2, 3, 4, 5]}).to_csv(p, index=False)
        new_mtime = time.time() + 1
        os.utime(p, (new_mtime, new_mtime))

        second = load_data_safely(str(p))
        assert len(second) == 5, "Cache should have been invalidated by mtime change"

    def test_latin1_fallback_on_bad_utf8(self, tmp_path: Path):
        p = tmp_path / "latin.csv"
        with open(p, "wb") as f:
            f.write(b"name,value\n")
            f.write("caf\xe9,1\n".encode("latin1"))
        df = load_data_safely(str(p))
        assert len(df) == 1
        assert df["value"].iloc[0] == 1


class TestGetPlotPath:
    def test_sanitizes_unsafe_chars(self, tmp_path: Path):
        data = tmp_path / "data.csv"
        data.write_text("a\n1\n")
        path = get_plot_path(str(data), "my plot/with $weird chars!", ext=".html")
        name = os.path.basename(path)
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
        assert all(c in allowed for c in name)
        assert name.endswith(".html")

    def test_falls_back_to_plot_when_name_empty(self, tmp_path: Path):
        data = tmp_path / "data.csv"
        data.write_text("a\n1\n")
        path = get_plot_path(str(data), "!!!@@@", ext=".png")
        assert os.path.basename(path) == "plot.png"

    def test_creates_plots_subdir(self, tmp_path: Path):
        data = tmp_path / "data.csv"
        data.write_text("a\n1\n")
        path = get_plot_path(str(data), "chart", ext=".html")
        assert os.path.isdir(os.path.join(tmp_path, "plots"))
        assert path.startswith(str(tmp_path))


class TestStripShowCalls:
    def test_removes_plt_show(self):
        code = "fig, ax = plt.subplots()\nplt.show()"
        cleaned = _strip_show_calls(code)
        assert "plt.show()" not in cleaned
        assert "plt.subplots" in cleaned

    def test_removes_fig_show(self):
        code = "fig = px.scatter(df)\nfig.show()"
        cleaned = _strip_show_calls(code)
        assert "fig.show()" not in cleaned

    def test_removes_savefig(self):
        code = "plt.plot([1,2])\nplt.savefig('bad_path.png')"
        cleaned = _strip_show_calls(code)
        assert "savefig" not in cleaned


class TestGenerateCodeSnippet:
    def test_csv_loader(self, csv_path: str):
        snippet = generate_code_snippet("fig = px.scatter(df)", csv_path)
        assert "pd.read_csv" in snippet

    def test_tsv_loader(self, tsv_path: str):
        snippet = generate_code_snippet("fig = px.scatter(df)", tsv_path)
        assert "sep='\\t'" in snippet

    def test_json_loader(self, json_path: str):
        snippet = generate_code_snippet("fig = px.scatter(df)", json_path)
        assert "read_json" in snippet
