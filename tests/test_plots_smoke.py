"""Smoke tests for plot generation: confirm output files are created."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

from ds_mcp_server._tools.plot_interactive import (
    plot_correlation_heatmap_impl as interactive_heatmap,
    plot_histogram_impl as interactive_histogram,
    plot_scatterplot_impl as interactive_scatter,
)
from ds_mcp_server._tools.plot_static import (
    plot_static_histogram_impl,
    plot_static_scatterplot_impl,
)


def _split(result: str) -> tuple[str, str]:
    assert "|||" in result, f"Unexpected result format: {result[:200]}"
    path, code = result.split("|||", 1)
    return path, code


class TestInteractivePlots:
    def test_histogram_creates_html(self, csv_path: str):
        path, code = _split(interactive_histogram(csv_path, "x", "Distribution of x"))
        assert os.path.exists(path)
        assert path.endswith(".html")
        assert "plotly" in code.lower()

    def test_scatter_creates_html(self, csv_path: str):
        path, _ = _split(interactive_scatter(csv_path, "x", "y_linear", "Scatter"))
        assert os.path.exists(path)

    def test_correlation_heatmap(self, csv_path: str):
        path, _ = _split(interactive_heatmap(csv_path, "Correlations"))
        assert os.path.exists(path)


class TestStaticPlots:
    def test_static_histogram_creates_png(self, csv_path: str):
        path, _ = _split(plot_static_histogram_impl(csv_path, "x", "Hist of x", "x"))
        assert os.path.exists(path)

    def test_static_scatter_creates_png(self, csv_path: str):
        path, _ = _split(
            plot_static_scatterplot_impl(csv_path, "x", "y_linear", "Scatter", "x", "y")
        )
        assert os.path.exists(path)


class TestPlotErrorHandling:
    def test_histogram_missing_column(self, csv_path: str):
        out = interactive_histogram(csv_path, "does_not_exist", "Title")
        assert "Error" in out
        assert "|||" not in out

    def test_scatter_missing_column(self, csv_path: str):
        out = interactive_scatter(csv_path, "x", "does_not_exist", "Title")
        assert "Error" in out
