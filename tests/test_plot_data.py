"""Tests for _tools.plot_data column summary helpers."""
from __future__ import annotations

from ds_mcp_server._tools.plot_data import (
    get_all_columns_summary_impl,
    get_column_summary_impl,
)


def test_numeric_column_summary(csv_path: str):
    out = get_column_summary_impl(csv_path, "x")
    assert "Numeric Column 'x'" in out
    assert "Min=" in out and "Max=" in out and "Mean=" in out


def test_categorical_column_summary(csv_path: str):
    out = get_column_summary_impl(csv_path, "category")
    assert "Categorical Column 'category'" in out
    assert "3 unique" in out


def test_missing_column_returns_error(csv_path: str):
    out = get_column_summary_impl(csv_path, "does_not_exist")
    assert "Error" in out and "does_not_exist" in out


def test_missing_file_returns_error(tmp_path):
    out = get_column_summary_impl(str(tmp_path / "nope.csv"), "x")
    assert "Error" in out


def test_all_columns_summary_lists_types(csv_path: str):
    out = get_all_columns_summary_impl(csv_path)
    assert "Numeric columns" in out
    assert "Categorical columns" in out
    assert "x" in out
    assert "category" in out
