"""Tests for _tools.stats_analysis."""
from __future__ import annotations

import pandas as pd

from ds_mcp_server._tools.stats_analysis import (
    rank_target_correlations_impl,
    run_correlation_impl,
    run_group_comparison_impl,
    run_linear_regression_impl,
)


class TestRunCorrelation:
    def test_near_perfect_positive_correlation(self, csv_path: str):
        out = run_correlation_impl(csv_path, "x", "y_linear", method="pearson")
        assert "Correlation Analysis" in out
        assert "```python" in out
        assert "pg.corr" in out

    def test_missing_column_reports_error(self, csv_path: str):
        out = run_correlation_impl(csv_path, "x", "not_a_column")
        assert "Error" in out


class TestGroupComparison:
    def test_ttest_for_two_groups(self, csv_path: str):
        out = run_group_comparison_impl(csv_path, "x", "binary_text")
        assert "T-test" in out
        assert "```python" in out

    def test_anova_for_three_plus_groups(self, csv_path: str):
        out = run_group_comparison_impl(csv_path, "x", "category")
        assert "ANOVA" in out
        assert "```python" in out

    def test_single_group_returns_error(self, tmp_path):
        p = tmp_path / "single.csv"
        pd.DataFrame({"g": ["A"] * 10, "v": range(10)}).to_csv(p, index=False)
        out = run_group_comparison_impl(str(p), "v", "g")
        assert "Error" in out


class TestLinearRegression:
    def test_ols_on_linear_relationship(self, csv_path: str):
        out = run_linear_regression_impl(csv_path, "y_linear", ["x"])
        assert "R-squared" in out
        assert "```python" in out

    def test_string_predictor_is_split(self, csv_path: str):
        out = run_linear_regression_impl(csv_path, "y_noisy", "x,z")  # type: ignore[arg-type]
        assert "R-squared" in out

    def test_missing_column_reports_error(self, csv_path: str):
        out = run_linear_regression_impl(csv_path, "y_linear", ["not_a_column"])
        assert "Error" in out and "not_a_column" in out


class TestRankTargetCorrelations:
    def test_ranks_all_numeric_features(self, csv_path: str):
        out = rank_target_correlations_impl(csv_path, "y_linear")
        assert "Correlation Ranking" in out
        assert "x" in out

    def test_binary_text_target_is_auto_encoded(self, csv_path: str):
        out = rank_target_correlations_impl(csv_path, "binary_text")
        assert "Correlation Ranking" in out
        assert "val_map" in out or "Binary encode" in out

    def test_non_binary_text_target_reports_error(self, csv_path: str):
        out = rank_target_correlations_impl(csv_path, "category")
        assert "Error" in out
        assert "binary" in out.lower()

    def test_missing_target_reports_error(self, csv_path: str):
        out = rank_target_correlations_impl(csv_path, "not_a_column")
        assert "Error" in out
