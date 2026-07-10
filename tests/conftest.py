"""Shared fixtures for the ds-mcp-server test suite."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Deterministic dataset used across many tests."""
    rng = np.random.default_rng(42)
    n = 60
    x = np.linspace(0.0, 10.0, n)
    df = pd.DataFrame(
        {
            "x": x,
            "y_linear": 2.0 * x + 1.0 + rng.normal(0, 0.01, n),
            "y_noisy": 2.0 * x + rng.normal(0, 3.0, n),
            "z": rng.normal(0, 1, n),
            "category": ["A", "B", "C"] * (n // 3),
            "binary_text": ["yes" if v > 5 else "no" for v in x],
        }
    )
    return df


@pytest.fixture
def csv_path(tmp_path: Path, sample_df: pd.DataFrame) -> str:
    p = tmp_path / "data.csv"
    sample_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def tsv_path(tmp_path: Path, sample_df: pd.DataFrame) -> str:
    p = tmp_path / "data.tsv"
    sample_df.to_csv(p, index=False, sep="\t")
    return str(p)


@pytest.fixture
def json_path(tmp_path: Path, sample_df: pd.DataFrame) -> str:
    p = tmp_path / "data.json"
    with open(p, "w", encoding="utf-8") as f:
        for row in sample_df.to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")
    return str(p)


@pytest.fixture
def xlsx_path(tmp_path: Path, sample_df: pd.DataFrame) -> str:
    pytest.importorskip("openpyxl")
    p = tmp_path / "data.xlsx"
    sample_df.to_excel(p, index=False, engine="openpyxl")
    return str(p)


@pytest.fixture(autouse=True)
def _clear_load_cache():
    """Reset the module-level lru_cache and truncation dict before each test."""
    from ds_mcp_server._tools import viz_utils

    viz_utils._load_data_cached.cache_clear()
    viz_utils.LAST_LOAD_TRUNCATED.clear()
    yield
    viz_utils._load_data_cached.cache_clear()
    viz_utils.LAST_LOAD_TRUNCATED.clear()
