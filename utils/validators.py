from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
import numpy as np
import streamlit as st


def coerce_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> Tuple[pd.DataFrame, list[str]]:
    """Try to coerce the named columns to numeric. Returns (df, failed_cols)."""
    df = df.copy()
    failed = []
    for c in cols:
        try:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.isna().all():
                failed.append(c)
            df[c] = coerced
        except Exception:
            failed.append(c)
    return df, failed


def validate_payload_numeric_ranges(payload: dict, ranges: dict) -> list[str]:
    issues = []
    for k, r in (ranges or {}).items():
        if k not in payload:
            issues.append(f"Missing numeric column: {k}")
            continue
        try:
            v = float(payload[k])
        except Exception:
            issues.append(f"Value for {k} is not numeric")
            continue
        if v < r.get("min", -np.inf) or v > r.get("max", np.inf):
            issues.append(f"{k} outside training range [{r.get('min')}, {r.get('max')}]")
    return issues
def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature column lists.

    - returns (num_cols, cat_cols)
    - robust to None or empty DataFrame
    """
    if df is None or df.empty:
        return [], []

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


