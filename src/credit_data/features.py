from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def add_lags(df: pd.DataFrame, by: List[str], date_col: str, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Add lagged columns for specified numeric columns.

    Parameters:
    - df: panel DataFrame
    - by: keys to group by (e.g., ["loan_id"])
    - date_col: temporal column name (e.g., "asof_month")
    - cols: columns to lag
    - lags: list of lag sizes in months
    """
    out = df.sort_values(by + [date_col]).copy()
    for c in cols:
        for l in lags:
            out[f"{c}_lag{l}"] = out.groupby(by)[c].shift(l)
    return out


def add_leads(df: pd.DataFrame, by: List[str], date_col: str, cols: List[str], leads: List[int]) -> pd.DataFrame:
    """Add lead columns for specified columns (e.g., labels like default_flag)."""
    out = df.sort_values(by + [date_col]).copy()
    for c in cols:
        for L in leads:
            out[f"{c}_lead{L}"] = out.groupby(by)[c].shift(-L)
    return out


def add_rolling(df: pd.DataFrame, by: List[str], date_col: str, cols: List[str], windows: List[int]) -> pd.DataFrame:
    """Add rolling mean and std for specified columns and windows."""
    out = df.sort_values(by + [date_col]).copy()
    for c in cols:
        g = out.groupby(by)[c]
        for w in windows:
            out[f"{c}_roll{w}_mean"] = g.rolling(w, min_periods=1).mean().reset_index(level=by, drop=True)
            out[f"{c}_roll{w}_std"] = g.rolling(w, min_periods=1).std().reset_index(level=by, drop=True)
    return out


def add_interactions(df: pd.DataFrame, interactions: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Add interaction terms as product of pairs of columns.

    interactions: list of tuples (col_a, col_b, new_name)
    """
    out = df.copy()
    for a, b, name in interactions:
        out[name] = out[a] * out[b]
    return out
