from __future__ import annotations

import pandas as pd


def add_lags(df: pd.DataFrame, by: list[str], date_col: str, cols: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.sort_values(by + [date_col]).copy()
    for c in cols:
        for l in lags:
            out[f"{c}_lag{l}"] = out.groupby(by)[c].shift(l)
    return out


def add_leads(df: pd.DataFrame, by: list[str], date_col: str, cols: list[str], leads: list[int]) -> pd.DataFrame:
    out = df.sort_values(by + [date_col]).copy()
    for c in cols:
        for L in leads:
            out[f"{c}_lead{L}"] = out.groupby(by)[c].shift(-L)
    return out


def add_rolling(df: pd.DataFrame, by: list[str], date_col: str, cols: list[str], windows: list[int]) -> pd.DataFrame:
    out = df.sort_values(by + [date_col]).copy()
    for c in cols:
        g = out.groupby(by)[c]
        for w in windows:
            out[f"{c}_roll{w}_mean"] = g.rolling(w, min_periods=1).mean().reset_index(level=by, drop=True)
            out[f"{c}_roll{w}_std"] = g.rolling(w, min_periods=1).std().reset_index(level=by, drop=True)
    return out


def add_interactions(df: pd.DataFrame, interactions: list[tuple[str, str, str]]) -> pd.DataFrame:
    out = df.copy()
    for a, b, name in interactions:
        out[name] = out[a] * out[b]
    return out
