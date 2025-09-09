from __future__ import annotations

import numpy as np
import pandas as pd


def distribution_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = []
    for c in columns:
        s = df[c].dropna()
        out.append(
            {
                "column": c,
                "count": int(s.shape[0]),
                "mean": float(s.mean())
                if np.issubdtype(s.dtype, np.number)
                else np.nan,
                "std": float(s.std()) if np.issubdtype(s.dtype, np.number) else np.nan,
                "p10": float(s.quantile(0.10))
                if np.issubdtype(s.dtype, np.number)
                else np.nan,
                "p50": float(s.quantile(0.50))
                if np.issubdtype(s.dtype, np.number)
                else np.nan,
                "p90": float(s.quantile(0.90))
                if np.issubdtype(s.dtype, np.number)
                else np.nan,
            }
        )
    return pd.DataFrame(out)


def roll_rate_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute roll rate matrix across buckets C,30,60,90+,CO from t to t+1."""
    req = {"loan_id", "asof_month", "roll_rate_bucket"}
    if not req.issubset(panel.columns):
        raise ValueError(f"panel missing required columns: {req - set(panel.columns)}")
    panel = panel.sort_values(["loan_id", "asof_month"])  # ensure order
    panel["next_bucket"] = panel.groupby("loan_id")["roll_rate_bucket"].shift(-1)
    valid = panel.dropna(subset=["next_bucket"])  # last month has no next

    mat = (
        valid.groupby(["roll_rate_bucket", "next_bucket"]).size().unstack(fill_value=0)
    )
    # Normalize rows to probabilities
    mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return mat.reset_index().rename(columns={"roll_rate_bucket": "from"})
