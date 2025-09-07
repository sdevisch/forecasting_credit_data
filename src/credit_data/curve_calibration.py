from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def hazards_from_cumulative(cum: Sequence[float]) -> np.ndarray:
    """Convert a cumulative default curve to per-period hazards.

    cum: cumulative defaults (monotone non-decreasing) of length T in [0,1].
    Returns: numpy array of hazards length T in [0,1].
    """
    cum_arr = np.asarray(cum, dtype=float)
    t = len(cum_arr)
    hazards = np.zeros(t, dtype=float)
    prev_cum = 0.0
    survival = 1.0
    for i in range(t):
        inc = max(cum_arr[i] - prev_cum, 0.0)
        hazards[i] = inc / max(survival, 1e-9)
        prev_cum = cum_arr[i]
        survival *= (1.0 - hazards[i])
    clipped = np.asarray(np.clip(hazards, 0.0, 1.0), dtype=float)
    return clipped


def compute_scalers(model_hazards: Sequence[float], target_cum: Sequence[float]) -> np.ndarray:
    """Compute multiplicative scalers so model hazards align to target cumulative curve."""
    target_haz = hazards_from_cumulative(target_cum)
    model = np.asarray(model_hazards, dtype=float)
    eps = 1e-9
    scalers = target_haz / np.clip(model, eps, None)
    clipped = np.asarray(np.clip(scalers, 0.1, 10.0), dtype=float)
    return clipped


def apply_monthly_ecl_scaling(monthly_df: pd.DataFrame, scalers: Sequence[float], months: int) -> pd.DataFrame:
    """Apply per-month scaling factors to monthly_ecl column.

    Expects columns: loan_id, asof_month, monthly_ecl
    """
    df = monthly_df.copy()
    df = df.sort_values(["loan_id", "asof_month"])  # ensure order
    df["month_idx"] = df.groupby("loan_id").cumcount()
    df = df[df["month_idx"] < months]
    scale_map = dict(enumerate(scalers))
    df["scale_factor"] = df["month_idx"].map(scale_map).fillna(1.0)
    df["monthly_ecl_scaled"] = df["monthly_ecl"] * df["scale_factor"]
    return df
