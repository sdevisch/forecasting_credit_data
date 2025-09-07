from __future__ import annotations

import numpy as np
import pandas as pd


def hazards_from_cumulative(cum: np.ndarray) -> np.ndarray:
    cum = np.asarray(cum, dtype=float)
    t = len(cum)
    hazards = np.zeros(t, dtype=float)
    prev_cum = 0.0
    survival = 1.0
    for i in range(t):
        inc = max(cum[i] - prev_cum, 0.0)
        hazards[i] = inc / max(survival, 1e-9)
        prev_cum = cum[i]
        survival *= (1.0 - hazards[i])
    return np.clip(hazards, 0.0, 1.0)


def compute_scalers(model_hazards: np.ndarray, target_cum: np.ndarray) -> np.ndarray:
    target_haz = hazards_from_cumulative(np.asarray(target_cum, dtype=float))
    eps = 1e-9
    scalers = target_haz / np.clip(model_hazards, eps, None)
    return np.clip(scalers, 0.1, 10.0)


def apply_monthly_ecl_scaling(monthly_df: pd.DataFrame, scalers: np.ndarray, months: int) -> pd.DataFrame:
    df = monthly_df.copy()
    # Expect consecutive months; group by loan to map month index 0..months-1
    df = df.sort_values(["loan_id", "asof_month"])  # ensure order
    df["month_idx"] = df.groupby("loan_id").cumcount()
    df = df[df["month_idx"] < months]
    df["scale_factor"] = df["month_idx"].map(dict(enumerate(scalers)))
    df["monthly_ecl_scaled"] = df["monthly_ecl"] * df["scale_factor"]
    return df
