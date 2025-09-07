from __future__ import annotations

import numpy as np
import pandas as pd


def compute_monthly_ecl(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute per-loan per-month ECL = PD * LGD * EAD.

    Expects panel columns: asof_month, loan_id, balance_ead, default_flag, loss_given_default.
    Approximates monthly PD as the observed default_flag for simplicity in prototype.
    """
    required = {"asof_month", "loan_id", "balance_ead", "default_flag", "loss_given_default"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")

    df = panel.copy()
    # Prototype monthly PD proxy: default occurrence in month
    df["pd_t"] = df["default_flag"].astype(float)
    df["ead_t"] = df["balance_ead"].astype(float)
    df["lgd_t"] = df["loss_given_default"].astype(float).fillna(0.85)
    df["monthly_ecl"] = df["pd_t"] * df["lgd_t"] * df["ead_t"]
    return df


def compute_lifetime_ecl(monthly_ecl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly ECL to lifetime per as-of and loan."""
    required = {"asof_month", "loan_id", "monthly_ecl"}
    missing = required - set(monthly_ecl_df.columns)
    if missing:
        raise ValueError(f"monthly_ecl_df missing required columns: {missing}")

    lifetime = (
        monthly_ecl_df.groupby(["loan_id"], as_index=False)["monthly_ecl"].sum()
        .rename(columns={"monthly_ecl": "lifetime_ecl"})
    )
    return lifetime


def compute_portfolio_aggregates(monthly_ecl_df: pd.DataFrame) -> pd.DataFrame:
    """Portfolio-level sums by month."""
    agg = (
        monthly_ecl_df.groupby(["asof_month"], as_index=False)[["monthly_ecl", "ead_t"]].sum()
        .rename(columns={"monthly_ecl": "portfolio_monthly_ecl", "ead_t": "portfolio_ead"})
    )
    return agg
