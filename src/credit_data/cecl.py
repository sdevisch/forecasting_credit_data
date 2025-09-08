from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def compute_monthly_ecl(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute per-loan, per-month ECL.

    ECL_t = PD_t × LGD_t × EAD_t.

    Parameters:
    - panel: DataFrame with columns: asof_month, loan_id, balance_ead, default_flag, loss_given_default.

    Returns:
    - DataFrame with added columns: pd_t, ead_t, lgd_t, monthly_ecl
    """
    required = {"asof_month", "loan_id", "balance_ead", "default_flag", "loss_given_default"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")

    df = panel.copy()
    df["pd_t"] = df["default_flag"].astype(float)
    df["ead_t"] = df["balance_ead"].astype(float)
    df["lgd_t"] = df["loss_given_default"].astype(float).fillna(0.85)
    df["monthly_ecl"] = df["pd_t"] * df["lgd_t"] * df["ead_t"]
    return df


def compute_lifetime_ecl(monthly_ecl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly ECL to lifetime per loan.

    Parameters:
    - monthly_ecl_df: DataFrame with at least: loan_id, monthly_ecl

    Returns:
    - DataFrame with columns: loan_id, lifetime_ecl
    """
    required = {"asof_month", "loan_id", "monthly_ecl"}
    missing = required - set(monthly_ecl_df.columns)
    if missing:
        raise ValueError(f"monthly_ecl_df missing required columns: {missing}")

    lifetime = (
        monthly_ecl_df.groupby(["loan_id"], as_index=False, observed=False)["monthly_ecl"].sum()
        .rename(columns={"monthly_ecl": "lifetime_ecl"})
    )
    return lifetime


def compute_portfolio_aggregates(monthly_ecl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute portfolio-level monthly aggregates.

    Parameters:
    - monthly_ecl_df: DataFrame with at least: asof_month, monthly_ecl, ead_t

    Returns:
    - DataFrame with columns: asof_month, portfolio_monthly_ecl, portfolio_ead
    """
    agg = (
        monthly_ecl_df.groupby(["asof_month"], as_index=False, observed=False)[["monthly_ecl", "ead_t"]].sum()
        .rename(columns={"monthly_ecl": "portfolio_monthly_ecl", "ead_t": "portfolio_ead"})
    )
    return agg
