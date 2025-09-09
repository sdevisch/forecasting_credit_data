from __future__ import annotations

import os
import pandas as pd


def coverage_by_product(cecl_dir: str) -> pd.DataFrame:
    by_product = pd.read_parquet(
        os.path.join(cecl_dir, "portfolio_aggregates_by_product.parquet")
    )
    latest = (
        by_product.sort_values("asof_month").groupby("product", observed=False).tail(1)
    )
    latest = latest.rename(
        columns={"portfolio_monthly_ecl": "monthly_ecl", "portfolio_ead": "ead"}
    )
    latest["coverage_pct"] = (latest["monthly_ecl"] / latest["ead"]).fillna(0.0) * 100.0
    return latest[["product", "asof_month", "monthly_ecl", "ead", "coverage_pct"]]


def monthly_summary_overall(cecl_dir: str) -> pd.DataFrame:
    overall = pd.read_parquet(
        os.path.join(cecl_dir, "portfolio_aggregates_overall.parquet")
    )
    overall = overall.sort_values("asof_month")
    overall["coverage_pct"] = (
        overall["portfolio_monthly_ecl"] / overall["portfolio_ead"]
    ).fillna(0.0) * 100.0
    return overall


def vintage_summary(loans_dir: str) -> pd.DataFrame:
    loan_files = [
        p
        for p in os.listdir(loans_dir)
        if p.startswith("loans_") and p.endswith(".parquet")
    ]
    dfs = [pd.read_parquet(os.path.join(loans_dir, p)) for p in loan_files]
    loans = pd.concat(dfs, ignore_index=True)
    vint = (
        loans.groupby(["product", "vintage"], as_index=False, observed=False)[
            "orig_balance"
        ]
        .sum()
        .rename(columns={"orig_balance": "orig_balance_total"})
    )
    return vint.sort_values(["product", "vintage"])
