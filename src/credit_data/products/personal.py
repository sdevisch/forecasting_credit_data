from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def generate_personal_loans(borrowers: pd.DataFrame, seed: int | None = 12345) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 201)
    n = len(borrowers)
    loan_id = np.arange(20_000_000, 20_000_000 + n, dtype=np.int64)
    term = rs.integers(24, 61, size=n)  # 2-5 years
    rate = np.clip(0.12 + (700 - borrowers["fico_baseline"]).clip(0) * 0.0005 + rs.normal(0, 0.01, n), 0.08, 0.36)
    amount = np.clip(rs.normal(12000, 6000, n), 1000, 60000)

    df = pd.DataFrame(
        {
            "loan_id": loan_id,
            "borrower_id": borrowers["borrower_id"].values,
            "product": "personal",
            "origination_dt": pd.to_datetime("2020-01-01") + pd.to_timedelta(rs.integers(0, 24, n), unit="D"),
            "maturity_months": term,
            "interest_rate": rate,
            "orig_balance": amount,
            "secured_flag": False,
            "ltv_at_orig": np.nan,
            "risk_grade": pd.cut(borrowers["fico_baseline"], bins=[0, 629, 669, 709, 749, 850], labels=["E","D","C","B","A"]).astype(str),
            "underwriting_dti": np.clip(rs.normal(0.4, 0.12, n), 0.05, 0.95),
            "underwriting_fico": borrowers["fico_baseline"].values,
            "channel": rs.choice(["branch","online"], size=n, p=[0.4, 0.6]),
            "state": borrowers["state"].values,
            "vintage": pd.to_datetime("2020-01-01").strftime("%Y-%m"),
            "credit_limit": np.nan,
        }
    )
    return df


def simulate_personal_panel(loans: pd.DataFrame, macro: pd.DataFrame, months: int, seed: int | None = 12345) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 202)
    start_month = pd.to_datetime("2020-01-01")
    panel_months = pd.date_range(start=start_month, periods=months, freq="MS")
    m = len(panel_months)
    n = len(loans)

    macro_aligned = macro.set_index("asof_month").reindex(panel_months).ffill().bfill()
    unemp = macro_aligned["unemployment"].to_numpy()
    unemp_dev = unemp - unemp.mean()

    balance = loans["orig_balance"].to_numpy().astype(float)
    rate = loans["interest_rate"].to_numpy()
    term = loans["maturity_months"].to_numpy()

    r_m = rate / 12.0
    payment = np.where(r_m > 0, balance * (r_m / (1 - (1 + r_m) ** (-term))), balance / term)

    fico = loans["underwriting_fico"].to_numpy()
    base_pd = (0.002 + 0.00004 * (720 - fico).clip(0))

    records = []
    charged_off = np.zeros(n, dtype=bool)
    for t_idx, asof in enumerate(panel_months):
        pd_t = np.clip(base_pd * (1.0 + 0.12 * max(0.0, unemp_dev[t_idx])), 0.0004, 0.15)
        u = rs.random(n)
        default_flag = (u < pd_t) & (~charged_off) & (balance > 0)
        charged_off |= default_flag

        interest = r_m * balance
        principal = np.minimum(payment - interest, balance)
        balance = np.where(charged_off, 0.0, balance - principal)

        lgd = np.clip(0.85 + 0.03 * unemp_dev[t_idx], 0.6, 0.98)
        recovery = np.where(default_flag, (1 - lgd) * (balance + principal + interest), 0.0)

        rec = pd.DataFrame(
            {
                "asof_month": asof,
                "loan_id": loans["loan_id"].values,
                "borrower_id": loans["borrower_id"].values,
                "product": loans["product"].values,
                "balance_ead": balance,
                "scheduled_principal": principal,
                "current_principal": balance,
                "current_interest": interest,
                "utilization": np.nan,
                "prepay_flag": np.zeros(n, dtype=bool),
                "days_past_due": (charged_off * 120).astype(int),
                "roll_rate_bucket": np.where(charged_off, "CO", "C"),
                "default_flag": default_flag,
                "chargeoff_flag": charged_off,
                "recovery_amt": recovery,
                "recovery_lag_m": np.zeros(n, dtype=int),
                "cure_flag": np.zeros(n, dtype=bool),
                "loss_given_default": np.where(default_flag | charged_off, lgd, lgd),
                "effective_rate": rate,
                "forbearance_flag": np.zeros(n, dtype=bool),
            }
        )
        records.append(rec)

    return pd.concat(records, ignore_index=True)
