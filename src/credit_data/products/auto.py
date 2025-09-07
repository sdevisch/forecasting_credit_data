from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def generate_auto_loans(borrowers: pd.DataFrame, seed: int | None = 12345) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 101)
    n = len(borrowers)
    loan_id = np.arange(10_000_000, 10_000_000 + n, dtype=np.int64)
    term = rs.integers(36, 85, size=n)  # 3-7 years
    rate = np.clip(0.05 + (700 - borrowers["fico_baseline"]).clip(0) * 0.0002 + rs.normal(0, 0.005, n), 0.03, 0.18)
    amount = np.clip(rs.normal(25000, 8000, n), 5000, 80000)
    ltv = np.clip(rs.normal(0.95, 0.1, n), 0.4, 1.2)

    df = pd.DataFrame(
        {
            "loan_id": loan_id,
            "borrower_id": borrowers["borrower_id"].values,
            "product": "auto",
            "origination_dt": pd.to_datetime("2019-01-01") + pd.to_timedelta(rs.integers(0, 48, n), unit="D"),
            "maturity_months": term,
            "interest_rate": rate,
            "orig_balance": amount,
            "secured_flag": True,
            "ltv_at_orig": ltv,
            "risk_grade": pd.cut(borrowers["fico_baseline"], bins=[0, 629, 669, 709, 749, 850], labels=["E","D","C","B","A"]).astype(str),
            "underwriting_dti": np.clip(rs.normal(0.35, 0.1, n), 0.05, 0.85),
            "underwriting_fico": borrowers["fico_baseline"].values,
            "channel": "dealer",
            "state": borrowers["state"].values,
            "vintage": pd.to_datetime("2019-01-01").strftime("%Y-%m"),
            "credit_limit": np.nan,
        }
    )
    return df


def simulate_auto_panel(loans: pd.DataFrame, macro: pd.DataFrame, months: int, seed: int | None = 12345) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 102)
    start_month = pd.to_datetime("2020-01-01")
    panel_months = pd.date_range(start=start_month, periods=months, freq="MS")
    m = len(panel_months)
    n = len(loans)

    macro_aligned = macro.set_index("asof_month").reindex(panel_months).ffill().bfill()
    unemp = macro_aligned["unemployment"].to_numpy()
    unemp_dev = unemp - unemp.mean()
    hpi_yoy = macro_aligned.get("hpi_yoy", pd.Series(0, index=panel_months)).to_numpy()

    balance = loans["orig_balance"].to_numpy().astype(float)
    rate = loans["interest_rate"].to_numpy()
    term = loans["maturity_months"].to_numpy()
    fico = loans["underwriting_fico"].to_numpy()

    # Fixed payment amortization
    r_m = rate / 12.0
    payment = np.where(r_m > 0, balance * (r_m / (1 - (1 + r_m) ** (-term))), balance / term)

    # State machine: 0=C,1=30,2=60,3=90+,4=CO
    state = np.zeros(n, dtype=np.int8)
    recovery_scheduled = np.zeros((n, m), dtype=float)
    recovery_lag_sched = np.zeros((n, m), dtype=int)
    lgd_at_co = np.full(n, np.nan, dtype=float)

    def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(x, lo, hi)

    records: list[pd.DataFrame] = []
    for t_idx, asof in enumerate(panel_months):
        risk = (720 - fico).clip(0)
        udev = unemp_dev[t_idx]
        hpi_down = np.clip(-hpi_yoy[t_idx] / 10.0, 0.0, 0.5)  # higher when HPI falling

        # Roll and cure probabilities
        p_c_to_30 = clamp(0.006 + 0.00005 * risk + 0.008 * max(0.0, udev), 0.001, 0.2)
        p_30_to_60 = clamp(0.07 + 0.00007 * risk + 0.009 * max(0.0, udev), 0.01, 0.3)
        p_60_to_90 = clamp(0.10 + 0.00009 * risk + 0.012 * max(0.0, udev), 0.02, 0.4)
        p_90_to_co = clamp(0.14 + 0.00011 * risk + 0.015 * max(0.0, udev), 0.03, 0.55)

        p_30_cure = clamp(0.32 - 0.00018 * risk - 0.008 * max(0.0, udev), 0.03, 0.65)
        p_60_cure = clamp(0.20 - 0.00014 * risk - 0.006 * max(0.0, udev), 0.02, 0.45)
        p_90_cure = clamp(0.08 - 0.0001 * risk - 0.004 * max(0.0, udev), 0.0, 0.25)

        # Random uniforms
        u = rs.random(n)
        mask_c = state == 0
        trans_c_to_30 = mask_c & (u < p_c_to_30)
        state[trans_c_to_30] = 1

        mask_30 = state == 1
        u = rs.random(n)
        trans_30_to_60 = mask_30 & (u < p_30_to_60)
        trans_30_cure = mask_30 & (u >= p_30_to_60) & (u < p_30_to_60 + p_30_cure)
        state[trans_30_to_60] = 2
        state[trans_30_cure] = 0

        mask_60 = state == 2
        u = rs.random(n)
        trans_60_to_90 = mask_60 & (u < p_60_to_90)
        trans_60_cure = mask_60 & (u >= p_60_to_90) & (u < p_60_to_90 + p_60_cure)
        state[trans_60_to_90] = 3
        state[trans_60_cure] = 0

        mask_90 = state == 3
        u = rs.random(n)
        trans_90_to_co = mask_90 & (u < p_90_to_co)
        trans_90_cure = mask_90 & (u >= p_90_to_co) & (u < p_90_to_co + p_90_cure)
        became_co = np.where(trans_90_to_co)[0]
        state[trans_90_to_co] = 4
        state[trans_90_cure] = 0

        # Amortization (skip on CO)
        interest = r_m * balance
        principal = np.minimum(payment - interest, balance)
        balance = np.where(state == 4, 0.0, balance - principal)

        # LGD at CO depends on HPI and udev (secured)
        if became_co.size > 0:
            lgd_vals = clamp(0.45 + 0.12 * hpi_down + 0.03 * max(0.0, udev) + rs.normal(0.0, 0.03, became_co.size), 0.2, 0.9)
            lgd_at_co[became_co] = lgd_vals
            lag = rs.integers(2, 8, size=became_co.size)  # faster recoveries than unsecured
            rec_month = t_idx + lag
            valid = rec_month < m
            idx_valid = became_co[valid]
            rec_month_valid = rec_month[valid]
            if idx_valid.size > 0:
                rec_amt = (1.0 - lgd_vals[valid]) * (balance[idx_valid] + principal[idx_valid] + interest[idx_valid])
                recovery_scheduled[idx_valid, rec_month_valid] += rec_amt
                recovery_lag_sched[idx_valid, rec_month_valid] = lag[valid]

        roll_bucket = np.where(state == 0, "C", np.where(state == 1, "30", np.where(state == 2, "60", np.where(state == 3, "90+", "CO"))))
        dpp = np.where(state == 0, 0, np.where(state == 1, 30, np.where(state == 2, 60, np.where(state == 3, 90, 120))))
        default_flag = (trans_90_to_co).astype(bool)
        chargeoff_flag = (state == 4)
        cure_flag = (trans_30_cure | trans_60_cure | trans_90_cure)

        lgd_t = np.where(chargeoff_flag, np.nan_to_num(lgd_at_co, nan=0.6), np.nan)  # only meaningful at CO
        rec_amt_t = recovery_scheduled[:, t_idx]
        rec_lag_t = recovery_lag_sched[:, t_idx]

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
                "days_past_due": dpp.astype(int),
                "roll_rate_bucket": roll_bucket,
                "default_flag": default_flag,
                "chargeoff_flag": chargeoff_flag,
                "recovery_amt": rec_amt_t,
                "recovery_lag_m": rec_lag_t,
                "cure_flag": cure_flag,
                "loss_given_default": lgd_t,
                "effective_rate": rate,
                "forbearance_flag": np.zeros(n, dtype=bool),
            }
        )
        records.append(rec)

    return pd.concat(records, ignore_index=True)
