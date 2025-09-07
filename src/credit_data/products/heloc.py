from __future__ import annotations

import numpy as np
import pandas as pd


def generate_heloc_loans(borrowers: pd.DataFrame, seed: int | None = 12345) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 351)
    n = len(borrowers)
    loan_id = np.arange(40_000_000, 40_000_000 + n, dtype=np.int64)
    rate = np.clip(0.055 + (700 - borrowers["fico_baseline"]).clip(0) * 0.00015 + rs.normal(0, 0.005, n), 0.03, 0.18)
    limit = np.clip(rs.normal(120000, 60000, n), 10000, 500000)
    ltv = np.clip(rs.normal(0.7, 0.1, n), 0.2, 1.1)

    df = pd.DataFrame(
        {
            "loan_id": loan_id,
            "borrower_id": borrowers["borrower_id"].values,
            "product": "heloc",
            "origination_dt": pd.to_datetime("2018-01-01") + pd.to_timedelta(rs.integers(0, 365 * 4, n), unit="D"),
            "maturity_months": rs.integers(120, 241, size=n),
            "interest_rate": rate,
            "orig_balance": (limit * rs.uniform(0.2, 0.8, n)),
            "secured_flag": True,
            "ltv_at_orig": ltv,
            "risk_grade": pd.cut(borrowers["fico_baseline"], bins=[0, 639, 679, 719, 759, 850], labels=["D","C","B","A","AA"]).astype(str),
            "underwriting_dti": np.clip(rs.normal(0.34, 0.1, n), 0.05, 0.85),
            "underwriting_fico": borrowers["fico_baseline"].values,
            "channel": rs.choice(["branch","online"], size=n, p=[0.6, 0.4]),
            "state": borrowers["state"].values,
            "vintage": pd.to_datetime("2018-01-01").strftime("%Y-%m"),
            "credit_limit": limit,
        }
    )
    return df


def simulate_heloc_panel(loans: pd.DataFrame, macro: pd.DataFrame, months: int, seed: int | None = 12345) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 352)
    start_month = pd.to_datetime("2020-01-01")
    panel_months = pd.date_range(start=start_month, periods=months, freq="MS")
    m = len(panel_months)
    n = len(loans)

    macro_aligned = macro.set_index("asof_month").reindex(panel_months).ffill().bfill()
    unemp = macro_aligned["unemployment"].to_numpy()
    unemp_dev = unemp - unemp.mean()
    hpi_yoy = macro_aligned.get("hpi_yoy", pd.Series(0, index=panel_months)).to_numpy()
    fed = macro_aligned.get("fed_funds")
    fed_dev = (fed - fed.mean()).to_numpy() if fed is not None else np.zeros(m)

    credit_limit = loans["credit_limit"].to_numpy()[:, None]
    util0 = np.clip(loans["orig_balance"].to_numpy() / loans["credit_limit"].to_numpy(), 0.05, 0.95)
    util_shocks = rs.normal(0.0, 0.025, size=(n, m)) + 0.015 * (unemp_dev.reshape(1, m)) - 0.01 * (fed_dev.reshape(1, m))
    util_path = np.clip(util0[:, None] + util_shocks.cumsum(axis=1), 0.01, 0.99)

    fico = loans["underwriting_fico"].to_numpy()
    risk = (720 - fico).clip(0)

    state = np.zeros(n, dtype=np.int8)

    records = []
    for t_idx, asof in enumerate(panel_months):
        udev = unemp_dev[t_idx]
        hpi_down = np.clip(-hpi_yoy[t_idx] / 10.0, 0.0, 0.6)
        seasoning = min((t_idx + 1) / 18.0, 1.0)

        p_c_to_30 = np.clip((0.008 + 0.00005 * risk + 0.008 * max(0.0, udev)) * seasoning, 0.001, 0.22)
        # Aggregate default proxy
        default_flag = rs.random(n) < p_c_to_30 * 0.2
        state = np.where(default_flag, 4, state)

        bal_t = util_path[:, t_idx] * credit_limit[:, 0]
        interest_t = loans["interest_rate"].to_numpy() * bal_t / 12.0
        payment = 0.02 * credit_limit[:, 0]

        # Prepayment boosted by higher rates (closing balances)
        smm = np.clip(0.002 + 0.01 * np.clip(-fed_dev[t_idx], 0.0, None), 0.0, 0.06)
        prepay_amt = smm * bal_t

        bal_t = np.where(state == 4, 0.0, np.clip(bal_t - payment - prepay_amt, 0.0, None))

        lgd_t = np.where(state == 4, np.clip(0.4 + 0.2 * hpi_down + 0.03 * max(0.0, udev), 0.2, 0.9), 0.5)

        rec = pd.DataFrame(
            {
                "asof_month": asof,
                "loan_id": loans["loan_id"].values,
                "borrower_id": loans["borrower_id"].values,
                "product": loans["product"].values,
                "balance_ead": bal_t,
                "scheduled_principal": np.zeros(n),
                "current_principal": bal_t,
                "current_interest": interest_t,
                "utilization": np.clip(bal_t / credit_limit[:, 0], 0.0, 1.0),
                "prepay_flag": (prepay_amt > 0),
                "days_past_due": (state * 30).clip(0, 120).astype(int),
                "roll_rate_bucket": np.where(state == 0, "C", np.where(state == 1, "30", np.where(state == 2, "60", np.where(state == 3, "90+", "CO")))),
                "default_flag": (state == 4),
                "chargeoff_flag": (state == 4),
                "recovery_amt": np.zeros(n),
                "recovery_lag_m": np.zeros(n, dtype=int),
                "cure_flag": (state == 0),
                "loss_given_default": lgd_t,
                "effective_rate": loans["interest_rate"].values,
                "forbearance_flag": np.zeros(n, dtype=bool),
            }
        )
        records.append(rec)

    return pd.concat(records, ignore_index=True)
