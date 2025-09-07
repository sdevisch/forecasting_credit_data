from __future__ import annotations

import numpy as np
import pandas as pd


def generate_mortgage_loans(borrowers: pd.DataFrame, seed: int | None = 12345) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 301)
    n = len(borrowers)
    loan_id = np.arange(30_000_000, 30_000_000 + n, dtype=np.int64)
    term = rs.integers(180, 361, size=n)  # 15-30 years
    rate = np.clip(0.035 + (700 - borrowers["fico_baseline"]).clip(0) * 0.00005 + rs.normal(0, 0.003, n), 0.02, 0.08)
    amount = np.clip(rs.normal(350000, 120000, n), 50000, 1500000)
    ltv = np.clip(rs.normal(0.8, 0.1, n), 0.3, 1.2)

    df = pd.DataFrame(
        {
            "loan_id": loan_id,
            "borrower_id": borrowers["borrower_id"].values,
            "product": "mortgage",
            "origination_dt": pd.to_datetime("2017-01-01") + pd.to_timedelta(rs.integers(0, 365 * 3, n), unit="D"),
            "maturity_months": term,
            "interest_rate": rate,
            "orig_balance": amount,
            "secured_flag": True,
            "ltv_at_orig": ltv,
            "risk_grade": pd.cut(borrowers["fico_baseline"], bins=[0, 639, 679, 719, 759, 850], labels=["D","C","B","A","AA"]).astype(str),
            "underwriting_dti": np.clip(rs.normal(0.32, 0.08, n), 0.05, 0.7),
            "underwriting_fico": borrowers["fico_baseline"].values,
            "channel": rs.choice(["retail","broker"], size=n, p=[0.7, 0.3]),
            "state": borrowers["state"].values,
            "vintage": pd.to_datetime("2017-01-01").strftime("%Y-%m"),
            "credit_limit": np.nan,
        }
    )
    return df


def simulate_mortgage_panel(loans: pd.DataFrame, macro: pd.DataFrame, months: int, seed: int | None = 12345) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 302)
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

    balance = loans["orig_balance"].to_numpy().astype(float)
    rate = loans["interest_rate"].to_numpy()
    term = loans["maturity_months"].to_numpy()
    fico = loans["underwriting_fico"].to_numpy()

    r_m = rate / 12.0
    payment = np.where(r_m > 0, balance * (r_m / (1 - (1 + r_m) ** (-term))), balance / term)

    state = np.zeros(n, dtype=np.int8)

    records: list[pd.DataFrame] = []
    for t_idx, asof in enumerate(panel_months):
        risk = (720 - fico).clip(0)
        udev = unemp_dev[t_idx]
        hpi_down = np.clip(-hpi_yoy[t_idx] / 10.0, 0.0, 0.6)
        seasoning = min((t_idx + 1) / 24.0, 1.0)

        # Simple delinquency progression probabilities
        p_c_to_30 = np.clip(0.003 + 0.00003 * risk + 0.006 * max(0.0, udev), 0.0005, 0.12) * seasoning
        # Aggregate defaults via a hazard on 90+ bucket not explicitly modeled here
        default_flag = (rs.random(n) < p_c_to_30 * 0.2)
        state = np.where(default_flag, 4, state)

        interest = r_m * balance
        principal = np.minimum(payment - interest, balance)

        # Prepayment via SMM: higher when rates drop
        smm = np.clip(0.004 + 0.02 * np.clip(-fed_dev[t_idx], 0.0, None), 0.0, 0.10)
        prepay_amt = smm * balance

        balance = np.where(state == 4, 0.0, balance - principal - prepay_amt)

        lgd_t = np.where(state == 4, np.clip(0.35 + 0.15 * hpi_down + 0.03 * max(0.0, udev), 0.15, 0.8), np.nan)

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
                "prepay_flag": (prepay_amt > 0),
                "days_past_due": (state * 30).clip(0, 120).astype(int),
                "roll_rate_bucket": np.where(state == 0, "C", np.where(state == 1, "30", np.where(state == 2, "60", np.where(state == 3, "90+", "CO")))),
                "default_flag": default_flag,
                "chargeoff_flag": (state == 4),
                "recovery_amt": np.zeros(n),
                "recovery_lag_m": np.zeros(n, dtype=int),
                "cure_flag": np.zeros(n, dtype=bool),
                "loss_given_default": lgd_t,
                "effective_rate": rate,
                "forbearance_flag": np.zeros(n, dtype=bool),
            }
        )
        records.append(rec)

    return pd.concat(records, ignore_index=True)
