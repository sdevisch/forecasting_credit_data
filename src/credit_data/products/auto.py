from __future__ import annotations


import numpy as np
import pandas as pd


def generate_auto_loans(
    borrowers: pd.DataFrame, seed: int | None = 12345
) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 101)
    n = len(borrowers)
    loan_id = np.arange(10_000_000, 10_000_000 + n, dtype=np.int64)
    term = rs.integers(36, 85, size=n)  # 3-7 years
    rate = np.clip(
        0.05
        + (700 - borrowers["fico_baseline"]).clip(0) * 0.0002
        + rs.normal(0, 0.005, n),
        0.03,
        0.18,
    )
    amount = np.clip(rs.normal(25000, 8000, n), 5000, 80000)
    ltv = np.clip(rs.normal(0.95, 0.1, n), 0.4, 1.2)

    df = pd.DataFrame(
        {
            "loan_id": loan_id,
            "borrower_id": borrowers["borrower_id"].values,
            "product": "auto",
            "origination_dt": pd.to_datetime("2019-01-01")
            + pd.to_timedelta(rs.integers(0, 48, n), unit="D"),
            "maturity_months": term,
            "interest_rate": rate,
            "orig_balance": amount,
            "secured_flag": True,
            "ltv_at_orig": ltv,
            "risk_grade": pd.cut(
                borrowers["fico_baseline"],
                bins=[0, 629, 669, 709, 749, 850],
                labels=["E", "D", "C", "B", "A"],
            ).astype(str),
            "underwriting_dti": np.clip(rs.normal(0.35, 0.1, n), 0.05, 0.85),
            "underwriting_fico": borrowers["fico_baseline"].values,
            "channel": "dealer",
            "state": borrowers["state"].values,
            "vintage": pd.to_datetime("2019-01-01").strftime("%Y-%m"),
            "credit_limit": np.nan,
        }
    )
    return df


def simulate_auto_panel(
    loans: pd.DataFrame, macro: pd.DataFrame, months: int, seed: int | None = 12345
) -> pd.DataFrame:
    rs = np.random.default_rng(None if seed is None else seed + 102)
    start_month = pd.to_datetime("2020-01-01")
    panel_months = pd.date_range(start=start_month, periods=months, freq="MS")
    m = len(panel_months)
    n = len(loans)

    macro_aligned = macro.set_index("asof_month").reindex(panel_months).ffill().bfill()
    unemp = macro_aligned["unemployment"].to_numpy()
    unemp_dev = unemp - unemp.mean()
    fed = macro_aligned.get("fed_funds")
    fed_dev = (fed - fed.mean()).to_numpy() if fed is not None else np.zeros(m)
    hpi_yoy = macro_aligned.get("hpi_yoy", pd.Series(0, index=panel_months)).to_numpy()

    balance = loans["orig_balance"].to_numpy().astype(float)
    rate = loans["interest_rate"].to_numpy()
    term = loans["maturity_months"].to_numpy()

    # Fixed payment amortization approximation
    r_m = rate / 12.0
    payment = np.where(
        r_m > 0, balance * (r_m / (1 - (1 + r_m) ** (-term))), balance / term
    )

    fico = loans["underwriting_fico"].to_numpy()
    base_pd = 0.001 + 0.00002 * (720 - fico).clip(0)

    records = []
    charged_off = np.zeros(n, dtype=bool)
    for t_idx, asof in enumerate(panel_months):
        seasoning = min((t_idx + 1) / 18.0, 1.0)
        # PD increases in early seasoning, macro stress
        pd_t = np.clip(
            base_pd
            * (0.8 + 0.4 * seasoning)
            * (1.0 + 0.1 * max(0.0, unemp_dev[t_idx])),
            0.0002,
            0.08,
        )
        u = rs.random(n)
        default_flag = (u < pd_t) & (~charged_off) & (balance > 0)
        charged_off |= default_flag

        # Interest and amortization
        interest = r_m * balance
        principal = np.minimum(payment - interest, balance)

        # Simple prepayment: higher when rates drop
        smm = np.clip(0.002 + 0.01 * np.clip(-fed_dev[t_idx], 0.0, None), 0.0, 0.05)
        prepay_amt = smm * balance

        balance = np.where(charged_off, 0.0, balance - principal - prepay_amt)

        # Recoveries: function of unemployment and HPI
        lgd = np.clip(
            0.5
            + 0.02 * unemp_dev[t_idx]
            + 0.1 * np.clip(-hpi_yoy[t_idx] / 10.0, 0.0, 0.5),
            0.3,
            0.9,
        )
        recovery = np.where(
            default_flag, (1 - lgd) * (balance + principal + interest), 0.0
        )

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
