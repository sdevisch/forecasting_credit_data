from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Tuple

import numpy as np
import pandas as pd


US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]
SEGMENTS = ["mass", "affluent", "small_business", "private", "student"]
PRODUCTS = ["card"]  # initial prototype focuses on card; extendable
INDUSTRIES = [
    "tech","finance","healthcare","education","manufacturing","retail","hospitality","transport","construction","other",
]


def _rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = 12345
    return np.random.default_rng(seed)


def generate_borrowers(num_borrowers: int, seed: int | None = 12345) -> pd.DataFrame:
    rs = _rng(seed)
    borrower_id = np.arange(1, num_borrowers + 1, dtype=np.int64)
    state = rs.choice(US_STATES, size=num_borrowers, replace=True)
    income_annual = rs.lognormal(mean=10.5, sigma=0.5, size=num_borrowers)  # ~ $36k-$200k
    employment_tenure_months = rs.integers(0, 360, size=num_borrowers)
    industry = rs.choice(INDUSTRIES, size=num_borrowers, replace=True)
    education = rs.choice(["hs","college","grad"], size=num_borrowers, p=[0.35, 0.45, 0.2])
    household_size = rs.integers(1, 6, size=num_borrowers)

    # Correlated fico/utilization using a simple factor model
    systemic = rs.standard_normal(num_borrowers)
    fico_raw = 690 + 60 * systemic + rs.normal(0, 40, num_borrowers)
    fico_baseline = np.clip(fico_raw, 500, 850).astype(int)
    util_base = np.clip(0.35 + 0.1 * (-systemic) + rs.normal(0, 0.15, num_borrowers), 0.0, 1.0)

    prior_delinquencies = rs.poisson(lam=np.clip((720 - fico_baseline) / 200.0, 0.05, 2.0))
    bank_tenure_months = rs.integers(1, 360, size=num_borrowers)
    segment = rs.choice(SEGMENTS, size=num_borrowers, p=[0.6, 0.2, 0.08, 0.06, 0.06])

    df = pd.DataFrame(
        {
            "borrower_id": borrower_id,
            "state": state,
            "zip3": rs.integers(100, 999, size=num_borrowers).astype(str),
            "income_annual": income_annual,
            "employment_tenure_months": employment_tenure_months,
            "industry": industry,
            "education": education,
            "household_size": household_size,
            "fico_baseline": fico_baseline,
            "credit_utilization_baseline": util_base,
            "prior_delinquencies": prior_delinquencies,
            "bank_tenure_months": bank_tenure_months,
            "segment": segment,
        }
    )
    return df


def generate_loans(borrowers: pd.DataFrame, seed: int | None = 12345) -> pd.DataFrame:
    rs = _rng(None if seed is None else seed + 7)
    n = len(borrowers)
    # One revolving card account per borrower for prototype
    loan_id = np.arange(1, n + 1, dtype=np.int64)
    product = np.repeat("card", n)
    periods = pd.period_range(start="2018-01", periods=60, freq="M")
    origination_dt = periods[rs.integers(0, len(periods), n)].to_timestamp()
    maturity_months = np.repeat(120, n)  # not used for revolving but kept
    interest_rate = np.clip(0.12 + (720 - borrowers["fico_baseline"]) * 0.00025 + rs.normal(0, 0.01, n), 0.08, 0.35)
    credit_limit = np.clip((borrowers["income_annual"] / 12.0) * rs.uniform(0.2, 0.5, n), 1000, 30000)
    orig_balance = np.clip(credit_limit * borrowers["credit_utilization_baseline"] * rs.uniform(0.6, 1.0, n), 100, None)

    loans = pd.DataFrame(
        {
            "loan_id": loan_id,
            "borrower_id": borrowers["borrower_id"].values,
            "product": product,
            "origination_dt": origination_dt,
            "maturity_months": maturity_months,
            "interest_rate": interest_rate,
            "orig_balance": orig_balance,
            "secured_flag": np.repeat(False, n),
            "ltv_at_orig": np.repeat(np.nan, n),
            "risk_grade": pd.cut(borrowers["fico_baseline"], bins=[0, 639, 679, 719, 759, 850], labels=["D","C","B","A","AA"]).astype(str),
            "underwriting_dti": np.clip((orig_balance / credit_limit) + rs.normal(0.3, 0.1, n), 0.05, 0.8),
            "underwriting_fico": borrowers["fico_baseline"].values,
            "channel": rs.choice(["branch","online","mobile","other"], size=n, p=[0.3,0.35,0.3,0.05]),
            "state": borrowers["state"].values,
            "vintage": origination_dt.strftime("%Y-%m"),
            "credit_limit": credit_limit,
        }
    )
    return loans


def _simulate_card_panel(
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    months: int,
    seed: int | None = 12345,
) -> pd.DataFrame:
    rs = _rng(None if seed is None else seed + 21)

    # Build monthly index per loan
    start_month = pd.to_datetime("2020-01-01")
    panel_months = pd.date_range(start=start_month, periods=months, freq="MS")
    m = len(panel_months)

    n = len(loans)
    # Simple utilization dynamics and PD model
    # Base hazard increases with unemployment and lower fico
    fico = loans["underwriting_fico"].to_numpy()

    # Align macro series to panel months
    macro_aligned = (
        macro.set_index("asof_month").reindex(panel_months).ffill().bfill()
    )
    unemployment = macro_aligned["unemployment"].to_numpy().reshape(1, m)

    base_pd = (0.002 + 0.00003 * (720 - fico).clip(0)[:, None]) * (1.0 + 0.08 * (unemployment - unemployment.mean()))
    base_pd = np.clip(base_pd, 0.0005, 0.05)

    # Simulate defaults
    u = rs.random((n, m))
    default_matrix = (u < base_pd).astype(np.int8)
    default_cum = default_matrix.cumsum(axis=1)
    # Once defaulted, stay defaulted
    defaulted = (default_cum > 0).astype(np.int8)

    # EAD/utilization dynamics
    util = loans["orig_balance"].to_numpy() / loans["credit_limit"].to_numpy()
    util = np.clip(util, 0.05, 0.95)

    util_shocks = rs.normal(0.0, 0.03, size=(n, m)) + 0.02 * (unemployment - unemployment.mean())
    util_path = np.clip(util[:, None] + util_shocks.cumsum(axis=1), 0.01, 0.99)

    credit_limit = loans["credit_limit"].to_numpy()[:, None]
    balance_ead = util_path * credit_limit

    # Payments reduce balances modestly if not defaulted
    payment = 0.025 * credit_limit
    balance_ead = np.where(defaulted == 1, balance_ead, np.clip(balance_ead - payment, 0.0, None))

    # LGD simple rule for unsecured
    lgd = np.full((n, m), 0.85, dtype=float)

    # Build DataFrame
    records = []
    for t_idx, asof in enumerate(panel_months):
        rec = pd.DataFrame(
            {
                "asof_month": asof,
                "loan_id": loans["loan_id"].values,
                "borrower_id": loans["borrower_id"].values,
                "product": loans["product"].values,
                "balance_ead": balance_ead[:, t_idx],
                "scheduled_principal": np.zeros(n),
                "current_principal": balance_ead[:, t_idx],
                "current_interest": loans["interest_rate"].values * balance_ead[:, t_idx] / 12.0,
                "utilization": np.clip(balance_ead[:, t_idx] / credit_limit[:, 0], 0.0, 1.0),
                "prepay_flag": np.zeros(n, dtype=bool),
                "days_past_due": (defaulted[:, t_idx] * 90).astype(int),
                "roll_rate_bucket": np.where(defaulted[:, t_idx] == 1, "90+", "C"),
                "default_flag": default_matrix[:, t_idx].astype(bool),
                "chargeoff_flag": defaulted[:, t_idx].astype(bool),
                "recovery_amt": np.zeros(n),
                "recovery_lag_m": np.zeros(n),
                "cure_flag": np.zeros(n, dtype=bool),
                "loss_given_default": lgd[:, t_idx],
                "effective_rate": loans["interest_rate"].values,
                "forbearance_flag": np.zeros(n, dtype=bool),
            }
        )
        records.append(rec)

    panel = pd.concat(records, ignore_index=True)
    return panel


def generate_dataset(
    num_borrowers: int,
    months: int,
    macro: pd.DataFrame,
    seed: int | None = 12345,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    borrowers = generate_borrowers(num_borrowers, seed=seed)
    loans = generate_loans(borrowers, seed=seed)
    panel = _simulate_card_panel(loans, macro, months=months, seed=seed)
    return borrowers, loans, panel
