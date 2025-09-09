from __future__ import annotations

import numpy as np
import pandas as pd


US_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
SEGMENTS = ["mass", "affluent", "small_business", "private", "student"]
PRODUCTS = ["card"]  # initial prototype focuses on card; extendable
INDUSTRIES = [
    "tech",
    "finance",
    "healthcare",
    "education",
    "manufacturing",
    "retail",
    "hospitality",
    "transport",
    "construction",
    "other",
]


def _rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = 12345
    return np.random.default_rng(seed)


def generate_borrowers(num_borrowers: int, seed: int | None = 12345) -> pd.DataFrame:
    rs = _rng(seed)
    borrower_id = np.arange(1, num_borrowers + 1, dtype=np.int64)
    state = rs.choice(US_STATES, size=num_borrowers, replace=True)
    income_annual = rs.lognormal(
        mean=10.5, sigma=0.5, size=num_borrowers
    )  # ~ $36k-$200k
    employment_tenure_months = rs.integers(0, 360, size=num_borrowers)
    industry = rs.choice(INDUSTRIES, size=num_borrowers, replace=True)
    education = rs.choice(
        ["hs", "college", "grad"], size=num_borrowers, p=[0.35, 0.45, 0.2]
    )
    household_size = rs.integers(1, 6, size=num_borrowers)

    # Correlated fico/utilization using a simple factor model
    systemic = rs.standard_normal(num_borrowers)
    fico_raw = 690 + 60 * systemic + rs.normal(0, 40, num_borrowers)
    fico_baseline = np.clip(fico_raw, 500, 850).astype(int)
    util_base = np.clip(
        0.35 + 0.1 * (-systemic) + rs.normal(0, 0.15, num_borrowers), 0.0, 1.0
    )

    prior_delinquencies = rs.poisson(
        lam=np.clip((720 - fico_baseline) / 200.0, 0.05, 2.0)
    )
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
    interest_rate = np.clip(
        0.12 + (720 - borrowers["fico_baseline"]) * 0.00025 + rs.normal(0, 0.01, n),
        0.08,
        0.35,
    )
    credit_limit = np.clip(
        (borrowers["income_annual"] / 12.0) * rs.uniform(0.2, 0.5, n), 1000, 30000
    )
    orig_balance = np.clip(
        credit_limit
        * borrowers["credit_utilization_baseline"]
        * rs.uniform(0.6, 1.0, n),
        100,
        None,
    )

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
            "risk_grade": pd.cut(
                borrowers["fico_baseline"],
                bins=[0, 639, 679, 719, 759, 850],
                labels=["D", "C", "B", "A", "AA"],
            ).astype(str),
            "underwriting_dti": np.clip(
                (orig_balance / credit_limit) + rs.normal(0.3, 0.1, n), 0.05, 0.8
            ),
            "underwriting_fico": borrowers["fico_baseline"].values,
            "channel": rs.choice(
                ["branch", "online", "mobile", "other"],
                size=n,
                p=[0.3, 0.35, 0.3, 0.05],
            ),
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
    fico = loans["underwriting_fico"].to_numpy()

    # Align macro series to panel months
    macro_aligned = macro.set_index("asof_month").reindex(panel_months).ffill().bfill()
    unemp = macro_aligned["unemployment"].to_numpy()
    unemp_dev = unemp - unemp.mean()
    fed = (
        macro_aligned.get("fed_funds").to_numpy()
        if "fed_funds" in macro_aligned.columns
        else np.zeros(m)
    )
    fed_dev = fed - (fed.mean() if fed.size > 0 else 0.0)

    # Utilization and EAD path pre-simulation
    util0 = loans["orig_balance"].to_numpy() / loans["credit_limit"].to_numpy()
    util0 = np.clip(util0, 0.05, 0.95)

    # Rate-sensitive utilization shocks: higher rates -> lower spend/utilization
    util_shocks = (
        rs.normal(0.0, 0.025, size=(n, m))
        + 0.015 * unemp_dev.reshape(1, m)
        - 0.01 * fed_dev.reshape(1, m)
    )
    util_path = np.clip(util0[:, None] + util_shocks.cumsum(axis=1), 0.01, 0.99)
    credit_limit = loans["credit_limit"].to_numpy()[:, None]
    balance_path = util_path * credit_limit

    # State machine: 0=C,1=30,2=60,3=90+,4=CO
    state = np.zeros(n, dtype=np.int8)

    records = []
    for t_idx, asof in enumerate(panel_months):
        # Seasoning factor: early months slightly higher roll risk; stabilizes by 12m
        seasoning = min((t_idx + 1) / 12.0, 1.0)
        risk = (720 - fico).clip(0)
        udev = unemp_dev[t_idx]

        # Transition probabilities influenced by risk, macro, seasoning
        p_c_to_30 = np.clip(
            (0.01 + 0.00006 * risk + 0.01 * max(0.0, udev)) * seasoning, 0.001, 0.25
        )
        p_30_to_60 = np.clip(
            (0.08 + 0.00008 * risk + 0.01 * max(0.0, udev)) * seasoning, 0.01, 0.35
        )
        p_60_to_90 = np.clip(
            (0.12 + 0.00010 * risk + 0.015 * max(0.0, udev)) * seasoning, 0.02, 0.45
        )
        p_90_to_co = np.clip(
            (0.18 + 0.00012 * risk + 0.02 * max(0.0, udev)) * seasoning, 0.03, 0.6
        )

        p_30_cure = np.clip(0.30 - 0.0002 * risk - 0.01 * max(0.0, udev), 0.02, 0.6)
        p_60_cure = np.clip(0.18 - 0.00015 * risk - 0.008 * max(0.0, udev), 0.01, 0.4)
        p_90_cure = np.clip(0.06 - 0.0001 * risk - 0.005 * max(0.0, udev), 0.0, 0.2)

        # Transitions
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
        state[trans_90_to_co] = 4
        state[trans_90_cure] = 0

        # Compute balances and payments
        bal_t = balance_path[:, t_idx]
        # Base minimum payment
        min_payment = 0.02 * credit_limit[:, 0]
        # Prepayment probability: higher when utilization high and rates high
        util_now = np.clip(bal_t / credit_limit[:, 0], 0.0, 1.0)
        prepay_base = 0.01 + 0.02 * util_now + 0.01 * np.clip(fed_dev[t_idx], 0.0, None)
        prepay_prob = np.clip(prepay_base, 0.0, 0.4)
        prepay_flag = rs.random(n) < prepay_prob
        extra_payment = np.where(prepay_flag, 0.05 * credit_limit[:, 0], 0.0)

        payment = min_payment + extra_payment
        bal_t = np.where(state == 4, 0.0, np.clip(bal_t - payment, 0.0, None))
        interest_t = loans["interest_rate"].to_numpy() * bal_t / 12.0

        roll_bucket = np.where(
            state == 0,
            "C",
            np.where(
                state == 1,
                "30",
                np.where(state == 2, "60", np.where(state == 3, "90+", "CO")),
            ),
        )
        dpp = np.where(
            state == 0,
            0,
            np.where(
                state == 1, 30, np.where(state == 2, 60, np.where(state == 3, 90, 120))
            ),
        )

        lgd_t = np.where(state == 4, 0.88, 0.88)

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
                "utilization": np.clip(
                    np.divide(
                        bal_t,
                        credit_limit[:, 0],
                        out=np.zeros_like(bal_t),
                        where=credit_limit[:, 0] != 0,
                    ),
                    0.0,
                    1.0,
                ),
                "prepay_flag": prepay_flag,
                "days_past_due": dpp.astype(int),
                "roll_rate_bucket": roll_bucket,
                "default_flag": trans_90_to_co.astype(bool),
                "chargeoff_flag": (state == 4),
                "recovery_amt": np.zeros(n),
                "recovery_lag_m": np.zeros(n, dtype=int),
                "cure_flag": (trans_30_cure | trans_60_cure | trans_90_cure),
                "loss_given_default": lgd_t,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    borrowers = generate_borrowers(num_borrowers, seed=seed)
    loans = generate_loans(borrowers, seed=seed)
    panel = _simulate_card_panel(loans, macro, months=months, seed=seed)
    return borrowers, loans, panel
