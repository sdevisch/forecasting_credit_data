from __future__ import annotations

import pandas as pd

from credit_data.features import add_lags, add_leads, add_rolling, add_interactions


def make_panel(n_loans: int = 5, months: int = 6) -> pd.DataFrame:
    rows = []
    for lid in range(1, n_loans + 1):
        for m in range(months):
            rows.append(
                {
                    "loan_id": lid,
                    "asof_month": pd.Timestamp("2020-01-01") + pd.offsets.MonthBegin(m),
                    "balance_ead": 1000 + m,
                    "current_interest": 10 + m,
                    "utilization": 0.5,
                    "days_past_due": 0,
                    "default_flag": False,
                    "chargeoff_flag": False,
                }
            )
    return pd.DataFrame(rows)


def test_add_lags_leads_and_rolling():
    df = make_panel()
    df1 = add_lags(df, ["loan_id"], "asof_month", ["balance_ead"], lags=[1, 2])
    assert "balance_ead_lag1" in df1.columns and "balance_ead_lag2" in df1.columns

    df2 = add_leads(df1, ["loan_id"], "asof_month", ["default_flag"], leads=[1])
    assert "default_flag_lead1" in df2.columns

    df3 = add_rolling(df2, ["loan_id"], "asof_month", ["current_interest"], windows=[3])
    assert "current_interest_roll3_mean" in df3.columns
    assert "current_interest_roll3_std" in df3.columns


def test_add_interactions():
    df = make_panel()
    df4 = add_interactions(df, [("current_interest", "balance_ead", "int_x_ead")])
    assert "int_x_ead" in df4.columns
