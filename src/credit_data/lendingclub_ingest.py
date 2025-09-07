from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd


LC_TO_BORROWER = {
    "member_id": "borrower_id",
    "addr_state": "state",
    "annual_inc": "income_annual",
    "emp_length": "employment_tenure_months",
    "fico_range_high": "fico_baseline",
}

LC_TO_LOAN = {
    "id": "loan_id",
    "member_id": "borrower_id",
    "issue_d": "origination_dt",
    "term": "maturity_months",
    "int_rate": "interest_rate",
    "loan_amnt": "orig_balance",
    "dti": "underwriting_dti",
    "addr_state": "state",
}


def _clean_emp_length(val: str | float | int) -> int:
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return int(max(val, 0))
    s = str(val)
    if s == "< 1 year":
        return 6
    if s == "10+ years":
        return 120
    try:
        num = int(s.strip().split()[0])
        return num * 12
    except Exception:
        return 0


def _clean_term(val: str | int | float) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val)
    return int(s.strip().split()[0])


def _clean_int_rate(val: str | float) -> float:
    if isinstance(val, (int, float)):
        return float(val) / 100.0 if val > 1 else float(val)
    s = str(val).replace("%", "")
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def map_borrowers(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for lc_col, our_col in LC_TO_BORROWER.items():
        if lc_col in df.columns:
            out[our_col] = df[lc_col]
        else:
            out[our_col] = pd.NA
    out["employment_tenure_months"] = df.get("emp_length", pd.Series([pd.NA] * len(df))).apply(_clean_emp_length)
    out["fico_baseline"] = df.get("fico_range_high", pd.Series([pd.NA] * len(df))).fillna(680).astype(float).round().astype("Int64")
    out["credit_utilization_baseline"] = pd.NA
    out["prior_delinquencies"] = pd.NA
    out["bank_tenure_months"] = pd.NA
    out["segment"] = pd.NA
    out["zip3"] = pd.NA
    return out


def map_loans(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for lc_col, our_col in LC_TO_LOAN.items():
        if lc_col in df.columns:
            out[our_col] = df[lc_col]
        else:
            out[our_col] = pd.NA
    out["product"] = "personal"
    out["origination_dt"] = pd.to_datetime(out["origination_dt"], errors="coerce")
    out["maturity_months"] = df.get("term", pd.Series([pd.NA] * len(df))).apply(_clean_term)
    out["interest_rate"] = df.get("int_rate", pd.Series([pd.NA] * len(df))).apply(_clean_int_rate)
    out["secured_flag"] = False
    out["ltv_at_orig"] = pd.NA
    out["risk_grade"] = df.get("grade", pd.Series([pd.NA] * len(df)))
    out["underwriting_fico"] = df.get("fico_range_high", pd.Series([pd.NA] * len(df))).fillna(680).astype(float).round().astype("Int64")
    out["channel"] = pd.NA
    out["vintage"] = out["origination_dt"].dt.to_period("M").astype(str)
    out["credit_limit"] = pd.NA
    return out


def ingest_lendingclub_csv(csv_path: str, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, low_memory=False)
    borrowers = map_borrowers(df)
    loans = map_loans(df)

    b_path = os.path.join(out_dir, "borrowers_lc.parquet")
    l_path = os.path.join(out_dir, "loans_lc.parquet")
    borrowers.to_parquet(b_path)
    loans.to_parquet(l_path)
    return b_path, l_path
