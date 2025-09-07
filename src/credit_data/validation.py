from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd


def _check_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    missing = [c for c in required if c not in df.columns]
    return missing


def validate_loans(df: pd.DataFrame, product: str) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    req = [
        "loan_id",
        "borrower_id",
        "product",
        "origination_dt",
        "maturity_months",
        "interest_rate",
        "orig_balance",
    ]
    missing = _check_columns(df, req)
    if missing:
        issues.append({"level": "ERROR", "scope": f"loans_{product}", "message": f"Missing columns: {missing}"})
        return issues
    # Basic ranges
    if (df["orig_balance"] < 0).any():
        issues.append({"level": "ERROR", "scope": f"loans_{product}", "message": "Negative orig_balance found"})
    if (df["interest_rate"] < 0).any():
        issues.append({"level": "ERROR", "scope": f"loans_{product}", "message": "Negative interest_rate found"})
    return issues


def validate_panel(df: pd.DataFrame, product: str) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    req = [
        "asof_month",
        "loan_id",
        "balance_ead",
        "default_flag",
        "chargeoff_flag",
    ]
    missing = _check_columns(df, req)
    if missing:
        issues.append({"level": "ERROR", "scope": f"panel_{product}", "message": f"Missing columns: {missing}"})
        return issues
    if (df["balance_ead"] < 0).any():
        issues.append({"level": "ERROR", "scope": f"panel_{product}", "message": "Negative balance_ead found"})
    # Null ratios
    null_ratio = df["balance_ead"].isna().mean()
    if null_ratio > 0.01:
        issues.append({"level": "WARN", "scope": f"panel_{product}", "message": f"balance_ead null ratio {null_ratio:.2%} > 1%"})
    return issues


def validate_dataset_dir(dataset_dir: str) -> pd.DataFrame:
    issues: List[Dict[str, str]] = []
    # Loans
    for p in sorted([f for f in os.listdir(dataset_dir) if f.startswith("loans_") and f.endswith(".parquet")]):
        prod = p.replace("loans_", "").replace(".parquet", "")
        df = pd.read_parquet(os.path.join(dataset_dir, p))
        issues.extend(validate_loans(df, prod))
    # Panels
    for p in sorted([f for f in os.listdir(dataset_dir) if f.startswith("loan_monthly_") and f.endswith(".parquet")]):
        prod = p.replace("loan_monthly_", "").replace(".parquet", "")
        df = pd.read_parquet(os.path.join(dataset_dir, p))
        issues.extend(validate_panel(df, prod))
    report = pd.DataFrame(issues, columns=["level", "scope", "message"]) if issues else pd.DataFrame(columns=["level", "scope", "message"])
    return report
