#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import pandas as pd

from credit_data.cecl import (
    compute_monthly_ecl,
    compute_lifetime_ecl,
    compute_portfolio_aggregates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CECL calculation on a dataset folder"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset folder with loan_monthly.parquet",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output folder; defaults to <input>/cecl"
    )
    args = parser.parse_args()

    input_dir = args.input
    out_dir = args.out or os.path.join(input_dir, "cecl")
    os.makedirs(out_dir, exist_ok=True)

    panel_path = os.path.join(input_dir, "loan_monthly.parquet")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Missing panel parquet at {panel_path}")

    panel = pd.read_parquet(panel_path)

    monthly = compute_monthly_ecl(panel)
    lifetime = compute_lifetime_ecl(monthly)
    portfolio = compute_portfolio_aggregates(monthly)

    monthly.to_parquet(os.path.join(out_dir, "monthly_ecl.parquet"))
    lifetime.to_parquet(os.path.join(out_dir, "lifetime_ecl.parquet"))
    portfolio.to_parquet(os.path.join(out_dir, "portfolio_aggregates.parquet"))

    print(f"Wrote CECL outputs to {out_dir}")


if __name__ == "__main__":
    main()
