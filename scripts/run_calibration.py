#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import glob

import pandas as pd

from credit_data.calibration import distribution_summary, roll_rate_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute basic calibration summaries from dataset folder")
    parser.add_argument("--input", type=str, required=True, help="Path to dataset folder with loan_monthly_*.parquet and loans_*.parquet")
    args = parser.parse_args()

    in_dir = args.input
    out_dir = os.path.join(in_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)

    # Summaries for loans across products
    loan_files = glob.glob(os.path.join(in_dir, "loans_*.parquet"))
    loan_dfs = [pd.read_parquet(f) for f in loan_files]
    loans_all = pd.concat(loan_dfs, ignore_index=True)
    loans_summary = distribution_summary(
        loans_all,
        [
            "underwriting_fico",
            "underwriting_dti",
            "orig_balance",
            "interest_rate",
            "ltv_at_orig",
        ],
    )
    loans_summary.to_parquet(os.path.join(out_dir, "loans_distribution_summary.parquet"))

    # Roll rate matrices per product
    panel_files = glob.glob(os.path.join(in_dir, "loan_monthly_*.parquet"))
    for f in panel_files:
        product = os.path.basename(f).replace("loan_monthly_", "").replace(".parquet", "")
        panel = pd.read_parquet(f)
        rr = roll_rate_matrix(panel)
        rr.to_parquet(os.path.join(out_dir, f"roll_rates_{product}.parquet"))

    print(f"Wrote calibration outputs to {out_dir}")


if __name__ == "__main__":
    main()
