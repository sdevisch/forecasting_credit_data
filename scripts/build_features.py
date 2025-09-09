#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import glob

import pandas as pd

from credit_data.features import add_lags, add_leads, add_rolling, add_interactions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build features from loan monthly panels"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset folder with loan_monthly_*.parquet",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output folder; defaults to <input>/features",
    )
    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.out or os.path.join(in_dir, "features")
    os.makedirs(out_dir, exist_ok=True)

    panel_files = glob.glob(os.path.join(in_dir, "loan_monthly_*.parquet"))
    for path in panel_files:
        product = (
            os.path.basename(path).replace("loan_monthly_", "").replace(".parquet", "")
        )
        df = pd.read_parquet(path)
        df = df.sort_values(["loan_id", "asof_month"])  # ensure temporal order

        # Build core features
        cols_numeric = [
            "balance_ead",
            "current_principal",
            "current_interest",
            "utilization",
            "days_past_due",
        ]
        df_f = add_lags(df, ["loan_id"], "asof_month", cols_numeric, lags=[1, 3, 6])
        df_f = add_leads(
            df_f,
            ["loan_id"],
            "asof_month",
            ["default_flag", "chargeoff_flag"],
            leads=[1, 3, 6],
        )
        df_f = add_rolling(
            df_f,
            ["loan_id"],
            "asof_month",
            ["utilization", "current_interest"],
            windows=[3, 6],
        )
        df_f = add_interactions(
            df_f,
            interactions=[
                ("utilization", "days_past_due", "util_x_dpd"),
                ("current_interest", "balance_ead", "int_x_ead"),
            ],
        )

        out_path = os.path.join(out_dir, f"features_{product}.parquet")
        df_f.to_parquet(out_path)

    print(f"Wrote features to {out_dir}")


if __name__ == "__main__":
    main()
