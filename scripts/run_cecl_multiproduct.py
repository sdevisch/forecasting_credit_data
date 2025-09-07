#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import glob

import pandas as pd

from credit_data.cecl import (
    compute_monthly_ecl,
    compute_lifetime_ecl,
    compute_portfolio_aggregates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CECL across multiple product panel files")
    parser.add_argument("--input", type=str, required=True, help="Path to dataset folder with loan_monthly_*.parquet files")
    parser.add_argument("--out", type=str, default=None, help="Output folder; defaults to <input>/cecl_multi")
    args = parser.parse_args()

    input_dir = args.input
    out_dir = args.out or os.path.join(input_dir, "cecl_multi")
    os.makedirs(out_dir, exist_ok=True)

    panel_paths = sorted(glob.glob(os.path.join(input_dir, "loan_monthly_*.parquet")))
    if not panel_paths:
        raise FileNotFoundError(f"No per-product panel files found in {input_dir}")

    monthly_list = []
    lifetime_list = []
    portfolio_list = []

    for path in panel_paths:
        product = os.path.basename(path).replace("loan_monthly_", "").replace(".parquet", "")
        panel = pd.read_parquet(path)
        m = compute_monthly_ecl(panel)
        m["product"] = product
        l = compute_lifetime_ecl(m)
        l["product"] = product
        p = compute_portfolio_aggregates(m)
        p["product"] = product

        monthly_list.append(m)
        lifetime_list.append(l)
        portfolio_list.append(p)

    monthly_all = pd.concat(monthly_list, ignore_index=True)
    lifetime_all = pd.concat(lifetime_list, ignore_index=True)
    portfolio_all = pd.concat(portfolio_list, ignore_index=True)

    # Also produce overall portfolio aggregates across products by month
    portfolio_overall = (
        monthly_all.groupby(["asof_month"], as_index=False)[["monthly_ecl", "ead_t"]].sum()
        .rename(columns={"monthly_ecl": "portfolio_monthly_ecl", "ead_t": "portfolio_ead"})
    )

    monthly_all.to_parquet(os.path.join(out_dir, "monthly_ecl_all.parquet"))
    lifetime_all.to_parquet(os.path.join(out_dir, "lifetime_ecl_all.parquet"))
    portfolio_all.to_parquet(os.path.join(out_dir, "portfolio_aggregates_by_product.parquet"))
    portfolio_overall.to_parquet(os.path.join(out_dir, "portfolio_aggregates_overall.parquet"))

    print(f"Wrote multi-product CECL outputs to {out_dir}")


if __name__ == "__main__":
    main()
