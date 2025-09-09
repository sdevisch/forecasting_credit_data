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
from credit_data.logging_utils import get_logger, timed


def load_panels(input_dir: str) -> pd.DataFrame:
    part_path = os.path.join(input_dir, "loan_monthly_partitioned.parquet")
    if os.path.exists(part_path):
        return pd.read_parquet(part_path)
    panel_paths = sorted(glob.glob(os.path.join(input_dir, "loan_monthly_*.parquet")))
    if not panel_paths:
        raise FileNotFoundError(f"No panel files found under {input_dir}")
    frames = []
    for path in panel_paths:
        product = (
            os.path.basename(path).replace("loan_monthly_", "").replace(".parquet", "")
        )
        df = pd.read_parquet(path)
        if "product" not in df.columns:
            df["product"] = product
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CECL across multiple product panel files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset folder with loan_monthly_*.parquet files or partitioned",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output folder; defaults to <input>/cecl_multi",
    )
    args = parser.parse_args()

    logger = get_logger("cecl_runner")

    input_dir = args.input
    out_dir = args.out or os.path.join(input_dir, "cecl_multi")
    os.makedirs(out_dir, exist_ok=True)

    with timed(logger, "load_panels"):
        panel = load_panels(input_dir)

    with timed(logger, "compute_monthly"):
        monthly = compute_monthly_ecl(panel)
    with timed(logger, "compute_lifetime"):
        lifetime = compute_lifetime_ecl(monthly)

    products = monthly["product"].unique()
    portfolio_list = []
    with timed(logger, "compute_aggregates"):
        for prod in products:
            m_prod = monthly[monthly["product"] == prod]
            p = compute_portfolio_aggregates(m_prod)
            p["product"] = prod
            portfolio_list.append(p)
        portfolio_all = pd.concat(portfolio_list, ignore_index=True)
        overall = compute_portfolio_aggregates(monthly)

    with timed(logger, "write_outputs"):
        monthly.to_parquet(os.path.join(out_dir, "monthly_ecl.parquet"))
        lifetime.to_parquet(os.path.join(out_dir, "lifetime_ecl.parquet"))
        portfolio_all.to_parquet(
            os.path.join(out_dir, "portfolio_aggregates_by_product.parquet")
        )
        overall.to_parquet(
            os.path.join(out_dir, "portfolio_aggregates_overall.parquet")
        )

    logger.info(f"Wrote CECL outputs to {out_dir}")


if __name__ == "__main__":
    main()
