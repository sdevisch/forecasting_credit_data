#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import yaml  # type: ignore

import numpy as np
import pandas as pd

from credit_data.curve_calibration import compute_scalers, apply_monthly_ecl_scaling


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply target curve calibration to monthly ECLs"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to cecl_multi folder with monthly_ecl_all.parquet",
    )
    parser.add_argument(
        "--targets",
        type=str,
        required=True,
        help="YAML with product target cumulative curves",
    )
    parser.add_argument("--months", type=int, default=12)
    args = parser.parse_args()

    in_dir = args.input
    monthly_path = os.path.join(in_dir, "monthly_ecl_all.parquet")
    if not os.path.exists(monthly_path):
        raise FileNotFoundError(f"Missing {monthly_path}")
    monthly = pd.read_parquet(monthly_path)

    with open(args.targets, "r") as f:
        targets = yaml.safe_load(f).get("targets", {})

    out_dir = os.path.join(in_dir, "calibrated")
    os.makedirs(out_dir, exist_ok=True)

    out_scaled = []
    for product, curve in targets.items():
        curve = np.asarray(curve, dtype=float)
        # Build model monthly hazard proxy from default flags per month index
        sub = monthly[monthly["product"] == product].copy()
        sub = sub.sort_values(["loan_id", "asof_month"])  # ensure order
        sub["month_idx"] = sub.groupby("loan_id").cumcount()
        hazards = (
            sub.groupby("month_idx")["default_flag"]
            .mean()
            .reindex(range(len(curve)))
            .fillna(0.0)
            .to_numpy()
        )
        scalers = compute_scalers(hazards, curve)
        scaled = apply_monthly_ecl_scaling(sub, scalers, months=len(curve))
        out_scaled.append(scaled)

    scaled_all = pd.concat(out_scaled, ignore_index=True)
    scaled_all.to_parquet(os.path.join(out_dir, "monthly_ecl_all_calibrated.parquet"))
    print(f"Wrote calibrated monthly ECLs to {out_dir}")


if __name__ == "__main__":
    main()
