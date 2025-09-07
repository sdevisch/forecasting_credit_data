#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import re

import pandas as pd


def load_overall(path: str) -> pd.DataFrame:
    by_product_root = os.path.join(path, "portfolio_aggregates_by_product.parquet")
    cecl_overall = os.path.join(path, "cecl_multi", "portfolio_aggregates_overall.parquet")
    if os.path.exists(cecl_overall):
        df = pd.read_parquet(cecl_overall)
        return df.sort_values("asof_month")
    if os.path.exists(by_product_root):
        bp = pd.read_parquet(by_product_root)
        overall = (
            bp.groupby("asof_month", as_index=False)[["portfolio_monthly_ecl", "portfolio_ead"]].sum()
            .sort_values("asof_month")
        )
        return overall
    raise FileNotFoundError(f"No portfolio aggregates found under {path}")


def main() -> None:
    base = "data/scenario_runs"
    runs = sorted(glob.glob(os.path.join(base, "scenario_*")))
    if not runs:
        print("No scenario runs found under data/scenario_runs")
        return

    records = []
    for r in runs:
        name = re.sub(r"^scenario_", "", os.path.basename(r)).split("_")[0]
        overall = load_overall(r)
        total_ecl = overall["portfolio_monthly_ecl"].sum()
        last_cov = (overall["portfolio_monthly_ecl"].iloc[-1] / overall["portfolio_ead"].iloc[-1]) * 100.0
        records.append({
            "run": os.path.basename(r),
            "scenario": name,
            "total_monthly_ecl": total_ecl,
            "last_coverage_pct": last_cov,
        })

    out = pd.DataFrame(records)
    out_path = os.path.join(base, "scenario_comparison.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
