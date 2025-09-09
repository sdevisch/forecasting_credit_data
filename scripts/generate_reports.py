#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from credit_data.reporting import (
    coverage_by_product,
    monthly_summary_overall,
    vintage_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate portfolio coverage and vintage reports"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Dataset folder (with cecl_multi and loan files)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output folder for CSV reports; defaults to <input>/reports",
    )
    args = parser.parse_args()

    in_dir = args.input
    cecl_dir = os.path.join(in_dir, "cecl_multi")
    out_dir = args.out or os.path.join(in_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)

    cov = coverage_by_product(cecl_dir)
    cov.to_csv(os.path.join(out_dir, "coverage_by_product.csv"), index=False)

    overall = monthly_summary_overall(cecl_dir)
    overall.to_csv(os.path.join(out_dir, "summary_overall.csv"), index=False)

    vint = vintage_summary(in_dir)
    vint.to_csv(os.path.join(out_dir, "vintage_summary.csv"), index=False)

    print(f"Wrote reports to {out_dir}")


if __name__ == "__main__":
    main()
