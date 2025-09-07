#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from credit_data.spark_pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PySpark pipeline over panel parquet files")
    parser.add_argument("--input", type=str, required=True, help="Input dataset directory with loan_monthly_*.parquet")
    parser.add_argument("--out", type=str, required=True, help="Output directory for Spark aggregates")
    args = parser.parse_args()

    run_pipeline(args.input, args.out)
    print(f"Wrote outputs to {args.out}")


if __name__ == "__main__":
    main()
