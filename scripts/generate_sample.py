#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd

from credit_data.macro import get_macro_data
from credit_data.generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit dataset sample")
    parser.add_argument("--n_borrowers", type=int, default=10000)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default="2024-12-01")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    print("Fetching macro data ...")
    macro = get_macro_data(args.start, args.end)

    print("Generating synthetic dataset ...")
    borrowers, loans, panel = generate_dataset(
        num_borrowers=args.n_borrowers,
        months=args.months,
        macro=macro,
        seed=args.seed,
    )

    ts = datetime.now().strftime("sample_%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, ts)
    os.makedirs(out_path, exist_ok=True)

    print(f"Writing outputs to {out_path} ...")
    borrowers.to_parquet(os.path.join(out_path, "borrowers.parquet"))
    loans.to_parquet(os.path.join(out_path, "loans.parquet"))
    panel.to_parquet(os.path.join(out_path, "loan_monthly.parquet"))
    macro.to_parquet(os.path.join(out_path, "macro.parquet"))

    # Also small CSV heads for quick inspection
    borrowers.head(1000).to_csv(os.path.join(out_path, "borrowers_head.csv"), index=False)
    loans.head(1000).to_csv(os.path.join(out_path, "loans_head.csv"), index=False)
    panel.head(1000).to_csv(os.path.join(out_path, "loan_monthly_head.csv"), index=False)

    print("Done.")


if __name__ == "__main__":
    main()
