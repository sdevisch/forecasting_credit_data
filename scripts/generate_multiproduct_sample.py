#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd

from credit_data.macro import get_macro_data
from credit_data.generator import generate_borrowers, generate_loans, _simulate_card_panel
from credit_data.products.auto import generate_auto_loans, simulate_auto_panel
from credit_data.products.personal import generate_personal_loans, simulate_personal_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-product synthetic dataset sample")
    parser.add_argument("--n_borrowers", type=int, default=20000)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default="2024-12-01")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    print("Fetching macro data ...")
    macro = get_macro_data(args.start, args.end)

    print("Generating borrowers ...")
    borrowers = generate_borrowers(args.n_borrowers, seed=args.seed)

    print("Generating loans for card, auto, personal ...")
    card_loans = generate_loans(borrowers, seed=args.seed)
    auto_loans = generate_auto_loans(borrowers, seed=args.seed)
    personal_loans = generate_personal_loans(borrowers, seed=args.seed)

    print("Simulating monthly panels ...")
    card_panel = _simulate_card_panel(card_loans, macro, months=args.months, seed=args.seed)
    auto_panel = simulate_auto_panel(auto_loans, macro, months=args.months, seed=args.seed)
    personal_panel = simulate_personal_panel(personal_loans, macro, months=args.months, seed=args.seed)

    ts = datetime.now().strftime("sample_multi_%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, ts)
    os.makedirs(out_path, exist_ok=True)

    print(f"Writing outputs to {out_path} ...")
    borrowers.to_parquet(os.path.join(out_path, "borrowers.parquet"))
    card_loans.to_parquet(os.path.join(out_path, "loans_card.parquet"))
    auto_loans.to_parquet(os.path.join(out_path, "loans_auto.parquet"))
    personal_loans.to_parquet(os.path.join(out_path, "loans_personal.parquet"))
    card_panel.to_parquet(os.path.join(out_path, "loan_monthly_card.parquet"))
    auto_panel.to_parquet(os.path.join(out_path, "loan_monthly_auto.parquet"))
    personal_panel.to_parquet(os.path.join(out_path, "loan_monthly_personal.parquet"))

    print("Done.")


if __name__ == "__main__":
    main()
