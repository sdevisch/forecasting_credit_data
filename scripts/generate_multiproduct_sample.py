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
from credit_data.products.mortgage import generate_mortgage_loans, simulate_mortgage_panel
from credit_data.products.heloc import generate_heloc_loans, simulate_heloc_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-product synthetic dataset sample")
    parser.add_argument("--n_borrowers", type=int, default=20000)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default="2024-12-01")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--partitioned", action="store_true", help="Write partitioned Parquet outputs")
    args = parser.parse_args()

    print("Fetching macro data ...")
    macro = get_macro_data(args.start, args.end)

    print("Generating borrowers ...")
    borrowers = generate_borrowers(args.n_borrowers, seed=args.seed)

    print("Generating loans for card, auto, personal, mortgage, heloc ...")
    card_loans = generate_loans(borrowers, seed=args.seed)
    auto_loans = generate_auto_loans(borrowers, seed=args.seed)
    personal_loans = generate_personal_loans(borrowers, seed=args.seed)
    mortgage_loans = generate_mortgage_loans(borrowers, seed=args.seed)
    heloc_loans = generate_heloc_loans(borrowers, seed=args.seed)

    print("Simulating monthly panels ...")
    card_panel = _simulate_card_panel(card_loans, macro, months=args.months, seed=args.seed)
    auto_panel = simulate_auto_panel(auto_loans, macro, months=args.months, seed=args.seed)
    personal_panel = simulate_personal_panel(personal_loans, macro, months=args.months, seed=args.seed)
    mortgage_panel = simulate_mortgage_panel(mortgage_loans, macro, months=args.months, seed=args.seed)
    heloc_panel = simulate_heloc_panel(heloc_loans, macro, months=args.months, seed=args.seed)

    ts = datetime.now().strftime("sample_multi_%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, ts)
    os.makedirs(out_path, exist_ok=True)

    print(f"Writing outputs to {out_path} ...")
    borrowers.to_parquet(os.path.join(out_path, "borrowers.parquet"))

    # Loans
    if args.partitioned:
        loans_all = pd.concat([
            card_loans.assign(product="card"),
            auto_loans.assign(product="auto"),
            personal_loans.assign(product="personal"),
            mortgage_loans.assign(product="mortgage"),
            heloc_loans.assign(product="heloc"),
        ], ignore_index=True)
        loans_all.to_parquet(os.path.join(out_path, "loans_partitioned.parquet"), partition_cols=["product"])
    else:
        card_loans.to_parquet(os.path.join(out_path, "loans_card.parquet"))
        auto_loans.to_parquet(os.path.join(out_path, "loans_auto.parquet"))
        personal_loans.to_parquet(os.path.join(out_path, "loans_personal.parquet"))
        mortgage_loans.to_parquet(os.path.join(out_path, "loans_mortgage.parquet"))
        heloc_loans.to_parquet(os.path.join(out_path, "loans_heloc.parquet"))

    # Panels
    if args.partitioned:
        panels_all = pd.concat([
            card_panel.assign(product="card"),
            auto_panel.assign(product="auto"),
            personal_panel.assign(product="personal"),
            mortgage_panel.assign(product="mortgage"),
            heloc_panel.assign(product="heloc"),
        ], ignore_index=True)
        panels_all.to_parquet(os.path.join(out_path, "loan_monthly_partitioned.parquet"), partition_cols=["product", "asof_month"])
    else:
        card_panel.to_parquet(os.path.join(out_path, "loan_monthly_card.parquet"))
        auto_panel.to_parquet(os.path.join(out_path, "loan_monthly_auto.parquet"))
        personal_panel.to_parquet(os.path.join(out_path, "loan_monthly_personal.parquet"))
        mortgage_panel.to_parquet(os.path.join(out_path, "loan_monthly_mortgage.parquet"))
        heloc_panel.to_parquet(os.path.join(out_path, "loan_monthly_heloc.parquet"))

    print("Done.")


if __name__ == "__main__":
    main()
