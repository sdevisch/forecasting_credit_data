#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List

import pandas as pd

from credit_data.macro import get_macro_data
from credit_data.generator import generate_borrowers, generate_loans, _simulate_card_panel
from credit_data.products.auto import generate_auto_loans, simulate_auto_panel
from credit_data.products.personal import generate_personal_loans, simulate_personal_panel
from credit_data.products.mortgage import generate_mortgage_loans, simulate_mortgage_panel
from credit_data.products.heloc import generate_heloc_loans, simulate_heloc_panel
from credit_data.logging_utils import get_logger, timed
from credit_data.manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-product synthetic dataset sample")
    parser.add_argument("--n_borrowers", type=int, default=20000)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default="2024-12-01")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--partitioned", action="store_true", help="Write partitioned Parquet outputs")
    parser.add_argument("--products", type=str, default="card,auto,personal,mortgage,heloc", help="Comma-separated products to include")
    args = parser.parse_args()

    logger = get_logger("generator")

    products: List[str] = [p.strip() for p in args.products.split(",") if p.strip()]

    with timed(logger, "fetch_macro"):
        macro = get_macro_data(args.start, args.end)

    with timed(logger, "generate_borrowers"):
        borrowers = generate_borrowers(args.n_borrowers, seed=args.seed)

    loans_map = {}
    panels_map = {}
    with timed(logger, "generate_loans_panels"):
        if "card" in products:
            loans_map["card"] = generate_loans(borrowers, seed=args.seed)
            panels_map["card"] = _simulate_card_panel(loans_map["card"], macro, months=args.months, seed=args.seed)
        if "auto" in products:
            loans_map["auto"] = generate_auto_loans(borrowers, seed=args.seed)
            panels_map["auto"] = simulate_auto_panel(loans_map["auto"], macro, months=args.months, seed=args.seed)
        if "personal" in products:
            loans_map["personal"] = generate_personal_loans(borrowers, seed=args.seed)
            panels_map["personal"] = simulate_personal_panel(loans_map["personal"], macro, months=args.months, seed=args.seed)
        if "mortgage" in products:
            loans_map["mortgage"] = generate_mortgage_loans(borrowers, seed=args.seed)
            panels_map["mortgage"] = simulate_mortgage_panel(loans_map["mortgage"], macro, months=args.months, seed=args.seed)
        if "heloc" in products:
            loans_map["heloc"] = generate_heloc_loans(borrowers, seed=args.seed)
            panels_map["heloc"] = simulate_heloc_panel(loans_map["heloc"], macro, months=args.months, seed=args.seed)

    ts = datetime.now().strftime("sample_multi_%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, ts)
    os.makedirs(out_path, exist_ok=True)

    logger.info(f"Writing outputs to {out_path} ...")
    with timed(logger, "write_outputs"):
        borrowers.to_parquet(os.path.join(out_path, "borrowers.parquet"))
        if args.partitioned:
            loans_all = pd.concat([df.assign(product=prod) for prod, df in loans_map.items()], ignore_index=True)
            loans_all.to_parquet(os.path.join(out_path, "loans_partitioned.parquet"), partition_cols=["product"]) 
        else:
            for prod, df in loans_map.items():
                df.to_parquet(os.path.join(out_path, f"loans_{prod}.parquet"))
        if args.partitioned:
            panels_all = pd.concat([df.assign(product=prod) for prod, df in panels_map.items()], ignore_index=True)
            panels_all.to_parquet(os.path.join(out_path, "loan_monthly_partitioned.parquet"), partition_cols=["product", "asof_month"]) 
        else:
            for prod, df in panels_map.items():
                df.to_parquet(os.path.join(out_path, f"loan_monthly_{prod}.parquet"))

    # Write metadata manifest
    write_manifest(out_path, params={
        "n_borrowers": args.n_borrowers,
        "months": args.months,
        "start": args.start,
        "end": args.end,
        "seed": args.seed,
        "partitioned": args.partitioned,
        "products": products,
    })

    logger.info("Done.")


if __name__ == "__main__":
    main()
