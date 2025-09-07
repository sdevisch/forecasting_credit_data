#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime

import yaml  # type: ignore
import pandas as pd

from credit_data.macro import get_macro_data
from credit_data.generator import generate_borrowers, generate_loans, _simulate_card_panel
from credit_data.products.auto import generate_auto_loans, simulate_auto_panel
from credit_data.products.personal import generate_personal_loans, simulate_personal_panel
from credit_data.products.mortgage import generate_mortgage_loans, simulate_mortgage_panel
from credit_data.products.heloc import generate_heloc_loans, simulate_heloc_panel
from credit_data.cecl import compute_monthly_ecl, compute_portfolio_aggregates


def adjust_macro(df: pd.DataFrame, unemployment_add: float = 0.0, hpi_yoy_add: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    if "unemployment" in out.columns:
        out["unemployment"] = out["unemployment"] + float(unemployment_add)
    if "hpi_yoy" in out.columns:
        out["hpi_yoy"] = out["hpi_yoy"] + float(hpi_yoy_add)
    return out


def run_scenario(name: str, n_borrowers: int, months: int, macro_adj: dict, out_root: str) -> str:
    macro = get_macro_data("2019-01-01", "2024-12-01")
    macro = adjust_macro(macro, macro_adj.get("unemployment_add", 0.0), macro_adj.get("hpi_yoy_add", 0.0))

    borrowers = generate_borrowers(n_borrowers)
    card_loans = generate_loans(borrowers)
    auto_loans = generate_auto_loans(borrowers)
    personal_loans = generate_personal_loans(borrowers)
    mortgage_loans = generate_mortgage_loans(borrowers)
    heloc_loans = generate_heloc_loans(borrowers)

    card_panel = _simulate_card_panel(card_loans, macro, months=months)
    auto_panel = simulate_auto_panel(auto_loans, macro, months=months)
    personal_panel = simulate_personal_panel(personal_loans, macro, months=months)
    mortgage_panel = simulate_mortgage_panel(mortgage_loans, macro, months=months)
    heloc_panel = simulate_heloc_panel(heloc_loans, macro, months=months)

    ts = datetime.now().strftime(f"scenario_{name}_%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    borrowers.to_parquet(os.path.join(out_dir, "borrowers.parquet"))
    card_loans.to_parquet(os.path.join(out_dir, "loans_card.parquet"))
    auto_loans.to_parquet(os.path.join(out_dir, "loans_auto.parquet"))
    personal_loans.to_parquet(os.path.join(out_dir, "loans_personal.parquet"))
    mortgage_loans.to_parquet(os.path.join(out_dir, "loans_mortgage.parquet"))
    heloc_loans.to_parquet(os.path.join(out_dir, "loans_heloc.parquet"))

    panels = {
        "card": card_panel,
        "auto": auto_panel,
        "personal": personal_panel,
        "mortgage": mortgage_panel,
        "heloc": heloc_panel,
    }
    for k, v in panels.items():
        v.to_parquet(os.path.join(out_dir, f"loan_monthly_{k}.parquet"))

    # Portfolio aggregates per scenario
    portfolio_list = []
    for k, v in panels.items():
        m = compute_monthly_ecl(v)
        p = compute_portfolio_aggregates(m)
        p["product"] = k
        portfolio_list.append(p)

    portfolio_all = pd.concat(portfolio_list, ignore_index=True)
    portfolio_all.to_parquet(os.path.join(out_dir, "portfolio_aggregates_by_product.parquet"))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-product scenarios and CECL portfolio aggregates")
    parser.add_argument("--config", type=str, required=True, help="YAML file with scenarios config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    scenarios = cfg.get("scenarios", [])
    out_root = cfg.get("output_dir", "data/scenario_runs")

    for s in scenarios:
        name = s.get("name", "scenario")
        n_borrowers = int(s.get("n_borrowers", 8000))
        months = int(s.get("months", 6))
        macro_adj = s.get("macro_adjustments", {})
        print(f"Running scenario: {name} ...")
        out_dir = run_scenario(name, n_borrowers, months, macro_adj, out_root)
        print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
