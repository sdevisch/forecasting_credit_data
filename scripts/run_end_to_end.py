#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    res = subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline: generate→CECL→features→reports")
    parser.add_argument("--n_borrowers", type=int, default=10000)
    parser.add_argument("--months", type=int, default=6)
    args = parser.parse_args()

    # 1) Generate multiproduct sample
    run([sys.executable, "scripts/generate_multiproduct_sample.py", "--n_borrowers", str(args.n_borrowers), "--months", str(args.months)])

    # Locate latest sample
    latest = sorted([p for p in os.listdir("data/processed") if p.startswith("sample_multi_")], reverse=True)[0]
    dataset_dir = os.path.join("data/processed", latest)

    # 2) CECL multiproduct
    run([sys.executable, "scripts/run_cecl_multiproduct.py", "--input", dataset_dir])

    # 3) Feature build
    run([sys.executable, "scripts/build_features.py", "--input", dataset_dir])

    # 4) Reports
    run([sys.executable, "scripts/generate_reports.py", "--input", dataset_dir])

    print(f"End-to-end complete. Outputs under {dataset_dir}")


if __name__ == "__main__":
    main()
