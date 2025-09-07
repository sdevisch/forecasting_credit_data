#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml  # type: ignore


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline from YAML config")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    pipe = cfg.get("pipeline", {})
    n_borrowers = int(pipe.get("n_borrowers", 10000))
    months = int(pipe.get("months", 6))
    steps = pipe.get("steps", [])

    dataset_dir = None

    for step in steps:
        if step == "generate_multiproduct_sample":
            run([sys.executable, "scripts/generate_multiproduct_sample.py", "--n_borrowers", str(n_borrowers), "--months", str(months)])
            latest = sorted([p for p in os.listdir("data/processed") if p.startswith("sample_multi_")], reverse=True)[0]
            dataset_dir = os.path.join("data/processed", latest)
        elif step == "run_cecl_multiproduct":
            assert dataset_dir is not None, "Run generate step first"
            run([sys.executable, "scripts/run_cecl_multiproduct.py", "--input", dataset_dir])
        elif step == "build_features":
            assert dataset_dir is not None, "Run generate step first"
            run([sys.executable, "scripts/build_features.py", "--input", dataset_dir])
        elif step == "generate_reports":
            assert dataset_dir is not None, "Run generate step first"
            run([sys.executable, "scripts/generate_reports.py", "--input", dataset_dir])
        else:
            raise ValueError(f"Unknown step: {step}")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
