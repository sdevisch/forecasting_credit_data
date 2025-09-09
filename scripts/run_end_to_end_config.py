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
    parser = argparse.ArgumentParser(
        description="Run end-to-end pipeline from YAML config"
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    pipe = cfg.get("pipeline", {})
    n_borrowers = int(pipe.get("n_borrowers", 10000))
    months = int(pipe.get("months", 6))
    products = pipe.get("products")  # optional comma-separated or list
    partitioned = bool(pipe.get("partitioned", False))
    steps = pipe.get("steps", [])

    # Calibration config
    calib = cfg.get("calibration", {})
    targets_path = calib.get("targets", "examples/target_curves.yaml")
    calib_months = int(calib.get("months", months))

    dataset_dir = None

    for step in steps:
        if step == "generate_multiproduct_sample":
            gen_cmd = [
                sys.executable,
                "scripts/generate_multiproduct_sample.py",
                "--n_borrowers",
                str(n_borrowers),
                "--months",
                str(months),
            ]
            if products:
                if isinstance(products, list):
                    products_arg = ",".join(products)
                else:
                    products_arg = str(products)
                gen_cmd += ["--products", products_arg]
            if partitioned:
                gen_cmd += ["--partitioned"]
            run(gen_cmd)
            latest = sorted(
                [
                    p
                    for p in os.listdir("data/processed")
                    if p.startswith("sample_multi_")
                ],
                reverse=True,
            )[0]
            dataset_dir = os.path.join("data/processed", latest)
        elif step == "run_cecl_multiproduct":
            assert dataset_dir is not None, "Run generate step first"
            run(
                [
                    sys.executable,
                    "scripts/run_cecl_multiproduct.py",
                    "--input",
                    dataset_dir,
                ]
            )
        elif step == "apply_curve_calibration":
            assert dataset_dir is not None, "Run CECL step first"
            cecl_dir = os.path.join(dataset_dir, "cecl_multi")
            run(
                [
                    sys.executable,
                    "scripts/apply_curve_calibration.py",
                    "--input",
                    cecl_dir,
                    "--targets",
                    str(targets_path),
                    "--months",
                    str(calib_months),
                ]
            )
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
