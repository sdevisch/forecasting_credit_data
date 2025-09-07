#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from credit_data.validation import validate_dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset quality and schema")
    parser.add_argument("--input", type=str, required=True, help="Dataset folder path")
    args = parser.parse_args()

    report = validate_dataset_dir(args.input)
    out_dir = os.path.join(args.input, "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "validation_report.csv")
    report.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(report)} issues")


if __name__ == "__main__":
    main()
