#!/usr/bin/env python3
from __future__ import annotations

import argparse

from credit_data.lendingclub_ingest import ingest_lendingclub_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest LendingClub CSV to standardized Parquet outputs"
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to LendingClub CSV file"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output directory for Parquet files"
    )
    args = parser.parse_args()

    b_path, l_path = ingest_lendingclub_csv(args.csv, args.out)
    print(f"Wrote borrowers to {b_path}")
    print(f"Wrote loans to {l_path}")


if __name__ == "__main__":
    main()
