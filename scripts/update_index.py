#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from glob import glob

import pandas as pd


def main() -> None:
    base = "data/processed"
    runs = sorted([p for p in glob(os.path.join(base, "sample_multi_*")) if os.path.isdir(p)])
    rows = []
    for r in runs:
        meta_path = os.path.join(r, "metadata.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        rows.append({
            "run_dir": r,
            "timestamp": metadata.get("timestamp"),
            "git_commit": metadata.get("git_commit"),
            "n_borrowers": metadata.get("params", {}).get("n_borrowers"),
            "months": metadata.get("params", {}).get("months"),
            "products": ",".join(metadata.get("params", {}).get("products", [])) if metadata.get("params") else None,
            "partitioned": metadata.get("params", {}).get("partitioned"),
        })
    df = pd.DataFrame(rows)
    out_path = os.path.join(base, "index.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows")


if __name__ == "__main__":
    main()
