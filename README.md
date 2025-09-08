# Forecasting Credit Data

Synthetic credit risk data generation and CECL allowance prototyping.

## Quickstart

1. Create and activate a virtual environment
```bash
python3 -m venv ~/venvs/credit_data
source ~/venvs/credit_data/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

3. Generate a small sample dataset
```bash
python scripts/generate_sample.py --n_borrowers 10000 --months 12
```

Artifacts are written to `data/processed/sample_YYYYMMDD_HHMMSS/`.

## Structure
- `src/credit_data/`: package code
- `scripts/`: entry points and utilities
- `data/`: raw and processed outputs (gitignored)
- `features/`: derived features (gitignored)
- `notebooks/`: exploration

## Configuration and environment
- Optional `.env` in project root to supply environment variables:
  - `FRED_API_KEY`: used by macro fetcher when available. Example:
    ```bash
    FRED_API_KEY=your_fred_api_key_here
    ```
- Programmatic config: use `credit_data.config.load_config("path/to/config.yaml")` to merge YAML with env vars.

## Useful scripts
- Multi-product sample (with product filter and partitioned outputs):
  ```bash
  # all products
  python scripts/generate_multiproduct_sample.py --n_borrowers 20000 --months 12
  # subset of products
  python scripts/generate_multiproduct_sample.py --n_borrowers 8000 --months 6 --products card,personal
  # partitioned parquet outputs (product, asof_month)
  python scripts/generate_multiproduct_sample.py --n_borrowers 8000 --months 6 --partitioned
  ```
- CECL (auto-detects partitioned or per-product panels):
  ```bash
  python scripts/run_cecl_multiproduct.py --input data/processed/sample_multi_YYYYMMDD_HHMMSS
  ```
- Feature engineering:
  ```bash
  python scripts/build_features.py --input data/processed/sample_multi_YYYYMMDD_HHMMSS
  ```
- Reports:
  ```bash
  python scripts/generate_reports.py --input data/processed/sample_multi_YYYYMMDD_HHMMSS
  ```
- Scenario runner (YAML):
  ```bash
  python scripts/run_scenarios.py --config examples/example_scenarios.yaml
  python scripts/compare_scenarios.py
  ```
- End-to-end (imperative):
  ```bash
  python scripts/run_end_to_end.py --n_borrowers 2000 --months 3 --validate
  ```
- End-to-end (YAML-driven):
  ```bash
  # products + partitioned supported in YAML
  python scripts/run_end_to_end_config.py --config examples/pipeline.yaml
  ```

## Docker & Makefile
- Build and run inside a container (mounts `data/`):
  ```bash
  make build
  make run
  ```
- Local convenience targets:
  ```bash
  make e2e         # end-to-end with validation
  make scenario-ci # scenarios + comparison summary
  ```

## Notes
- Macro data fetch will use FRED when a `FRED_API_KEY` env var is set; otherwise it falls back to synthetic macro series for development.
