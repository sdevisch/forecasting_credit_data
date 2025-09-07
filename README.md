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
- Multi-product sample:
  ```bash
  python scripts/generate_multiproduct_sample.py --n_borrowers 20000 --months 12
  ```
- CECL (multi-product):
  ```bash
  python scripts/run_cecl_multiproduct.py --input data/processed/sample_multi_YYYYMMDD_HHMMSS
  ```
- Scenario runner (YAML):
  ```bash
  python scripts/run_scenarios.py --config examples/example_scenarios.yaml
  ```
- Calibration summaries:
  ```bash
  python scripts/run_calibration.py --input data/processed/sample_multi_YYYYMMDD_HHMMSS
  ```
- Curve calibration (hazard scaling):
  ```bash
  python scripts/apply_curve_calibration.py --input data/processed/sample_multi_YYYYMMDD_HHMMSS/cecl_multi --targets examples/target_curves.yaml --months 12
  ```
- Feature engineering:
  ```bash
  python scripts/build_features.py --input data/processed/sample_multi_YYYYMMDD_HHMMSS
  ```

## Notes
- Macro data fetch will use FRED when a `FRED_API_KEY` env var is set; otherwise it falls back to synthetic macro series for development.
