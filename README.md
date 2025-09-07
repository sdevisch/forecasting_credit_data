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

## Notes
- Macro data fetch will use FRED when a `FRED_API_KEY` env var is set; otherwise it falls back to synthetic macro series for development.
