# Project Plan: Synthetic CECL & Credit Data Platform

## Goals
- Generate realistic synthetic borrower/loan panels at scale (200M borrowers)
- Compute CECL lifetime ECL with macro scenarios
- Calibrate to public benchmarks (LendingClub, BofA disclosures)

## Phases

### Phase 1: Prototype (Weeks 1–2)
- Data model and schemas finalized
- Macro fetcher with FRED + synthetic fallback
- Card-only generator producing monthly panel
- Basic CECL calc stub (PD×LGD×EAD) per month
- CI to run small sample generator

### Phase 2: Realism & Coverage (Weeks 3–6)
- Add auto, personal, mortgage/HELOC modules
- Transition model (roll-rate) with macro links
- Recoveries and LGD by product with lags
- Prepayment/cure intensities
- Feature engineering: lags/leads/rolling/interactions

### Phase 3: Calibration (Weeks 5–8)
- Ingest LendingClub sample for distribution fits
- Portfolio-level constraints using BofA disclosures
- Forecast curve matching (hazard scaling / parametric)

### Phase 4: Scale-out (Weeks 7–10)
- PySpark pipeline for 100M+ entities
- Parquet partitioning and checkpoints
- Reproducibility: partition seeds, schema registry

## Deliverables
- `scripts/generate_sample.py` small-data artifact + schema docs
- `src/credit_data/*` modular generators per product
- `features/` store with versioned schema
- `notebooks/` for EDA and calibration checks
- CECL reporting: lifetime ECL per loan and portfolio aggregates

## Milestones
- M1: Prototype sample end-to-end (card) [DONE]
- M2: Add products + transitions + recoveries
- M3: Calibration pass vs public benchmarks
- M4: Spark scale-out with reproducibility

## Open Questions
- Scenario probabilities and governance
- Discounting policy for ECL reporting
- Data privacy constraints for seeded realism

## Next Actions
- Add CECL calc module and reporting endpoints
- Implement transitions and recoveries for card
- Add auto and personal loan generators
- Set up calibration notebook with LC sample
