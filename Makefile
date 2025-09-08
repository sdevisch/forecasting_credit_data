PY=python

.PHONY: e2e scenario-ci clean

build:
	docker build -t credit-data:latest .

run:
	docker run --rm -v $(PWD)/data:/app/data credit-data:latest

e2e:
	$(PY) scripts/run_end_to_end.py --n_borrowers 2000 --months 3 --validate

scenario-ci:
	$(PY) scripts/run_scenarios.py --config examples/example_scenarios.yaml
	$(PY) scripts/compare_scenarios.py

clean:
	rm -rf data/processed/* data/scenario_runs/*
