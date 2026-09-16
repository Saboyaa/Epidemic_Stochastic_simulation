PY ?= .venv/bin/python

.PHONY: all test clean venv

all:
	$(PY) run_all.py

test:
	$(PY) -m pytest -q

venv:
	uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt

clean:
	rm -rf data/raw/* data/processed/* results/figures/* results/tables/* RESULTS.md
