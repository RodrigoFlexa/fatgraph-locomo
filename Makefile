PY ?= python3

.DEFAULT_GOAL := help
.PHONY: help install setup test lint smoke ingest qa all report notebooks clean clean-dry

help:            ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:         ## editable install with every optional extra
	$(PY) -m pip install -e ".[all]"

setup:           ## fetch the LoCoMo dataset and create .env from the template
	fgl setup

test:            ## full offline test suite (no network, no spend)
	$(PY) -m pytest tests/

lint:            ## ruff check + format check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

smoke:           ## every condition end to end, offline, 1 conversation
	fgl run-all --dry-run --limit-conversations 1 --limit-questions 10 --continue-on-error

ingest:          ## build the G1 memory for all 10 conversations
	fgl ingest G1

qa:              ## G1 question answering over the full question set
	fgl qa G1

all:             ## every condition, all 10 conversations, final report
	fgl run-all

report:          ## rebuild the tables from results/
	fgl report

notebooks:       ## launch Jupyter on the analysis notebooks
	$(PY) -m jupyter lab notebooks/

clean-dry:       ## remove dry-run artefacts only
	rm -rf results-dry artifacts/graphs-dry artifacts/logs-dry artifacts/facts-dry \
	       .cache/embeddings-dry

clean: clean-dry ## remove every generated artefact (keeps results/ and data/)
	rm -rf artifacts .cache .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
