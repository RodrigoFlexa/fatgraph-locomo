PY ?= python3

.DEFAULT_GOAL := help
.PHONY: help install setup test lint smoke ingest qa all report notebooks clean clean-dry

# Every target forwards to tasks.py, which is the actual (cross-platform)
# implementation — this file exists so `make <target>` keeps working on
# Linux/macOS. On Windows, skip make entirely: `python tasks.py <target>`.

help:            ## show this help
	@$(PY) tasks.py help

install:         ## editable install with every optional extra
	@$(PY) tasks.py install

setup:           ## fetch the LoCoMo dataset and create .env from the template
	@$(PY) tasks.py setup

test:            ## full offline test suite (no network, no spend)
	@$(PY) tasks.py test

lint:            ## ruff check + format check
	@$(PY) tasks.py lint

smoke:           ## every condition end to end, offline, 1 conversation
	@$(PY) tasks.py smoke

ingest:          ## build the G1 memory for all 10 conversations
	@$(PY) tasks.py ingest

qa:              ## G1 question answering over the full question set
	@$(PY) tasks.py qa

all:             ## every condition, all 10 conversations, final report
	@$(PY) tasks.py all

report:          ## rebuild the tables from results/
	@$(PY) tasks.py report

notebooks:       ## launch Jupyter on the analysis notebooks
	@$(PY) tasks.py notebooks

clean-dry:       ## remove dry-run artefacts only
	@$(PY) tasks.py clean-dry

clean:           ## remove every generated artefact (keeps results/ and data/)
	@$(PY) tasks.py clean
