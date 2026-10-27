# `make` sets everything up (once) and starts the live demo. Every step is skipped when it is already done.
#
#   make            venv + install + models + student + tests, then the webcam demo
#   make test       only the tests
#   make clean      remove the virtual environment (models in ~/.cache/bifold-morph stay)

PYTHON  ?= python3
VENV    := .venv
BIN     := $(VENV)/bin
MODELS  := $(or $(BIFOLD_MORPH_MODELS),$(HOME)/.cache/bifold-morph)
STUDENT := src/bifold_morph/weights/student.pt
WORK    := $(MODELS)/train

.PHONY: all run test fetch clean

all: test run

# a virtual environment with the package installed (again only when pyproject.toml changes)
$(BIN)/bifold-morph: pyproject.toml
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev]"
	touch $@

# the models the app needs (YuNet, FairFace, InsightFace genderage), checked by size
fetch: $(BIN)/bifold-morph
	$(BIN)/bifold-morph fetch

# the student network: comes with the repository; trained here only if it is missing (hours on a CPU)
$(STUDENT): | $(BIN)/bifold-morph
	$(BIN)/bifold-morph train --work $(WORK)
	mkdir -p $(dir $@) && cp $(WORK)/student.pt $@

test: $(BIN)/bifold-morph
	$(BIN)/pytest -q

run: fetch $(STUDENT)
	$(BIN)/bifold-morph

clean:
	rm -rf $(VENV)
