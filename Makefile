PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install-base install-data install-model install-ui install-dev test lint format

install-base:
	$(PIP) install -r requirements/base.txt

install-data:
	$(PIP) install -r requirements/data.txt

install-model:
	$(PIP) install -r requirements/model.txt

install-ui:
	$(PIP) install -r requirements/ui.txt

install-dev:
	$(PIP) install -r requirements/dev.txt

test:
	pytest

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check --fix src tests
