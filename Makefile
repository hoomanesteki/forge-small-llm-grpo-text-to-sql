.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
PHASE ?= 8
PYTEST := $(UV) run pytest

# pytest exits 5 when a folder has no tests yet; that is fine early in the plan.
define allow_empty
	$(1) || { status=$$?; [ $$status -eq 5 ] && echo "(no tests collected yet)"; [ $$status -eq 5 ]; }
endef

.PHONY: help setup lint fmt type test-unit test-int smoke check gate

help: ## Show the available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install the dev tools and the CPU extra
	$(UV) sync --extra cpu
	@# macOS can flag the editable-install .pth as hidden, and Python 3.12 then skips it.
	@if [ "$$(uname)" = "Darwin" ]; then chflags nohidden .venv/lib/python*/site-packages/*.pth; fi

lint: ## Ruff lint and format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: ## Apply ruff formatting and safe fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

type: ## Mypy (strict in core, sql, eval, train.rewards)
	$(UV) run mypy

test-unit: ## Unit tests
	$(call allow_empty,$(PYTEST) tests/unit)

test-int: ## Integration tests (fixture Spider subset)
	$(call allow_empty,$(PYTEST) tests/integration)

smoke: ## Full pipeline on CPU with a tiny model
	@echo "make smoke is added in FG-207 (CPU smoke pipeline in CI)"

check: lint type test-unit test-int ## Everything a PR must pass locally

gate: lint type ## What a phase gate runs: make gate PHASE=N
	$(call allow_empty,$(PYTEST) --phase $(PHASE) tests/unit tests/integration)
	$(MAKE) smoke
