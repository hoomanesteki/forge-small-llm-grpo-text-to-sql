.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
PHASE ?= 8
PYTEST := $(UV) run pytest

# pytest exits 5 when a folder has no tests yet; that is fine early in the plan.
define allow_empty
	$(1) || { status=$$?; [ $$status -eq 5 ] && echo "(no tests collected yet)"; [ $$status -eq 5 ]; }
endef

.PHONY: help setup unhide lint fmt type test-unit test-int smoke check gate

help: ## Show the available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## Install the dev tools and the CPU extra
	$(UV) sync --extra cpu
	@$(MAKE) --no-print-directory unhide

# On macOS the venv can come out with the hidden file flag set, and Python 3.12 skips
# hidden .pth files, so `import forge` breaks after uv re-syncs. Only the few .pth files
# matter, and touching just those keeps file-system event traffic negligible.
unhide: ## Clear the macOS hidden flag on the venv's .pth files
	@if [ "$$(uname)" = "Darwin" ]; then chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true; fi

lint: unhide ## Ruff lint and format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt: unhide ## Apply ruff formatting and safe fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

type: unhide ## Mypy (strict in core, sql, eval, train.rewards)
	$(UV) run mypy

test-unit: unhide ## Unit tests
	$(call allow_empty,$(PYTEST) tests/unit)

test-int: unhide ## Integration tests (fixture Spider subset)
	$(call allow_empty,$(PYTEST) tests/integration)

smoke: unhide ## Full pipeline on CPU with a tiny model
	@echo "make smoke is added in FG-207 (CPU smoke pipeline in CI)"

check: lint type test-unit test-int ## Everything a PR must pass locally

gate: unhide lint type ## What a phase gate runs: make gate PHASE=N
	$(call allow_empty,$(PYTEST) --phase $(PHASE) tests/unit tests/integration)
	$(UV) run python scripts/check_story_tests.py --phase $(PHASE)
	$(MAKE) smoke
