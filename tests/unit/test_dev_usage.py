from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from forge.ops.dev_usage import (
    MAX_DEV_USES,
    DevBudgetExceeded,
    count_dev_uses,
    log_dev_use,
    read_uses,
)


@pytest.mark.phase(0)
def test_log_dev_use_requires_reason(tmp_path: Path) -> None:
    path = tmp_path / "dev-usage.md"
    with pytest.raises(ValueError):
        log_dev_use("run-1", "  ", path=path)
    assert count_dev_uses(path) == 0


@pytest.mark.phase(0)
def test_dev_budget_is_six(tmp_path: Path) -> None:
    path = tmp_path / "dev-usage.md"
    for i in range(MAX_DEV_USES):
        assert log_dev_use(f"run-{i}", "baseline", path=path, today=date(2026, 11, 14)) == i + 1
    with pytest.raises(DevBudgetExceeded):
        log_dev_use("run-7", "one more look", path=path)
    assert read_uses(path)[0] == ("2026-11-14", "run-0", "baseline")


@pytest.mark.phase(0)
def test_budget_holds_under_concurrency(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "dev-usage.md"

    def attempt(i: int) -> bool:
        try:
            log_dev_use(f"run-{i}", "race", path=path)
        except DevBudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(attempt, range(16)))
    assert sum(outcomes) == MAX_DEV_USES
    assert count_dev_uses(path) == MAX_DEV_USES


@pytest.mark.phase(0)
def test_pipe_in_run_id_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        log_dev_use("run|1", "baseline", path=tmp_path / "dev-usage.md")
