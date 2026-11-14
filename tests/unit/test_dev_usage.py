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
