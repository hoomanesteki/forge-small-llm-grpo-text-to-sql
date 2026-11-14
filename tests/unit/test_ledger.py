from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from forge.ops.ledger import HEADER, LedgerRow, append, has_run, read_rows, summary

TODAY = date(2026, 11, 14)


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    path = tmp_path / "compute-ledger.md"
    path.write_text("# Compute ledger\n\n" + HEADER)
    return path


@pytest.mark.phase(0)
def test_append_and_summary(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_WEEKLY_GPU_HOURS", "30")
    rows = [
        LedgerRow(date(2026, 11, 13), "kaggle", "T4", 2.5, "run-a", "eval spike"),
        LedgerRow(date(2026, 11, 3), "kaggle", "T4", 4.0, "run-b", "older than a week"),
        LedgerRow(date(2026, 11, 12), "colab", "T4", 1.0, "run-c", "manual fallback"),
        LedgerRow(date(2026, 11, 10), "modal", "L4", 1.5, "run-d", "boost | seed 2", 1.2),
        LedgerRow(date(2026, 11, 9), "local-cpu", "", 3.0, "run-e", "smoke"),
    ]
    for row in rows:
        append(row, ledger)

    read = read_rows(ledger)
    assert [r.run_id for r in read] == ["run-a", "run-b", "run-c", "run-d", "run-e"]
    assert read[3].purpose == "boost / seed 2"  # a raw pipe would split the table cell
    assert read[0] == rows[0]
    s = summary(path=ledger, today=TODAY)
    assert s.quota_hours_window == pytest.approx(2.5)
    assert s.gpu_hours_total == pytest.approx(9.0)
    assert s.cash_total == pytest.approx(1.2)
    assert s.remaining_week == pytest.approx(27.5)
    assert s.sessions == 4
    assert "remaining 27.50" in s.render()
    assert has_run("run-c", ledger) and not has_run("run-z", ledger)


@pytest.mark.phase(0)
def test_append_creates_missing_ledger(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "compute-ledger.md"
    append(LedgerRow(TODAY, "kaggle", "T4", 0.5, "run-x", "first"), path)
    assert [r.run_id for r in read_rows(path)] == ["run-x"]
    assert summary(path=path, today=TODAY).quota_hours_window == pytest.approx(0.5)


@pytest.mark.phase(0)
def test_negative_hours_rejected(ledger: Path) -> None:
    with pytest.raises(ValueError):
        append(LedgerRow(TODAY, "kaggle", "T4", -1.0, "run-x", "bad"), ledger)
