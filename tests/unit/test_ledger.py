from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import get_args

import pytest

from forge.ops.ledger import (
    GPU_PLATFORMS,
    HEADER,
    PLATFORMS,
    QUOTA_PLATFORMS,
    LedgerRow,
    append,
    has_run,
    read_rows,
    summary,
)
from forge.ops.manifest import Platform

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


@pytest.mark.phase(0)
@pytest.mark.parametrize(
    "field, value",
    [
        ("platform", "Kaggle"),
        ("platform", "kagle"),
        ("hours", float("nan")),
        ("hours", float("inf")),
        ("cost_usd", -0.5),
        ("run_id", "run|x"),
        ("run_id", "  "),
        ("gpu", "T4\nA100"),
    ],
)
def test_bad_rows_rejected(ledger: Path, field: str, value: object) -> None:
    good = {
        "date": TODAY,
        "platform": "kaggle",
        "gpu": "T4",
        "hours": 1.0,
        "run_id": "r",
        "purpose": "p",
    }
    with pytest.raises(ValueError):
        append(LedgerRow(**{**good, field: value}), ledger)  # type: ignore[arg-type]
    assert read_rows(ledger) == []


@pytest.mark.phase(0)
def test_purpose_line_breaks_are_repaired(ledger: Path) -> None:
    append(LedgerRow(TODAY, "kaggle", "T4", 1.0, "r", "line one\r\nline two | done"), ledger)
    assert read_rows(ledger)[0].purpose == "line one / line two / done"


@pytest.mark.phase(0)
def test_concurrent_appends_keep_every_row(ledger: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def write(i: int) -> None:
        append(LedgerRow(TODAY, "kaggle", "T4", 0.25, f"run-{i}", "parallel"), ledger)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(write, range(200)))
    assert len(read_rows(ledger)) == 200


@pytest.mark.phase(0)
def test_no_repo_root_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from forge.ops.paths import RepoRootNotFound, ledger_path

    monkeypatch.delenv("FORGE_ROOT", raising=False)
    monkeypatch.delenv("FORGE_LEDGER", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RepoRootNotFound):
        ledger_path()
    monkeypatch.setenv("FORGE_LEDGER", str(tmp_path / "ledger.md"))
    assert ledger_path() == tmp_path / "ledger.md"


@pytest.mark.phase(2)
def test_local_mps_counts_in_neither_quota_nor_gpu_total(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pilot on the laptop's GPU is written down, but spends no free GPU budget (ADR-0016).
    monkeypatch.setenv("FORGE_WEEKLY_GPU_HOURS", "30")
    append(LedgerRow(TODAY, "kaggle", "T4", 1.5, "run-a", "eval"), ledger)
    append(LedgerRow(TODAY, "local-mps", "Apple M4", 0.25, "run-b", "pilot on val"), ledger)

    assert [(r.platform, r.gpu) for r in read_rows(ledger)][1] == ("local-mps", "Apple M4")
    s = summary(path=ledger, today=TODAY)
    assert s.quota_hours_window == pytest.approx(1.5)
    assert s.gpu_hours_total == pytest.approx(1.5)
    assert s.sessions == 1
    assert s.cash_total == 0.0


@pytest.mark.phase(2)
def test_ledger_takes_every_manifest_platform() -> None:
    # A run's manifest platform must be one a ledger row can name.
    assert frozenset(get_args(Platform)) == PLATFORMS
    assert QUOTA_PLATFORMS <= GPU_PLATFORMS < PLATFORMS
    assert not {"local-cpu", "local-mps"} & GPU_PLATFORMS
