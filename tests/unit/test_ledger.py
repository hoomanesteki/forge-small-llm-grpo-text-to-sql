from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, get_args

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import RunManifest
from forge.ops.hub import LocalRunStore, manifest_bytes
from forge.ops.ledger import (
    GPU_PLATFORMS,
    GPU_RUN_PLATFORMS,
    HEADER,
    PLATFORMS,
    QUOTA_PLATFORMS,
    LedgerRow,
    append,
    append_once,
    has_run,
    read_rows,
    record_run,
    session_row,
    summary,
)
from forge.ops.manifest import Platform
from forge.ops.sessions import Session
from tests.fakes import FakeHub

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


RUN = "2026-11-16-qwen3-4b-eval-abc1234"
# 23:30 in California is 07:30 the next day in UTC, the date a row takes.
STARTED = datetime(2026, 11, 15, 23, 30, tzinfo=timezone(timedelta(hours=-8)))


def _manifest(**changes: Any) -> RunManifest:
    fields: dict[str, Any] = {
        "run_id": RUN,
        "kind": "eval",
        "git_sha": "b" * 40,
        "config": {"split": "val", "prompt_version": "p1", "limit": 50, "adapter": None},
        "config_hash": "2" * 64,
        "data_manifest_hash": "3" * 64,
        "model_ref": "Qwen/Qwen3-4B",
        "platform": "kaggle",
        "gpu": "Tesla T4",
        "seed": 0,
        "started_at": STARTED,
        "ended_at": STARTED + timedelta(hours=1, minutes=30),
        "status": "stopped_time_guard",
        **changes,
    }
    return RunManifest(**fields)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    """A fake Hub run store, runs/ and the ledger under tmp_path, and no token."""
    fake = FakeHub()
    monkeypatch.setattr("huggingface_hub.HfApi", fake.client)
    monkeypatch.setenv("FORGE_RUN_STORE", "owner/forge-runs")
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_LEDGER", str(tmp_path / "compute-ledger.md"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return fake


def _pull(hub: FakeHub, manifest: RunManifest, sessions: list[Session] | None = None) -> str:
    hub.files[f"runs/{manifest.run_id}/manifest.json"] = manifest_bytes(manifest)
    if sessions is not None:
        record = json.dumps([session.to_json() for session in sessions]).encode()
        hub.files[f"runs/{manifest.run_id}/sessions.json"] = record
    result = CliRunner().invoke(cli.app, ["runs", "pull", manifest.run_id])
    assert result.exit_code == 0, result.output
    return str(result.stdout)


@pytest.mark.phase(3)
def test_pull_appends_once(hub: FakeHub, tmp_path: Path) -> None:
    ledger = tmp_path / "compute-ledger.md"

    first = _pull(hub, _manifest())
    second = _pull(hub, _manifest())

    assert first.endswith(f"ledger: added 1.50 h on kaggle for run {RUN}\n")
    assert second.endswith(f"ledger: run {RUN} has its row already\n")
    purpose = "eval Qwen/Qwen3-4B on val, prompt p1, limit 50, stopped_time_guard"
    expected = LedgerRow(date(2026, 11, 16), "kaggle", "Tesla T4", 1.5, RUN, purpose)
    assert read_rows(ledger) == [expected]
    assert ledger.read_text().startswith("# Compute ledger\n\n" + HEADER)


@pytest.mark.phase(3)
def test_pull_adds_no_row_without_a_gpu_or_an_end(hub: FakeHub, tmp_path: Path) -> None:
    running = _manifest(status="running", ended_at=None)
    assert _pull(hub, running).endswith(
        f"ledger: no row yet, run {RUN} is running; pull it once it ends\n"
    )
    laptop = _manifest(run_id="2026-11-16-m-eval-abc1234", platform="local-cpu", gpu=None)
    assert _pull(hub, laptop).endswith("ledger: no row, a local-cpu run holds no GPU\n")
    assert read_rows(tmp_path / "compute-ledger.md") == []

    # A pilot on the laptop's GPU is written down (ADR-0016), with its chip and adapter.
    pilot = _manifest(
        platform="local-mps",
        gpu="Apple M4",
        adapter_ref="runs/sft/checkpoints/best",
        config={"split": "val"},
        status="finished",
        cost_usd=0.0,
    )
    rows, notes = record_run(pilot, path=tmp_path / "compute-ledger.md")
    assert notes == [f"ledger: added 1.50 h on local-mps for run {RUN}"]
    (row,) = rows
    assert (row.platform, row.gpu, row.hours) == ("local-mps", "Apple M4", 1.5)
    assert row.purpose == "eval Qwen/Qwen3-4B with runs/sft/checkpoints/best on val, finished"
    # Every platform but the laptop's CPU holds a GPU.
    assert {p for p in PLATFORMS if p not in GPU_RUN_PLATFORMS} == {"local-cpu"}


@pytest.mark.phase(4)
def test_ledger_record_adds_a_local_run_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "compute-ledger.md"
    runs = LocalRunStore(tmp_path / "runs")
    monkeypatch.setenv("FORGE_RUNS_DIR", str(runs.root))
    monkeypatch.setenv("FORGE_LEDGER", str(ledger))
    pilot = _manifest(
        run_id="2026-11-16-qwen2-5-coder-0-5b-instruct-sft-abc1234",
        kind="sft",
        config={"data": {"split": "train"}},
        model_ref="Qwen/Qwen2.5-Coder-0.5B-Instruct@" + "e" * 40,
        platform="local-mps",
        gpu="Apple M4",
        status="finished",
    )
    laptop = _manifest(run_id="2026-11-16-tiny-sft-abc1234", platform="local-cpu", gpu=None)
    for manifest in (pilot, laptop):
        runs.put_manifest(manifest)

    def record(run_id: str) -> Any:
        return CliRunner().invoke(cli.app, ["ledger", "record", run_id])

    first, second = record(pilot.run_id), record(pilot.run_id)
    assert (first.exit_code, second.exit_code) == (0, 0), first.output
    # The hours are the manifest's, from its start to its end.
    assert first.stdout == f"ledger: added 1.50 h on local-mps for run {pilot.run_id}\n"
    assert second.stdout == f"ledger: run {pilot.run_id} has its row already\n"
    purpose = f"sft {pilot.model_ref}, finished"
    assert read_rows(ledger) == [
        LedgerRow(date(2026, 11, 16), "local-mps", "Apple M4", 1.5, pilot.run_id, purpose)
    ]
    assert record(laptop.run_id).stdout == "ledger: no row, a local-cpu run holds no GPU\n"
    missing = record("2026-11-16-nothing-abc1234")
    assert missing.exit_code == 1 and missing.stderr.startswith("error: ")
    assert len(read_rows(ledger)) == 1

    # A pilot resumed on this machine gets a row for each session, with its own hours,
    # never the time between them.
    resumed = pilot.model_copy(
        update={"run_id": "2026-11-16-qwen2-5-coder-0-5b-instruct-sft-def5678"}
    )
    runs.put_manifest(resumed)
    first_end = resumed.started_at + timedelta(hours=1)
    second_start = resumed.started_at + timedelta(days=1)
    (runs.root / resumed.run_id / "sessions.json").write_text(
        json.dumps(
            [
                Session(resumed.started_at, first_end, "stopped_time_guard").to_json(),
                Session(second_start, second_start + timedelta(minutes=30), "finished").to_json(),
            ]
        ),
        encoding="utf-8",
    )
    both = record(resumed.run_id)
    assert both.exit_code == 0, both.output
    assert both.stdout == (
        f"ledger: added 1.00 h on local-mps for session 1 of run {resumed.run_id}\n"
        f"ledger: added 0.50 h on local-mps for session 2 of run {resumed.run_id}\n"
    )
    assert [row.hours for row in read_rows(ledger) if row.run_id == resumed.run_id] == [1.0, 0.5]


@pytest.mark.phase(4)
def test_ledger_record_says_to_record_a_running_local_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "compute-ledger.md"
    runs = LocalRunStore(tmp_path / "runs")
    monkeypatch.setenv("FORGE_RUNS_DIR", str(runs.root))
    monkeypatch.setenv("FORGE_LEDGER", str(ledger))
    # A pilot still training on this machine: there is nothing to pull, so the note says to
    # run the same command again, for the run and for a session of a resumed one.
    running = _manifest(
        run_id="2026-11-16-qwen2-5-coder-0-5b-instruct-sft-abc1234",
        platform="local-mps",
        gpu="Apple M4",
        status="running",
        ended_at=None,
    )
    resumed = running.model_copy(update={"run_id": "2026-11-16-tiny-sft-def5678"})
    for manifest in (running, resumed):
        runs.put_manifest(manifest)
    (runs.root / resumed.run_id / "sessions.json").write_text(
        json.dumps([FIRST.to_json(), Session(LATER).to_json()]), encoding="utf-8"
    )

    def record(run_id: str) -> str:
        result = CliRunner().invoke(cli.app, ["ledger", "record", run_id])
        assert result.exit_code == 0, result.output
        return str(result.stdout)

    again = "run forge ledger record {} again once it ends\n"
    assert record(running.run_id) == (
        f"ledger: no row yet, run {running.run_id} is running; " + again.format(running.run_id)
    )
    assert record(resumed.run_id).endswith(
        f"ledger: session 2 of run {resumed.run_id} is running; " + again.format(resumed.run_id)
    )
    assert [row.run_id for row in read_rows(ledger)] == [resumed.run_id]
    # A pulled run's note still says to pull it.
    assert record_run(running, path=ledger)[1] == [
        f"ledger: no row yet, run {running.run_id} is running; pull it once it ends"
    ]


@pytest.mark.phase(3)
def test_append_once_holds_under_concurrent_pulls(ledger: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    row = LedgerRow(TODAY, "kaggle", "T4", 0.5, "run-x", "pulled twice at once")
    with ThreadPoolExecutor(max_workers=16) as pool:
        added = list(pool.map(lambda _: append_once(row, ledger), range(64)))
    assert added.count(True) == 1
    assert read_rows(ledger) == [row]
    assert not append_once(LedgerRow(TODAY, "colab", "T4", 1.0, "run-x", "other"), ledger)
    assert append_once(LedgerRow(TODAY, "colab", "T4", 1.0, "run-y", "another run"), ledger)


# A run stopped by the time guard after 11 h 15 min, resumed two days later for an hour.
FIRST = Session(STARTED, STARTED + timedelta(hours=11, minutes=15), "stopped_time_guard")
LATER = STARTED + timedelta(days=2)
SECOND = Session(LATER, LATER + timedelta(hours=1), "finished")
RESUMED = _manifest(status="finished", ended_at=SECOND.ended_at, gpu_hours=12.25)
PURPOSE = "eval Qwen/Qwen3-4B on val, prompt p1, limit 50"


@pytest.mark.phase(3)
def test_a_resumed_run_gets_a_row_for_each_session(hub: FakeHub, tmp_path: Path) -> None:
    ledger = tmp_path / "compute-ledger.md"
    stopped = _manifest(ended_at=FIRST.ended_at)
    assert _pull(hub, stopped, [FIRST]).endswith(f"added 11.25 h on kaggle for run {RUN}\n")

    # The second session gets a row of its own with its own hour, and pulling again adds none.
    after = _pull(hub, RESUMED, [FIRST, SECOND])
    assert after.endswith(
        f"ledger: session 1 of run {RUN} has its row already\n"
        f"ledger: added 1.00 h on kaggle for session 2 of run {RUN}\n"
    )
    _pull(hub, RESUMED, [FIRST, SECOND])
    rows = read_rows(ledger)
    assert [(row.date, row.hours, row.session) for row in rows] == [
        (date(2026, 11, 16), 11.25, 1),
        (date(2026, 11, 18), 1.0, 2),
    ]
    assert [row.purpose for row in rows] == [
        f"{PURPOSE}, stopped_time_guard",
        f"{PURPOSE}, finished (session 2)",
    ]

    # Pulled only once it has finished, it gets the same two rows, never the 49 h from its
    # first start to its last end.
    ledger.unlink()
    assert _pull(hub, RESUMED, [FIRST, SECOND]).endswith(
        f"ledger: added 11.25 h on kaggle for session 1 of run {RUN}\n"
        f"ledger: added 1.00 h on kaggle for session 2 of run {RUN}\n"
    )
    assert read_rows(ledger) == rows
    assert summary(path=ledger, today=date(2026, 11, 18)).quota_hours_window == 12.25


@pytest.mark.phase(3)
def test_session_rows_name_what_they_cannot_log(ledger: Path) -> None:
    killed = Session(STARTED, None, "failed")  # its attempt died before it could end it
    running = Session(LATER)
    rows, notes = record_run(RESUMED, [FIRST, killed, running], ledger)
    assert [row.session for row in rows] == [1]
    assert notes == [
        f"ledger: added 11.25 h on kaggle for session 1 of run {RUN}",
        f"ledger: session 2 of run {RUN} never recorded its end; add its hours with forge "
        "ledger add, ending its purpose with (session 2)",
        f"ledger: session 3 of run {RUN} is running; pull it once it ends",
    ]
    # A row added by hand for the session that never ended counts as that session's.
    append(LedgerRow(TODAY, "kaggle", "T4", 0.5, RUN, "killed by the host (session 2)"), ledger)
    assert [row.session for row in read_rows(ledger)] == [1, 2]
    assert not append_once(session_row(RESUMED, 2, SECOND), ledger)

    # A run's cost goes once, on its latest row, less what earlier rows hold.
    paid = RESUMED.model_copy(update={"cost_usd": 3.0, "platform": "modal"})
    rows, _ = record_run(paid, [FIRST], ledger.with_name("paid.md"))
    rows += record_run(
        paid.model_copy(update={"cost_usd": 5.0}), [FIRST, SECOND], ledger.with_name("paid.md")
    )[0]
    assert [(row.session, row.cost_usd) for row in rows] == [(1, 3.0), (2, 2.0)]
