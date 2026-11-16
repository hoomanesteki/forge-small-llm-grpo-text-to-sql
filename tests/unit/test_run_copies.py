"""Which copy of a run is newer, this machine's or the store's (``forge.ops.run_copies``),
with each store, the Hub's client stood in for by ``FakeHub``."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forge.core.types import RunManifest
from forge.eval.run_files import RunConflict
from forge.ops.hub import HubRunStore, LocalRunStore, RunNotFound, RunStore, manifest_bytes
from forge.ops.run_copies import StoreNewer, compare, fetch_newer, history, put_run
from forge.ops.sessions import Session
from tests.fakes import FakeHub

RUN = "2026-11-15-qwen3-4b-eval-abc1234"
START = datetime(2026, 11, 15, 8, 0, tzinfo=UTC)


def _at(hours: int) -> datetime:
    return START + timedelta(hours=hours)


def _write(folder: Path, starts: list[int], scores: int) -> Path:
    """A copy of the run whose sessions started ``starts`` hours in, holding ``scores``."""
    (folder / "eval").mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_id=RUN,
        kind="eval",
        git_sha="b" * 40,
        config={"split": "val"},
        config_hash="2" * 64,
        data_manifest_hash="3" * 64,
        model_ref="gold",
        platform="local-cpu",
        seed=0,
        started_at=_at(starts[0]),
        ended_at=_at(starts[-1] + 1),
        status="failed",
    )
    (folder / "manifest.json").write_bytes(manifest_bytes(manifest))
    sessions = [Session(_at(h), _at(h + 1), "failed").to_json() for h in starts]
    (folder / "sessions.json").write_text(json.dumps(sessions))
    (folder / "eval" / "val-scores.jsonl").write_text("{}\n" * scores)
    return folder


@pytest.fixture(params=["local", "hub"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> RunStore:
    if request.param == "local":
        return LocalRunStore(tmp_path / "store")
    return HubRunStore("owner/forge-runs", api=FakeHub(), sleep=lambda seconds: None)


def _stored(store: RunStore, tmp_path: Path, starts: list[int], scores: int) -> None:
    store.put_folder(RUN, _write(tmp_path / "elsewhere" / RUN, starts, scores))


def _scores(folder: Path) -> int:
    return len((folder / "eval" / "val-scores.jsonl").read_text().splitlines())


@pytest.mark.phase(3)
def test_a_copys_sessions_are_its_history(tmp_path: Path) -> None:
    one, two, three = _at(0), _at(2), _at(4)
    assert compare((one, two), (one, two)) == "same"
    assert compare((one, two, three), (one, two)) == "newer"
    assert compare((one,), (one, two)) == "older"
    assert compare((one, three), (one, two)) == "diverged"
    assert compare((two,), (one,)) == "diverged"

    folder = _write(tmp_path / RUN, [0, 2], 4)
    assert history(folder) == (one, two)
    # A folder from before runs kept sessions has the one its manifest spans.
    (folder / "sessions.json").unlink()
    assert history(folder) == (one,)
    assert history(tmp_path / "nothing") is None


@pytest.mark.phase(3)
def test_put_run_never_uploads_over_a_store_that_holds_more(
    store: RunStore, tmp_path: Path
) -> None:
    local = _write(tmp_path / "runs" / RUN, [0], 2)
    with pytest.raises(RunNotFound, match=r"has no manifest\.json"):
        put_run(store, tmp_path / "runs" / "empty", replace=True)
    # Nothing stored yet, then the same copy, then a newer one: each goes up.
    put_run(store, local, replace=True)
    put_run(store, local, replace=True)
    _write(local, [0, 2], 4)
    put_run(store, local, replace=True)
    pulled = tmp_path / "pulled" / RUN
    store.pull(RUN, pulled)
    assert _scores(pulled) == 4

    # Another machine carried the run on from there, so this older copy stays here.
    _stored(store, tmp_path, [0, 2, 4], 6)
    _write(local, [0, 2], 4)
    with pytest.raises(StoreNewer, match="holds 1 session more than") as raised:
        put_run(store, local, replace=True)
    assert "nothing was uploaded" in str(raised.value)
    # And a copy that carried on from session 2 itself went another way.
    _write(local, [0, 2, 5], 8)
    with pytest.raises(StoreNewer, match="went separate ways at session 3"):
        put_run(store, local, replace=True)
    store.pull(RUN, pulled)
    assert _scores(pulled) == 6


@pytest.mark.phase(3)
def test_fetch_newer_takes_the_stores_copy_only_when_it_is_newer(
    store: RunStore, tmp_path: Path
) -> None:
    local = tmp_path / "runs" / RUN
    assert fetch_newer(store, local) is False  # nothing stored: the run starts here
    _stored(store, tmp_path, [0, 2], 4)
    assert fetch_newer(store, local) is True and _scores(local) == 4
    assert fetch_newer(store, local) is False  # the same copy

    _write(local, [0, 2, 4], 6)  # an attempt here whose upload failed
    assert fetch_newer(store, local) is False and _scores(local) == 6

    _stored(store, tmp_path, [0, 2, 4, 6], 8)
    _write(local, [0, 2], 4)
    assert fetch_newer(store, local) is True and _scores(local) == 8

    _write(local, [0, 2, 4, 5], 7)
    with pytest.raises(RunConflict, match="went separate ways at session 4") as raised:
        fetch_newer(store, local)
    assert f"forge runs pull {RUN}" in str(raised.value)
    assert _scores(local) == 7
