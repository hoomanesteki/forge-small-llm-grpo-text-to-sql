"""An eval run's folder: the manifest's life, resume checks, the lock, kill-safe outputs."""

from __future__ import annotations

import fcntl
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from forge.core.types import Generation, RunManifest, Score
from forge.eval.run_files import (
    LOCK_FILE,
    MANIFEST_FILE,
    RunConflict,
    RunCorrupt,
    RunFiles,
    RunLocked,
    RunPaths,
    RunUnfinished,
    finish,
    hold,
    read_run,
    start,
)
from forge.ops.manifest import DirtyTreeError, config_hash, git_sha

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {(f"q{i}", 0) for i in range(4)}


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _manifest(**changes: Any) -> RunManifest:
    fields: dict[str, Any] = {
        "run_id": "2026-11-15-gold-eval-abc1234",
        "kind": "eval",
        "git_sha": "a" * 40,
        "config": {"split": "val", "limit": 5},
        "config_hash": "c" * 64,
        "data_manifest_hash": "d" * 64,
        "model_ref": "gold",
        "platform": "local-cpu",
        "seed": 0,
        "started_at": datetime(2026, 11, 15, 9, 0, tzinfo=UTC),
    }
    return RunManifest(**{**fields, **changes})


def _generation(example_id: str, sample_idx: int = 0) -> Generation:
    return Generation(
        example_id=example_id,
        model_ref="gold",
        sample_idx=sample_idx,
        text="```sql\nSELECT 1\n```",
        sql="SELECT 1",
        tokens_in=3,
        tokens_out=3,
        latency_ms=0.5,
    )


def _score(example_id: str, sample_idx: int = 0) -> Score:
    return Score(
        example_id=example_id, sample_idx=sample_idx, hardness="easy", valid=True, ex_fast=True
    )


def _first_try(example_id: str) -> Generation:
    """A first reply whose query failed to run, so the sample was asked again."""
    return _generation(example_id).model_copy(update={"text": "SELEC 1", "sql": "SELEC 1"})


def _retried(example_id: str) -> Score:
    return _score(example_id).model_copy(update={"attempts": 2})


def _lines(*items: Generation | Score) -> str:
    return "".join(item.model_dump_json() + "\n" for item in items)


@pytest.mark.phase(2)
def test_outputs_recover_from_a_kill_mid_write(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run", "val")
    files = RunFiles(paths, EXPECTED)
    assert (files.done, files.resumed) == (set(), 0)
    assert not paths.scores.exists()
    files.append([_generation("q0"), _generation("q1")], [_score("q0"), _score("q1")])

    # Killed during the next batch: its generations were stored, its scores only begun.
    with paths.generations.open("a") as handle:
        handle.write(_lines(_generation("q2"), _generation("q3")))
    with paths.scores.open("a") as handle:
        handle.write(_score("q2").model_dump_json()[:30])

    reopened = RunFiles(paths, EXPECTED)
    assert reopened.done == {("q0", 0), ("q1", 0)} and reopened.resumed == 2
    assert list(reopened.generations) == [("q0", 0), ("q1", 0)]
    # The files hold just what was scored, whole lines only, before anything is added.
    assert paths.scores.read_text() == _lines(_score("q0"), _score("q1"))
    assert paths.generations.read_text() == _lines(_generation("q0"), _generation("q1"))
    reopened.append([_generation("q2")], [_score("q2")])
    assert [s.example_id for s in RunFiles(paths, EXPECTED).scores] == ["q0", "q1", "q2"]


@pytest.mark.phase(2)
@pytest.mark.parametrize(
    ("generations", "scores", "message"),
    [
        (_lines(_generation("q0")), _lines(_score("q0"), _score("q0")), "has q0 sample 0 twice"),
        (_lines(_generation("q0"), _generation("q0")), "", "has q0 sample 0 twice"),
        (_lines(_generation("q0")), _lines(_score("q1")), "q1 sample 0, which was never generated"),
        (_lines(_generation("z9")), _lines(_score("z9")), "z9 sample 0, which this run never asks"),
        (_lines(_generation("q0", 1)), _lines(_score("q0", 1)), "q0 sample 1, which this run"),
        (_lines(_generation("q0")), "{}\n" + _lines(_score("q0")), "line 1 is not a Score"),
    ],
)
def test_outputs_that_contradict_each_other_are_refused(
    tmp_path: Path, generations: str, scores: str, message: str
) -> None:
    paths = RunPaths(tmp_path / "run", "val")
    paths.scores.parent.mkdir(parents=True)
    paths.generations.write_text(generations)
    paths.scores.write_text(scores)
    with pytest.raises(RunCorrupt, match=message):
        RunFiles(paths, EXPECTED)
    # Nothing was rewritten over the evidence.
    assert (paths.generations.read_text(), paths.scores.read_text()) == (generations, scores)


@pytest.mark.phase(2)
def test_first_tries_recover_from_a_kill_mid_write(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run", "val")
    files = RunFiles(paths, EXPECTED)
    files.append(
        [_generation("q0"), _generation("q1")], [_score("q0"), _retried("q1")], [_first_try("q1")]
    )
    assert paths.outputs()["first_tries"] == "eval/val-first-tries.jsonl"

    # Killed during the next batch: q2's first try and generation were stored, no score.
    with paths.first_tries.open("a") as handle:
        handle.write(_lines(_first_try("q2")))
    with paths.generations.open("a") as handle:
        handle.write(_lines(_generation("q2")))

    reopened = RunFiles(paths, EXPECTED)
    assert reopened.done == {("q0", 0), ("q1", 0)}
    assert reopened.first_tries == {("q1", 0): _first_try("q1")}
    assert paths.first_tries.read_text() == _lines(_first_try("q1"))
    assert paths.generations.read_text() == _lines(_generation("q0"), _generation("q1"))

    # A batch's first tries must be its retried scores' ones, or nothing is written.
    for score, first_tries in (
        (_retried("q2"), []),
        (_retried("q2"), [_first_try("q3")]),
        (_score("q2"), [_first_try("q2")]),
    ):
        with pytest.raises(ValueError, match="must be its retried scores' ones"):
            reopened.append([_generation("q2")], [score], first_tries)
    assert paths.first_tries.read_text() == _lines(_first_try("q1"))
    assert paths.scores.read_text() == _lines(_score("q0"), _retried("q1"))
    reopened.append([_generation("q2")], [_retried("q2")], [_first_try("q2")])
    assert list(RunFiles(paths, EXPECTED).first_tries) == [("q1", 0), ("q2", 0)]

    # A run that retries nothing has no first tries file.
    plain = RunPaths(tmp_path / "plain", "val")
    RunFiles(plain, EXPECTED).append([_generation("q0")], [_score("q0")])
    assert not plain.first_tries.exists() and "first_tries" not in plain.outputs()


@pytest.mark.phase(2)
@pytest.mark.parametrize(
    ("scores", "first_tries", "message"),
    [
        ([_score("q0")], [_first_try("q0")], "has q0 sample 0, whose score wasn't retried"),
        ([_retried("q0")], [], "has q0 sample 0 retried, but .* lacks its first try"),
        ([_retried("q0")], [_first_try("q0")] * 2, "has q0 sample 0 twice"),
    ],
    ids=["not-retried", "missing", "twice"],
)
def test_first_tries_that_contradict_the_scores_are_refused(
    tmp_path: Path, scores: list[Score], first_tries: list[Generation], message: str
) -> None:
    paths = RunPaths(tmp_path / "run", "val")
    paths.scores.parent.mkdir(parents=True)
    stored = {
        paths.generations: _lines(*(_generation(s.example_id) for s in scores)),
        paths.scores: _lines(*scores),
        paths.first_tries: _lines(*first_tries),
    }
    for path, text in stored.items():
        path.write_text(text)
    with pytest.raises(RunCorrupt, match=message):
        RunFiles(paths, EXPECTED)
    assert {path: path.read_text() for path in stored} == stored


@pytest.mark.phase(2)
def test_scores_are_replaced_only_by_the_same_samples(tmp_path: Path) -> None:
    files = RunFiles(RunPaths(tmp_path / "run", "val"), EXPECTED)
    files.append([_generation("q0"), _generation("q1")], [_score("q0"), _score("q1")])
    judged = [s.model_copy(update={"ex_official": True}) for s in files.scores]
    files.replace_scores(judged)
    assert RunFiles(files.paths, EXPECTED).scores == judged
    with pytest.raises(ValueError, match="same samples, in order"):
        files.replace_scores(judged[::-1])


@pytest.mark.phase(2)
def test_start_resumes_only_the_same_unfinished_run(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run", "val")
    fresh = _manifest()
    started = start(paths, fresh)
    assert started.status == "running" and started.outputs == {"config": "config.yaml"}
    assert RunManifest.model_validate_json(paths.manifest.read_bytes()) == started
    assert yaml.safe_load(paths.config.read_text()) == fresh.config

    # A later attempt at the same run keeps the first start time.
    later = start(paths, _manifest(started_at=datetime(2026, 11, 15, 11, 0, tzinfo=UTC)))
    assert later.started_at == fresh.started_at and later.status == "running"

    for change, message in (
        ({"config": {"split": "val", "limit": 6}, "config_hash": "e" * 64}, r"options \(limit\)"),
        ({"git_sha": "b" * 40}, f"made at commit {'a' * 40}, not {'b' * 40}"),
        ({"data_manifest_hash": "f" * 64}, "made on other data"),
    ):
        with pytest.raises(RunConflict, match=message):
            start(paths, _manifest(**change))
    stored = paths.manifest.read_bytes()

    paths.manifest.write_text(_manifest(kind="grpo").model_dump_json())
    with pytest.raises(RunConflict, match="is a grpo run"):
        start(paths, fresh)
    paths.manifest.write_bytes(stored)
    finish(paths, later, None)
    with pytest.raises(RunConflict, match="has finished and its results stand"):
        start(paths, fresh)

    paths.manifest.write_text("{")
    with pytest.raises(RunCorrupt, match="is not a run manifest"):
        start(paths, fresh)
    paths.manifest.unlink()
    paths.scores.parent.mkdir(parents=True)
    paths.scores.write_text("")
    with pytest.raises(RunCorrupt, match="outputs but no manifest"):
        start(paths, fresh)


@pytest.mark.phase(2)
def test_finish_records_a_failure_and_clears_it_on_success(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run", "val")
    manifest = start(paths, _manifest())
    try:
        raise RuntimeError("the backend fell over")
    except RuntimeError as exc:
        failed = finish(paths, manifest, exc)
    assert failed.status == "failed" and failed.ended_at is not None
    assert failed.outputs == {"config": "config.yaml", "error": "error.txt"}
    text = paths.error.read_text()
    assert text.startswith("Traceback") and text.endswith("RuntimeError: the backend fell over\n")
    assert RunManifest.model_validate_json(paths.manifest.read_bytes()) == failed

    done = finish(paths, start(paths, _manifest()), None)
    assert done.status == "finished" and done.outputs == {"config": "config.yaml"}
    assert not paths.error.exists()


@pytest.mark.phase(2)
def test_hold_keeps_a_second_process_off_the_run(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "run", "val")
    with hold(paths):
        # flock locks belong to an open file, so a second open stands for another process.
        with (paths.root / LOCK_FILE).open("w") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RunLocked, match="run is running in another process"), hold(paths):
            pass
    with hold(paths):
        pass


@pytest.mark.phase(2)
def test_run_lock_never_dirties_the_tree(tmp_path: Path) -> None:
    # A checkout with the repo's ignore rules and a committed run, as in docs/runs.
    repo = tmp_path / "repo"
    committed = repo / "docs" / "runs" / "r1"
    (committed / "eval").mkdir(parents=True)
    (committed / MANIFEST_FILE).write_text("{}\n")
    shutil.copy2(ROOT / ".gitignore", repo / ".gitignore")
    for args in (
        ["init", "-q"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
        ["add", "."],
        ["commit", "-q", "-m", "init"],
    ):
        _git(repo, *args)
    head = git_sha(repo)

    # Reports hold the run folder they write in, and the lock stays after the hold.
    for root in (committed, repo / "runs" / "r2"):
        with hold(RunPaths(root, "val")):
            assert git_sha(repo) == head
        assert (root / LOCK_FILE).is_file()
        assert git_sha(repo) == head
    ignored = _git(repo, "check-ignore", "docs/runs/r1/.lock", "runs/r2/.lock")
    assert ignored.splitlines() == ["docs/runs/r1/.lock", "runs/r2/.lock"]

    # Only the lock: a new file in a committed run still makes the tree dirty.
    (committed / "eval" / "val-report.md").write_text("# Run r1\n")
    with pytest.raises(DirtyTreeError):
        git_sha(repo)


def _finished(tmp_path: Path, *, retried: bool = False) -> RunPaths:
    """A run that finished with two questions scored, in a folder named for its id; with
    ``retried``, q1 was asked again and its first try is stored."""
    manifest = _manifest(config_hash=config_hash({"split": "val", "limit": 5}))
    paths = RunPaths(tmp_path / manifest.run_id, "val")
    started = start(paths, manifest)
    second = (_retried("q1"), [_first_try("q1")]) if retried else (_score("q1"), [])
    RunFiles(paths, EXPECTED).append(
        [_generation("q0"), _generation("q1")], [_score("q0"), second[0]], second[1]
    )
    finish(paths, started, None)
    return paths


def _rewrite_manifest(paths: RunPaths, **changes: Any) -> None:
    manifest = RunManifest.model_validate_json(paths.manifest.read_bytes())
    paths.manifest.write_text(manifest.model_copy(update=changes).model_dump_json())


@pytest.mark.phase(2)
def test_read_run_reads_a_finished_run_as_stored(tmp_path: Path) -> None:
    paths = _finished(tmp_path)
    stored = read_run(paths.root)
    assert stored.manifest == RunManifest.model_validate_json(paths.manifest.read_bytes())
    assert stored.manifest.status == "finished" and stored.paths == paths
    assert stored.scores == (_score("q0"), _score("q1"))
    assert dict(stored.generations) == {("q0", 0): _generation("q0"), ("q1", 0): _generation("q1")}
    assert dict(stored.first_tries) == {}

    retried = read_run(_finished(tmp_path / "retried", retried=True).root)
    assert retried.scores == (_score("q0"), _retried("q1"))
    assert dict(retried.first_tries) == {("q1", 0): _first_try("q1")}
    assert retried.manifest.outputs["first_tries"] == "eval/val-first-tries.jsonl"

    moved = paths.root.rename(tmp_path / "another-name")
    with pytest.raises(RunCorrupt, match="is the manifest of run 2026-11-15-gold-eval-abc1234"):
        read_run(moved)
    with pytest.raises(FileNotFoundError, match=r"has no manifest\.json"):
        read_run(paths.root)


@pytest.mark.phase(2)
@pytest.mark.parametrize(
    ("spoil", "error", "message"),
    [
        (
            lambda p: p.scores.write_bytes(p.scores.read_bytes() + b'{"example_id"'),
            RunCorrupt,
            "ends in a line cut short",
        ),
        (
            lambda p: p.generations.write_text(
                p.generations.read_text() + _lines(_generation("q2"))
            ),
            RunCorrupt,
            "q2 sample 0, which has no score",
        ),
        (
            lambda p: p.scores.write_text(p.scores.read_text() + _lines(_score("q1"))),
            RunCorrupt,
            "has q1 sample 0 twice",
        ),
        (lambda p: p.generations.unlink(), RunCorrupt, "val-generations.jsonl is missing"),
        (
            lambda p: _rewrite_manifest(p, config={"split": "val", "limit": 6}),
            RunCorrupt,
            "doesn't hash to its config_hash",
        ),
        (
            lambda p: _rewrite_manifest(p, config={}, config_hash=config_hash({})),
            RunCorrupt,
            "names no split",
        ),
        (lambda p: _rewrite_manifest(p, status="running"), RunUnfinished, "is still running"),
        (lambda p: _rewrite_manifest(p, status="failed"), RunUnfinished, "error.txt says why"),
        (lambda p: _rewrite_manifest(p, kind="grpo"), ValueError, "is a grpo run, not an eval"),
    ],
    ids=[
        "cut-short",
        "unscored",
        "scored-twice",
        "no-generations",
        "edited-config",
        "no-split",
        "running",
        "failed",
        "other-kind",
    ],
)
def test_read_run_refuses_a_run_reports_cant_trust(
    tmp_path: Path, spoil: Callable[[RunPaths], object], error: type[Exception], message: str
) -> None:
    paths = _finished(tmp_path)
    spoil(paths)
    files = {path: path.read_bytes() for path in paths.root.rglob("*") if path.is_file()}
    with pytest.raises(error, match=message):
        read_run(paths.root)
    # Reading changed nothing.
    assert {path: path.read_bytes() for path in paths.root.rglob("*") if path.is_file()} == files


@pytest.mark.phase(2)
@pytest.mark.parametrize(
    ("spoil", "message"),
    [
        (lambda p: p.first_tries.unlink(), "q1 sample 0 retried, but .* lacks its first try"),
        (
            lambda p: p.first_tries.write_text(
                p.first_tries.read_text() + _lines(_first_try("q2"))
            ),
            "has q2 sample 0, which has no score",
        ),
        (
            lambda p: p.first_tries.write_text(
                p.first_tries.read_text() + _lines(_first_try("q0"))
            ),
            "has q0 sample 0, whose score wasn't retried",
        ),
        (
            lambda p: p.first_tries.write_bytes(p.first_tries.read_bytes()[:-1]),
            "first-tries.jsonl ends in a line cut short",
        ),
    ],
    ids=["missing", "unscored", "not-retried", "cut-short"],
)
def test_read_run_pairs_first_tries_with_retried_scores(
    tmp_path: Path, spoil: Callable[[RunPaths], object], message: str
) -> None:
    paths = _finished(tmp_path, retried=True)
    spoil(paths)
    with pytest.raises(RunCorrupt, match=message):
        read_run(paths.root)
