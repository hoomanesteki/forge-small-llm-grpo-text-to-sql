"""The eval runner end to end on the fixture data, with fake backends (FG-202)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from forge.core.types import Example, Generation, GenParams, Prompt, RunManifest, Score
from forge.data.splits import load_val
from forge.eval import runner
from forge.eval.config import OFFICIAL_ITEM_TIMEOUT_S, PRED_TIMEOUT_MS, EvalConfig
from forge.eval.run_files import RunConflict
from forge.eval.runner import run_eval
from forge.infer import hf_backend
from forge.infer.base import FakeBackend
from tests.fakes import WordTokenizer

ROOT = Path(__file__).resolve().parents[2]
# Runs until the executor's deadline stops it, in constant memory.
ENDLESS = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c"

# A run that SIGKILLs itself while asking for one question's reply, as a kill from outside
# would stop it: no exception, no cleanup, nothing written after.
KILLED_RUN = """
import os
import signal
import sys

from forge.data.splits import load_val
from forge.eval.config import EvalConfig
from forge.eval.runner import run_eval
from forge.infer.base import FakeBackend


def main() -> None:
    kill_at = sys.argv[1]
    gold = {example.id: example.gold_sql for example in load_val()}

    def reply(prompt, params, sample_idx):
        if prompt.example_id == kill_at:
            os.kill(os.getpid(), signal.SIGKILL)
        return "```sql\\n" + gold[prompt.example_id] + "\\n```"

    run_eval(
        EvalConfig(model="gold", backend="fake", run_id="killed"),
        backend=FakeBackend(reply, model_ref="gold"),
        allow_dirty=True,
        workers=1,
        batch_size=4,
    )


if __name__ == "__main__":
    main()
"""


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_log: Path) -> Path:
    """Runs and the gold cache under tmp_path, and a scratch dev log: no real file is touched."""
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "runs"


def _val() -> list[Example]:
    return sorted(load_val(), key=lambda example: example.id)


def _gold_reply(examples: Sequence[Example]) -> Callable[[Prompt, GenParams, int], str]:
    gold = {example.id: example.gold_sql for example in examples}

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        return f"```sql\n{gold[prompt.example_id]}\n```"

    return reply


def _stored(run_dir: Path, split: str = "val") -> tuple[RunManifest, list[Generation], list[Score]]:
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_bytes())
    eval_dir = run_dir / "eval"
    generations = [
        Generation.model_validate_json(line)
        for line in (eval_dir / f"{split}-generations.jsonl").read_bytes().splitlines()
    ]
    scores = [
        Score.model_validate_json(line)
        for line in (eval_dir / f"{split}-scores.jsonl").read_bytes().splitlines()
    ]
    return manifest, generations, scores


@pytest.mark.phase(2)
def test_resume_no_duplicates(spider_mini: Path, runs: Path, tmp_path: Path) -> None:
    val = _val()
    ids = [example.id for example in val]
    script = tmp_path / "killed_run.py"
    script.write_text(KILLED_RUN, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    # Batches of 4, killed while asking for the ninth reply: two batches are stored.
    killed = subprocess.run(
        [sys.executable, str(script), ids[8]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    run_dir = runs / "killed"
    first, generations, scores = _stored(run_dir)
    assert first.status == "running"
    assert [s.example_id for s in scores] == ids[:8] == [g.example_id for g in generations]

    # A kill in the middle of the next write would leave a generation without its score
    # and a score line cut short: add both, as the third batch would have begun them.
    stray = Generation(
        example_id=ids[8],
        model_ref="gold",
        text="```sql\nSELECT 1\n```",
        sql="SELECT 1",
        tokens_in=1,
        tokens_out=1,
        latency_ms=0.0,
    )
    with (run_dir / "eval" / "val-generations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(stray.model_dump_json() + "\n")
    torn = Score(example_id=ids[8], hardness=val[8].hardness, valid=True, ex_fast=False)
    with (run_dir / "eval" / "val-scores.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(torn.model_dump_json()[:40])

    asked: list[str] = []
    reply = _gold_reply(val)

    def counting(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        asked.append(prompt.example_id)
        return reply(prompt, params, sample_idx)

    run = run_eval(
        EvalConfig(model="gold", backend="fake", run_id="killed"),
        backend=FakeBackend(counting, model_ref="gold"),
        allow_dirty=True,
        workers=1,
        batch_size=4,
    )
    # Only what the kill left undone was asked again, and everything is stored once.
    assert asked == ids[8:]
    manifest, generations, scores = _stored(run_dir)
    assert [s.example_id for s in scores] == ids == [g.example_id for g in generations]
    assert [g.sql for g in generations] == [e.gold_sql for e in val]
    assert all(s.ex_fast and s.ex_official for s in scores)
    assert run.resumed == 8 and list(run.scores) == scores
    assert manifest.status == "finished" and manifest.started_at == first.started_at


@pytest.mark.phase(2)
def test_official_ex_bounds(spider_mini: Path, runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    val = _val()[:3]
    calls: list[tuple[list[tuple[str, str | None, str]], dict[str, Any]]] = []
    official = runner.official_exec_match

    def spy(items: Sequence[tuple[str, str | None, str]], **kwargs: Any) -> list[bool]:
        calls.append((list(items), kwargs))
        return official(items, **kwargs)

    monkeypatch.setattr(runner, "official_exec_match", spy)
    reply = _gold_reply(val)

    def slow_second(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        if prompt.example_id == val[1].id:
            return f"```sql\n{ENDLESS}\n```"
        return reply(prompt, params, sample_idx)

    run = run_eval(
        EvalConfig(model="gold", backend="fake", limit=3, run_id="bounds"),
        backend=FakeBackend(slow_second, model_ref="gold"),
        allow_dirty=True,
        workers=1,
    )
    # --limit keeps the first questions by id.
    assert [s.example_id for s in run.scores] == [e.id for e in val]
    slow = run.scores[1]
    assert not slow.valid and not slow.ex_fast and slow.ex_official is False
    assert all(s.ex_fast and s.ex_official for s in (run.scores[0], run.scores[2]))
    # The query that ran out its 2 s never reached the evaluator, whose items get 10 s.
    ((items, kwargs),) = calls
    assert [pred for _, pred, _ in items] == [val[0].gold_sql, val[2].gold_sql]
    assert kwargs["item_timeout_s"] == OFFICIAL_ITEM_TIMEOUT_S == 10.0
    assert PRED_TIMEOUT_MS == 2000
    assert run.manifest.config["execution"]["timeout_ms"] == 2000
    assert run.manifest.config["execution"]["official_item_timeout_s"] == 10.0


@pytest.mark.phase(2)
def test_failed_run_is_recorded_then_resumed(spider_mini: Path, runs: Path) -> None:
    val = _val()
    reply = _gold_reply(val)

    def breaks_on_sixth(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        if prompt.example_id == val[5].id:
            raise RuntimeError("the backend fell over")
        return reply(prompt, params, sample_idx)

    config = EvalConfig(model="gold", backend="fake", run_id="flaky")
    with pytest.raises(RuntimeError, match="the backend fell over"):
        run_eval(
            config,
            backend=FakeBackend(breaks_on_sixth, model_ref="gold"),
            allow_dirty=True,
            workers=1,
            batch_size=4,
        )
    manifest, _, scores = _stored(runs / "flaky")
    assert manifest.status == "failed" and manifest.ended_at is not None
    assert manifest.outputs["error"] == "error.txt"
    error = (runs / "flaky" / "error.txt").read_text()
    assert error.startswith("Traceback") and "RuntimeError: the backend fell over" in error
    assert len(scores) == 4

    run = run_eval(
        config, backend=FakeBackend(reply, model_ref="gold"), allow_dirty=True, workers=1
    )
    assert run.manifest.status == "finished" and run.resumed == 4
    assert "error" not in run.manifest.outputs
    assert not (runs / "flaky" / "error.txt").exists()
    assert [s.example_id for s in run.scores] == [e.id for e in val]

    # A finished run's results stand: the same id is refused, not run again.
    with pytest.raises(RunConflict, match="has finished"):
        run_eval(config, backend=FakeBackend(reply, model_ref="gold"), allow_dirty=True)


@pytest.mark.phase(2)
def test_hf_backend_is_built_from_the_options(
    spider_mini: Path, runs: Path, dev_log: Path, with_dev: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val = _val()
    built: list[tuple[str, str | None]] = []
    tokenizer = WordTokenizer()

    class StubHf:
        """HfBackend as the runner sees it, without loading a model."""

        name = "hf"

        def __init__(self, model_ref: str, *, adapter_ref: str | None = None) -> None:
            built.append((model_ref, adapter_ref))
            self.tokenizer = tokenizer
            self._fake = FakeBackend(_gold_reply(val), model_ref=model_ref)

        def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
            return self._fake.generate(prompts, params)

    monkeypatch.setattr(hf_backend, "HfBackend", StubHf)
    run = run_eval(
        EvalConfig(model="org/model@abc123", adapter="org/adapter", limit=2, run_id="hf"),
        allow_dirty=True,
        workers=1,
    )
    assert built == [("org/model@abc123", "org/adapter")]
    assert (run.manifest.model_ref, run.manifest.adapter_ref) == ("org/model@abc123", "org/adapter")
    # The prompts were measured with the model's own tokenizer, assistant turn opened.
    assert len(tokenizer.calls) == 2
    assert all(kwargs["add_generation_prompt"] for _, kwargs in tokenizer.calls)
    assert all(s.ex_fast and s.ex_official for s in run.scores)

    # A model that fails to load is found out before dev is read, so no look is spent.
    def unloadable(model_ref: str, *, adapter_ref: str | None = None) -> StubHf:
        raise OSError(f"{model_ref} is not on the Hub")

    monkeypatch.setattr(hf_backend, "HfBackend", unloadable)
    monkeypatch.setenv("FORGE_DATA_DIR", str(with_dev))
    with pytest.raises(OSError, match="not on the Hub"):
        run_eval(
            EvalConfig(model="org/missing", split="dev", reason="check", run_id="no-model"),
            allow_dirty=True,
        )
    assert not dev_log.exists()
    failed = RunManifest.model_validate_json((runs / "no-model" / "manifest.json").read_bytes())
    assert failed.status == "failed"
