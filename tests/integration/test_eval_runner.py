"""The eval runner end to end on the fixture data, with fake backends (FG-202)."""

from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from forge import cli
from forge.core.types import Example, Generation, GenParams, Prompt, RunManifest, Score
from forge.data.manifest import data_manifest_hash
from forge.data.splits import load_val
from forge.eval import runner
from forge.eval.config import OFFICIAL_ITEM_TIMEOUT_S, PRED_TIMEOUT_MS, EvalConfig
from forge.eval.run_files import RunConflict
from forge.eval.runner import run_eval
from forge.eval.taxonomy import ERROR_CLASSES
from forge.infer import hf_backend
from forge.infer.base import FakeBackend
from forge.ops.dev_usage import read_uses
from tests.fakes import WordTokenizer
from tests.report_trace import untraced

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


def _summary(name: str, hits: int, total: int) -> str:
    return f"{name:<11}  {hits:>5}/{total}  {hits / total:.2%}"


def _file_hashes(folder: Path) -> dict[str, str]:
    return {
        path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


@pytest.mark.phase(2)
def test_end_to_end_with_fake_backend(spider_mini: Path, runs: Path, tmp_path: Path) -> None:
    before = _file_hashes(spider_mini / "database")
    val = _val()
    cli_runner = CliRunner()

    gold = cli_runner.invoke(
        cli.app, ["eval", "run", "--backend", "fake", "--model", "gold", "--allow-dirty"]
    )
    assert gold.exit_code == 0, gold.output
    (run_dir,) = runs.iterdir()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-gold-eval-[0-9a-f]{7}", run_dir.name)
    manifest, generations, scores = _stored(run_dir)
    assert (
        [s.example_id for s in scores] == [e.id for e in val] == [g.example_id for g in generations]
    )
    assert [g.sql for g in generations] == [e.gold_sql for e in val]
    assert [s.hardness for s in scores] == [e.hardness for e in val]
    assert all(s.valid and s.ex_fast and s.ex_official for s in scores)
    assert all(s.error_class is None for s in scores)
    assert _summary("fast EX", 20, 20) in gold.stdout
    assert _summary("official EX", 20, 20) in gold.stdout
    assert "scored 20 of 20" in gold.stderr
    assert "generated 20 samples in " in gold.stdout

    # The manifest says what made the scores, and lists the files beside it.
    assert manifest.run_id == run_dir.name
    assert (manifest.kind, manifest.status, manifest.model_ref) == ("eval", "finished", "gold")
    assert manifest.config["backend"] == "fake" and manifest.config["split"] == "val"
    assert manifest.data_manifest_hash == data_manifest_hash(spider_mini)
    assert manifest.ended_at is not None and manifest.ended_at >= manifest.started_at
    assert manifest.outputs == {
        "config": "config.yaml",
        "generations": "eval/val-generations.jsonl",
        "scores": "eval/val-scores.jsonl",
    }
    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == manifest.config
    (cache_file,) = (tmp_path / "cache" / "gold").iterdir()
    assert cache_file.name == f"val-{manifest.data_manifest_hash[:12]}.jsonl"
    built = cache_file.stat()

    constant = cli_runner.invoke(
        cli.app,
        [
            *("eval", "run", "--backend", "fake", "--model", "select-1"),
            *("--run-id", "select-one", "--allow-dirty"),
        ],
    )
    assert constant.exit_code == 0, constant.output
    _, generations, scores = _stored(runs / "select-one")
    assert [g.sql for g in generations] == ["SELECT 1"] * 20
    assert all(s.valid and not s.ex_fast and s.ex_official is False for s in scores)
    # SELECT 1 reads no table, and every val gold query reads one.
    assert all(s.error_class == "wrong_tables" for s in scores)
    assert _summary("fast EX", 0, 20) in constant.stdout
    assert _summary("official EX", 0, 20) in constant.stdout

    # The second run read the gold results the first one stored, without rebuilding them.
    assert [path.name for path in (tmp_path / "cache" / "gold").iterdir()] == [cache_file.name]
    assert (cache_file.stat().st_ino, cache_file.stat().st_mtime_ns) == (
        built.st_ino,
        built.st_mtime_ns,
    )
    # I2: nothing a run did changed a database.
    assert _file_hashes(spider_mini / "database") == before


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
def test_dev_requires_reason(
    with_dev: Path, runs: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_DATA_DIR", str(with_dev))
    real_log = ROOT / "docs" / "dev-usage.md"
    real = real_log.read_bytes()
    cli_runner = CliRunner()
    dev = ["eval", "run", "--backend", "fake", "--model", "gold", "--split", "dev", "--allow-dirty"]

    refused = cli_runner.invoke(cli.app, dev)
    assert refused.exit_code == 1
    assert "error: --split dev needs --reason" in refused.stderr
    blank = cli_runner.invoke(cli.app, [*dev, "--reason", "  "])
    assert blank.exit_code == 1 and "needs --reason" in blank.stderr
    # A backend that isn't built yet is refused before dev is read, so it spends no look.
    later = cli_runner.invoke(
        cli.app,
        [*dev[:2], "--model", "some/model", "--backend", "vllm", "--split", "dev", "--reason", "x"],
    )
    assert later.exit_code == 1 and "comes with FG-306" in later.stderr
    assert not dev_log.exists()
    assert not runs.exists()

    allowed = cli_runner.invoke(
        cli.app, [*dev, "--reason", "FG-202 dev guard", "--run-id", "dev-1"]
    )
    assert allowed.exit_code == 0, allowed.output
    assert [use[1:] for use in read_uses(dev_log)] == [("dev-1", "FG-202 dev guard")]
    manifest, _, scores = _stored(runs / "dev-1", "dev")
    assert manifest.status == "finished" and manifest.config["split"] == "dev"
    assert [s.example_id for s in scores] == [f"spider-dev-{i:05d}" for i in range(10)]
    assert all(s.ex_fast and s.ex_official for s in scores)

    # A reason with another split is a mistake, not a dev use.
    val_reason = cli_runner.invoke(
        cli.app, ["eval", "run", "--backend", "fake", "--model", "gold", "--reason", "x"]
    )
    assert val_reason.exit_code == 1 and "--reason goes with --split dev" in val_reason.stderr
    assert len(read_uses(dev_log)) == 1
    assert real_log.read_bytes() == real


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
def test_later_options_are_refused_up_front(spider_mini: Path, runs: Path) -> None:
    cli_runner = CliRunner()
    for options, message in (
        (["--backend", "vllm"], "the vllm backend is not available yet: it comes with FG-306"),
        (["--backend", "llamacpp"], "comes with FG-702"),
        (["--store", "hub"], "--store hub is available after FG-304"),
        (["--n", "2"], "needs a temperature above 0"),
        (["--run-id", "../escape"], "names a folder"),
    ):
        result = cli_runner.invoke(cli.app, ["eval", "run", "--model", "some/model", *options])
        assert result.exit_code == 1, (options, result.output)
        assert result.stderr.startswith("error: ") and message in result.stderr, result.stderr
    assert not runs.exists()


@pytest.mark.phase(2)
def test_hf_backend_is_built_from_the_options(
    spider_mini: Path, runs: Path, dev_log: Path, with_dev: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val = _val()
    built: list[tuple[str, str | None, str]] = []
    tokenizer = WordTokenizer()

    class StubHf:
        """HfBackend as the runner sees it, without loading a model."""

        name = "hf"

        def __init__(
            self, model_ref: str, *, adapter_ref: str | None = None, device: str = "auto"
        ) -> None:
            built.append((model_ref, adapter_ref, device))
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
    # The device is asked for by name, so a run on a Mac doesn't drift onto MPS unasked.
    assert built == [("org/model@abc123", "org/adapter", "cpu")]
    assert (run.manifest.model_ref, run.manifest.adapter_ref) == ("org/model@abc123", "org/adapter")
    assert (run.manifest.platform, run.manifest.config["device"]) == ("local-cpu", "cpu")
    # The prompts were measured with the model's own tokenizer, assistant turn opened.
    assert len(tokenizer.calls) == 2
    assert all(kwargs["add_generation_prompt"] for _, kwargs in tokenizer.calls)
    assert all(s.ex_fast and s.ex_official for s in run.scores)

    # A model that fails to load is found out before dev is read, so no look is spent.
    def unloadable(model_ref: str, *, adapter_ref: str | None = None, device: str) -> StubHf:
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


@pytest.mark.phase(2)
def test_mps_run_is_labeled_a_local_mps_pilot(
    spider_mini: Path, runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val = _val()
    devices: list[str] = []

    class StubHf:
        name = "hf"

        def __init__(self, model_ref: str, *, adapter_ref: str | None, device: str) -> None:
            devices.append(device)
            self.tokenizer = WordTokenizer()
            self._fake = FakeBackend(_gold_reply(val), model_ref=model_ref)

        def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
            return self._fake.generate(prompts, params)

    monkeypatch.setattr(hf_backend, "HfBackend", StubHf)
    cli_runner = CliRunner()
    args = ["--model", "org/model@abc123", "--device", "mps", "--limit", "3", "--run-id", "pilot"]
    result = cli_runner.invoke(cli.app, ["eval", "run", *args, "--workers", "1", "--allow-dirty"])
    assert result.exit_code == 0, result.output
    assert "platform local-mps\n" in result.stdout
    assert devices == ["mps"]
    manifest = RunManifest.model_validate_json((runs / "pilot" / "manifest.json").read_bytes())
    assert (manifest.platform, manifest.config["device"]) == ("local-mps", "mps")

    # The report says where the numbers were made, and every number still traces (I4).
    assert cli_runner.invoke(cli.app, ["report", "run", "pilot"]).exit_code == 0
    report = (runs / "pilot" / "eval" / "val-report.md").read_text(encoding="utf-8")
    assert "\n| Platform | `local-mps` |\n" in report
    assert untraced(report, runs / "pilot", data_dir=spider_mini) == []


@pytest.mark.phase(2)
def test_wrong_predictions_get_their_error_class(spider_mini: Path, runs: Path) -> None:
    val = _val()
    intersect = next(e.gold_sql for e in val if e.id == "spider-train-00025")
    # One wrong reply for each class, each run for real on flight_1; the rest answer gold.
    wrong = {
        "spider-train-00010": ("no_sql", "There are some aircraft."),
        "spider-train-00011": ("syntax", "SELECT count(* FROM Aircraft"),
        "spider-train-00012": ("schema_link", "SELECT name, range_km FROM Aircraft"),
        "spider-train-00013": ("wrong_tables", "SELECT origin, distance FROM Flight"),
        "spider-train-00027": (
            "join",
            "SELECT T2.name FROM Certificate AS T1 JOIN Aircraft AS T2 ON T2.aid = T1.eid "
            "GROUP BY T1.aid ORDER BY count(*) DESC LIMIT 1",
        ),
        "spider-train-00017": ("projection", "SELECT name FROM Aircraft WHERE aid = 12"),
        "spider-train-00019": (
            "aggregation",
            "SELECT min(distance), sum(distance), max(distance) FROM Aircraft",
        ),
        "spider-train-00014": ("where", "SELECT aid FROM Aircraft WHERE distance < 1000"),
        "spider-train-00024": (
            "group_order",
            "SELECT origin FROM Flight GROUP BY origin ORDER BY count(*) ASC LIMIT 1",
        ),
        "spider-train-00016": (
            "value",
            "SELECT count(*) FROM Aircraft WHERE distance BETWEEN 1000 AND 2000",
        ),
        "spider-train-00025": ("other", intersect.replace(" INTERSECT ", " UNION ")),
    }
    gold = _gold_reply(val)

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        if prompt.example_id not in wrong:
            return gold(prompt, params, sample_idx)
        text = wrong[prompt.example_id][1]
        return text if text.startswith("There") else f"```sql\n{text}\n```"

    run = run_eval(
        EvalConfig(model="gold", backend="fake", run_id="classes"),
        backend=FakeBackend(reply, model_ref="gold"),
        allow_dirty=True,
        workers=1,
    )
    _, _, scores = _stored(runs / "classes")
    assert scores == list(run.scores)
    assert {s.example_id: s.error_class for s in scores} == {
        e.id: wrong[e.id][0] if e.id in wrong else None for e in val
    }
    assert sorted({s.error_class for s in scores if s.error_class}) == sorted(ERROR_CLASSES)
    assert all(s.ex_fast == (s.error_class is None) for s in scores)


@pytest.mark.phase(2)
def test_retry_through_the_cli(
    spider_mini: Path, runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val = _val()
    gold = _gold_reply(val)
    told: list[str] = []
    # One first query runs out its time, another doesn't parse; both are fixed on retry.
    first_replies = {val[1].id: ENDLESS, val[2].id: "SELEC name, distance FROM Aircraft"}

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        if len(prompt.messages) > 2:
            told.append(prompt.messages[-1]["content"])
        elif prompt.example_id in first_replies:
            return f"```sql\n{first_replies[prompt.example_id]}\n```"
        return gold(prompt, params, sample_idx)

    def fake_backend(model: str, examples: Sequence[Example]) -> FakeBackend:
        return FakeBackend(reply, model_ref=model)

    calls: list[list[tuple[str, str | None, str]]] = []
    official = runner.official_exec_match

    def spy(items: Sequence[tuple[str, str | None, str]], **kwargs: Any) -> list[bool]:
        calls.append(list(items))
        return official(items, **kwargs)

    monkeypatch.setattr(runner, "fake_backend", fake_backend)
    monkeypatch.setattr(runner, "official_exec_match", spy)
    cli_runner = CliRunner()
    args = ["eval", "run", "--backend", "fake", "--model", "gold", "--run-id", "retry"]
    result = cli_runner.invoke(cli.app, [*args, "--retry", "1", "--allow-dirty"])
    assert result.exit_code == 0, result.output
    assert _summary("official EX", 20, 20) in result.stdout
    assert (
        "retried 2 of 20 samples whose first query failed to run; official EX without "
        "retry 18/20  90.00%"
    ) in result.stdout
    # The timeout is told as the deadline, not as the time the kill took.
    assert told == [
        "That query failed with: timed out after 2000 ms. Reply with a corrected query.",
        'That query failed with: near "SELEC": syntax error. Reply with a corrected query.',
    ]

    manifest, generations, scores = _stored(runs / "retry")
    assert manifest.config["retry"] == 1 and manifest.status == "finished"
    assert [g.sql for g in generations] == [e.gold_sql for e in val]
    assert {s.example_id: (s.attempts, s.first_try_ex) for s in scores if s.attempts > 1} == {
        val[1].id: (2, False),
        val[2].id: (2, False),
    }
    # One official batch: every scored reply, and the first try that didn't time out.
    (items,) = calls
    assert sorted(pred or "" for _, pred, _ in items) == sorted(
        [*(e.gold_sql for e in val), first_replies[val[2].id]]
    )

    reported = cli_runner.invoke(cli.app, ["report", "run", "retry"])
    assert reported.exit_code == 0, reported.output
    report = (runs / "retry" / "eval" / "val-report.md").read_text()
    assert "\n## Retry\n" in report
    assert "\n| retried: the first query failed to run | 2 |\n" in report
