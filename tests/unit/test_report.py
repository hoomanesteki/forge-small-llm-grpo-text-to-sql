"""Reports from runs' stored files.

Each test stores finished runs the way ``forge eval run`` does, over the fixture's val
questions (``spider-train-00010`` to ``00029`` on flight_1, five of each hardness).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from forge.core.hashing import canonical_json
from forge.core.types import Generation, RunManifest, Score, Split
from forge.data.manifest import data_manifest_hash
from forge.data.spider import load_raw
from forge.data.splits import split_examples
from forge.eval.config import EvalConfig
from forge.eval.report import load_run
from forge.eval.run_files import RunCorrupt, RunUnfinished
from forge.ops.dev_usage import log_dev_use, read_uses
from forge.ops.manifest import config_hash

AT = datetime(2026, 11, 15, 9, 30, tzinfo=UTC)
VAL = [f"spider-train-{n:05d}" for n in range(10, 30)]
GOLD = "<gold>"  # stands for the example's own gold query


class Outcome(NamedTuple):
    """How one sample went, as its score and generation record it."""

    valid: bool
    fast: bool
    official: bool
    error_class: str | None
    sql: str | None


RIGHT = Outcome(True, True, True, None, GOLD)
NO_SQL = Outcome(False, False, False, "no_sql", None)
SYNTAX = Outcome(False, False, False, "syntax", "SELEC count(*) FROM Aircraft")
SCHEMA = Outcome(False, False, False, "schema_link", "SELECT name FROM Pilot")
WHERE = Outcome(True, False, False, "where", "SELECT aid FROM Aircraft WHERE distance > 9")
GROUP = Outcome(True, False, False, "group_order", "SELECT origin FROM Flight GROUP BY origin")
VALUE = Outcome(True, False, False, "value", "SELECT count(*) FROM Aircraft WHERE aid = 3")
# The checkers disagree: only fast EX matches, or only official EX does.
ONLY_FAST = Outcome(True, True, False, None, "SELECT DISTINCT name FROM Aircraft")
ONLY_OFFICIAL = Outcome(True, False, True, "projection", "SELECT name FROM Aircraft")

# A reply longer than a report shows, that a fence of three backticks can't hold, with
# characters that don't print.
REPLY = "I would write ``` here,\x00 then\u2028 stop. " + "Padding words. " * 60

MIXED = dict(
    zip(
        VAL,
        [
            *(RIGHT, NO_SQL, RIGHT, SYNTAX, RIGHT),  # 00010 to 00014
            *(WHERE, ONLY_OFFICIAL, RIGHT, ONLY_OFFICIAL, RIGHT),  # 00015 to 00019
            *(NO_SQL, NO_SQL, SCHEMA, RIGHT, GROUP),  # 00020 to 00024, all hard
            *(NO_SQL, ONLY_FAST, NO_SQL, NO_SQL, RIGHT),  # 00025 to 00029, all extra
        ],
        strict=True,
    )
)


@pytest.fixture
def runs(spider_mini: Path, dev_log: Path, tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _store(
    runs: Path,
    run_id: str,
    outcomes: Mapping[str, Outcome],
    *,
    split: Split = "val",
    model: str = "gold",
    status: str = "finished",
    data_dir: Path | None = None,
    samples: int = 1,
    reply: str = REPLY,
) -> Path:
    """Store a run's manifest, generations and scores as the eval runner does."""
    examples = {e.id: e for e in split_examples(load_raw(data_dir))[split]}
    options = EvalConfig(
        model=model, backend="fake", split=split, n=samples, temperature=0.7 if samples > 1 else 0
    )
    config: dict[str, Any] = json.loads(canonical_json(options.manifest_config()))
    generations, scores = [], []
    for example_id, outcome in outcomes.items():
        example = examples[example_id]
        sql = example.gold_sql if outcome.sql == GOLD else outcome.sql
        for sample in range(samples):
            generations.append(
                Generation(
                    example_id=example_id,
                    model_ref=model,
                    sample_idx=sample,
                    text=reply if sql is None else f"```sql\n{sql}\n```",
                    sql=sql,
                    tokens_in=50,
                    tokens_out=12,
                    latency_ms=3.0,
                )
            )
            scores.append(
                Score(
                    example_id=example_id,
                    sample_idx=sample,
                    hardness=example.hardness,
                    valid=outcome.valid,
                    ex_fast=outcome.fast,
                    ex_official=outcome.official,
                    error_class=outcome.error_class,
                )
            )
    root = runs / run_id
    (root / "eval").mkdir(parents=True)
    for name, items in (("generations", generations), ("scores", scores)):
        lines = "".join(item.model_dump_json() + "\n" for item in items)
        (root / "eval" / f"{split}-{name}.jsonl").write_text(lines, encoding="utf-8")
    manifest = RunManifest.model_validate(
        {
            "run_id": run_id,
            "kind": "eval",
            "git_sha": "a" * 40,
            "config": config,
            "config_hash": config_hash(config),
            "data_manifest_hash": data_manifest_hash(data_dir),
            "model_ref": model,
            "platform": "local-cpu",
            "seed": 0,
            "started_at": AT,
            "ended_at": AT,
            "status": status,
        }
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return root


def _edit_first_score(root: Path, **changes: Any) -> None:
    path = root / "eval" / "val-scores.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = Score.model_validate_json(lines[0]).model_copy(update=changes).model_dump_json()
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.phase(2)
def test_report_refuses_runs_it_cant_trust(runs: Path, spider_mini: Path, tmp_path: Path) -> None:
    for status in ("running", "failed"):
        _store(runs, status, MIXED, status=status)
        with pytest.raises(
            RunUnfinished, match=f"resume it with `forge eval run --run-id {status}`"
        ):
            load_run(status, runs_root=runs)

    # Made on other data: the fixture with one more question.
    other = Path(shutil.copytree(spider_mini, tmp_path / "other-data"))
    items = json.loads((other / "train_spider.json").read_text())
    (other / "train_spider.json").write_text(json.dumps([*items, items[0]]))
    _store(runs, "elsewhere", MIXED, data_dir=other)
    with pytest.raises(ValueError, match="run elsewhere was made on other data"):
        load_run("elsewhere", runs_root=runs)

    # Scores a finished run can't hold: one without an official verdict, and a class on
    # a sample fast EX matched.
    for run_id, change in (
        ("no-verdict", {"ex_official": None}),
        ("odd", {"error_class": "where"}),
    ):
        _edit_first_score(_store(runs, run_id, MIXED), **change)
        with pytest.raises(RunCorrupt, match="spider-train-00010 sample 0"):
            load_run(run_id, runs_root=runs)

    with pytest.raises(ValueError, match="names a folder"):
        load_run("../mixed", runs_root=runs)
    with pytest.raises(FileNotFoundError, match="no run in"):
        load_run("missing", runs_root=runs)


@pytest.mark.phase(2)
def test_dev_report_belongs_to_the_runs_logged_look(
    with_dev: Path, dev_log: Path, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    dev = [example.id for example in split_examples(load_raw(with_dev))["dev"]]
    _store(runs, "dev-run", dict.fromkeys(dev, RIGHT), split="dev", data_dir=with_dev)
    with pytest.raises(ValueError, match=r"answered dev questions, but .* has no row for it"):
        load_run("dev-run", runs_root=runs, data_dir=with_dev)
    log_dev_use("dev-run", "check the dev report", path=dev_log)
    run = load_run("dev-run", runs_root=runs, data_dir=with_dev)
    assert (run.options.split, sorted(run.examples)) == ("dev", sorted(dev))
    # Reading the run's answers again spends no second look.
    assert [run_id for _, run_id, _ in read_uses(dev_log)] == ["dev-run"]
