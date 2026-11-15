"""Reports from runs' stored files.

Each test stores finished runs the way ``forge eval run`` does, over the fixture's val
questions (``spider-train-00010`` to ``00029`` on flight_1, five of each hardness).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
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
from forge.eval.metrics import Rate, gold_clusters, summarize
from forge.eval.report import load_run, write_run_report
from forge.eval.run_files import RunCorrupt, RunUnfinished
from forge.ops.dev_usage import log_dev_use, read_uses
from forge.ops.manifest import config_hash

AT = datetime(2026, 11, 15, 9, 30, tzinfo=UTC)
LATER = datetime(2026, 12, 1, 18, 45, 12, tzinfo=UTC)
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


def _scores(root: Path) -> list[Score]:
    lines = (root / "eval" / "val-scores.jsonl").read_text().splitlines()
    return [Score.model_validate_json(line) for line in lines]


def _edit_first_score(root: Path, **changes: Any) -> None:
    path = root / "eval" / "val-scores.jsonl"
    lines = path.read_text().splitlines()
    lines[0] = Score.model_validate_json(lines[0]).model_copy(update=changes).model_dump_json()
    path.write_text("\n".join(lines) + "\n")


def _clusters() -> dict[str, str]:
    return gold_clusters(e for e in split_examples(load_raw())["val"] if e.id in VAL)


def _pct(share: float) -> str:
    return f"{100 * share:.1f}%"


def _ci(rate: Rate | None) -> str:
    assert rate is not None
    return f"{_pct(rate.value)} [{_pct(rate.low)}, {_pct(rate.high)}]"


def _section(text: str, heading: str) -> str:
    """The lines under ``## heading``, up to the next section."""
    return text.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def _png_chunks(data: bytes) -> list[bytes]:
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    kinds, at = [], 8
    while at < len(data):
        (size,) = struct.unpack(">I", data[at : at + 4])
        kinds.append(data[at + 4 : at + 8])
        at += 12 + size
    return kinds


@pytest.mark.phase(2)
def test_report_renders_from_scores(runs: Path) -> None:
    root = _store(runs, "mixed", MIXED)
    written = write_run_report("mixed", runs_root=runs, now=AT)
    assert (written.report, written.chart) == (
        root / "eval" / "val-report.md",
        root / "eval" / "val-hardness.png",
    )
    text = written.report.read_text(encoding="utf-8")
    summary = summarize(_scores(root), clusters=_clusters())

    assert text.startswith(f"# Run mixed: official EX {_ci(summary.ex_official)} on val\n")
    assert [line for line in text.splitlines() if line.startswith("Generated: ")] == [
        "Generated: 2026-11-15 09:30:00 UTC"
    ]
    for row in (
        "| Model | `gold` |",
        "| Adapter | none |",
        "| Backend | `fake` |",
        "| Prompt version | `p1` |",
        "| Split | val: 20 questions, 1 sample each |",
        "| Generation | greedy, at most 256 new tokens |",
        f"| Git commit | `{'a' * 40}` |",
        f"| Data manifest hash | `{data_manifest_hash()}` |",
        "| Manifest | [manifest.json](../manifest.json) |",
    ):
        assert f"\n{row}\n" in text

    # 9 of 20 by official EX: the 7 right answers and the 2 only official EX matches.
    accuracy = _section(text, "Execution accuracy")
    assert f"| valid SQL | 12 | 20 | {_ci(summary.valid)} |" in accuracy
    assert f"| fast EX | 8 | 20 | {_ci(summary.ex_fast)} |" in accuracy
    assert f"| official EX | 9 | 20 | {_ci(summary.ex_official)} |" in accuracy
    assert "disagree on 3 samples: only fast EX matches 1, and only official EX 2." in accuracy
    assert "20 questions in 12 clusters." in accuracy

    hardness = _section(text, "By hardness")
    right = {"easy": (2, 3), "medium": (3, 4), "hard": (1, 1), "extra": (2, 1)}
    for level, ex in summary.by_hardness.items():
        assert ex.ex_official is not None
        assert (ex.ex_fast.hits, ex.ex_official.hits) == right[level]
        assert f"| {level} | 5 | {_ci(ex.ex_fast)} | {_ci(ex.ex_official)} |" in hardness
    assert "![Official EX by hardness, with 95% intervals](val-hardness.png)" in hardness

    # Official EX fails 11 samples. The one only fast EX matches has no class, and the
    # two only official EX matches are no failures.
    classes = _section(text, "Error classes")
    for row in (
        "| no_sql | 6 | 54.5% |",
        "| syntax | 1 | 9.1% |",
        "| schema_link | 1 | 9.1% |",
        "| projection | 0 | 0.0% |",
        "| where | 1 | 9.1% |",
        "| group_order | 1 | 9.1% |",
        "| unclassified | 1 | 9.1% |",
    ):
        assert f"\n{row}\n" in classes
    assert "Unclassified failures are those only official EX finds" in classes

    # The three most common classes: no_sql, then syntax and schema_link, which come
    # before where, group_order and unclassified in the taxonomy's order.
    failures = _section(text, "Example failures")
    assert [line for line in failures.splitlines() if line.startswith("### ")] == [
        "### no_sql: 5 of 6 shown",
        "### syntax: 1 of 1 shown",
        "### schema_link: 1 of 1 shown",
    ]
    no_sql = [example_id for example_id, outcome in MIXED.items() if outcome is NO_SQL]
    picked = sorted(no_sql, key=lambda i: hashlib.sha256(f"{i}/0".encode()).hexdigest())[:5]
    headings = [line for line in failures.splitlines() if line.startswith("#### ")]
    assert [heading.split()[1] for heading in headings[:5]] == sorted(picked)
    assert headings[5:] == [
        "#### spider-train-00013 on flight_1 (medium)",
        "#### spider-train-00022 on flight_1 (hard)",
    ]
    assert failures.split(headings[5] + "\n", 1)[1].startswith(
        "\nQuestion: What are the names and distances for all airplanes?\n\n"
        "Gold query:\n\n```sql\nSELECT name ,  distance FROM Aircraft\n```\n\n"
        "Predicted query:\n\n```sql\nSELEC count(*) FROM Aircraft\n```\n\n"
        "Error: syntax. The query failed to execute.\n"
    )
    # A reply without SQL shows cut short, in a fence its own backticks can't close, with
    # the characters that don't print escaped.
    shown = REPLY[:600].replace("\x00", "\\x00").replace("\u2028", "\\u2028")
    assert f"The reply has no SQL query. Its start:\n\n````text\n{shown}\n````\n" in failures
    assert "\x00" not in text and "\u2028" not in text
    assert "Error: no_sql. There is no query to execute." in failures

    chart = written.chart.read_bytes()
    # No Software text chunk, so the bytes don't name matplotlib's version.
    assert _png_chunks(chart) == [b"IHDR", b"pHYs", b"IDAT", b"IEND"]
    assert struct.unpack(">II", chart[16:24]) == (640, 360)


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
    written = write_run_report("dev-run", runs_root=runs, data_dir=with_dev, now=AT)
    assert "\n| Split | dev: 10 questions, 1 sample each |\n" in written.report.read_text()
    # Reading the run's answers again spends no second look.
    assert [run_id for _, run_id, _ in read_uses(dev_log)] == ["dev-run"]


@pytest.mark.phase(2)
def test_model_text_cannot_break_the_markdown(
    runs: Path, spider_mini: Path, tmp_path: Path
) -> None:
    data = Path(shutil.copytree(spider_mini, tmp_path / "data"))
    items = json.loads((data / "train_spider.json").read_text())
    items[11]["question"] = "How many *aircrafts* are there\nin <b>the</b> | `table`?"
    (data / "train_spider.json").write_text(json.dumps(items))
    fenced = Outcome(True, False, False, "other", "SELECT '````' FROM Aircraft")
    outcomes = {VAL[0]: RIGHT, VAL[1]: fenced, VAL[2]: NO_SQL}
    _store(runs, "odd", outcomes, data_dir=data, model="m|`x`", reply="No idea.")
    text = write_run_report("odd", runs_root=runs, data_dir=data, now=AT).report.read_text()
    question = "How many \\*aircrafts\\* are there in \\<b\\>the\\</b\\> \\| \\`table\\`?"
    assert f"\nQuestion: {question}\n" in text
    assert "\n`````sql\nSELECT '````' FROM Aircraft\n`````\n" in text
    assert "\nThe reply has no SQL query:\n\n```text\nNo idea.\n```\n" in text
    assert "\n| Model | `` m\\|`x` `` |\n" in text


@pytest.mark.phase(2)
def test_chart_is_the_same_bytes_each_time(runs: Path) -> None:
    _store(runs, "mixed", MIXED)
    first = write_run_report("mixed", runs_root=runs, now=AT).chart.read_bytes()
    assert write_run_report("mixed", runs_root=runs, now=LATER).chart.read_bytes() == first
    _store(runs, "other", dict.fromkeys(VAL, RIGHT))
    assert write_run_report("other", runs_root=runs, now=AT).chart.read_bytes() != first
