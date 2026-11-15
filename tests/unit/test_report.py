"""Reports from runs' stored files.

Each test stores finished runs the way ``forge eval run`` does, over the fixture's val
questions (``spider-train-00010`` to ``00029`` on flight_1, five of each hardness).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
import pytest
from PIL import Image, ImageDraw, PngImagePlugin

import forge.eval.report as report_module
from forge.core.hashing import canonical_json
from forge.core.types import Generation, RunManifest, Score, Split
from forge.data.manifest import data_manifest_hash
from forge.data.spider import load_raw
from forge.data.splits import split_examples
from forge.eval.config import EvalConfig
from forge.eval.metrics import Estimate, Rate, compare, gold_clusters, summarize
from forge.eval.report import (
    Check,
    chart_renderer,
    load_run,
    rebuild,
    write_compare_report,
    write_run_report,
)
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
    first_tries: Mapping[str, Outcome] | None = None,
    retry: int | None = None,
) -> Path:
    """Store a run's manifest, generations and scores as the eval runner does.

    ``first_tries`` gives how the first reply went for each sample that was retried; the
    run's retry option is 1 when there are any, unless ``retry`` says otherwise.
    """
    examples = {e.id: e for e in split_examples(load_raw(data_dir))[split]}
    firsts = first_tries or {}
    options = EvalConfig(
        model=model,
        backend="fake",
        split=split,
        n=samples,
        temperature=0.7 if samples > 1 else 0,
        retry=(1 if firsts else 0) if retry is None else retry,
    )
    config: dict[str, Any] = json.loads(canonical_json(options.manifest_config()))

    def generation(example_id: str, sample: int, sql: str | None) -> Generation:
        return Generation(
            example_id=example_id,
            model_ref=model,
            sample_idx=sample,
            text=reply if sql is None else f"```sql\n{sql}\n```",
            sql=sql,
            tokens_in=50,
            tokens_out=12,
            latency_ms=3.0,
        )

    generations, scores, stored_firsts = [], [], []
    for example_id, outcome in outcomes.items():
        example = examples[example_id]
        sql = example.gold_sql if outcome.sql == GOLD else outcome.sql
        first = firsts.get(example_id)
        for sample in range(samples):
            generations.append(generation(example_id, sample, sql))
            score = Score(
                example_id=example_id,
                sample_idx=sample,
                hardness=example.hardness,
                valid=outcome.valid,
                ex_fast=outcome.fast,
                ex_official=outcome.official,
                error_class=outcome.error_class,
            )
            if first is not None:
                score = score.model_copy(update={"attempts": 2, "first_try_ex": first.official})
                stored_firsts.append(generation(example_id, sample, first.sql))
            scores.append(score)
    root = runs / run_id
    (root / "eval").mkdir(parents=True)
    files = [("generations", generations), ("scores", scores)]
    if stored_firsts:
        files.append(("first-tries", stored_firsts))
    for name, items in files:
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


def _official(root: Path) -> Rate | None:
    return summarize(_scores(root), clusters=_clusters()).ex_official


def _pct(share: float) -> str:
    return f"{100 * share:.1f}%"


def _ci(rate: Rate | None) -> str:
    assert rate is not None
    return f"{_pct(rate.value)} [{_pct(rate.low)}, {_pct(rate.high)}]"


def _signed(change: float) -> str:
    return f"{round(100 * change, 1) + 0.0:+.1f}"


def _change(change: Estimate) -> str:
    return f"{_signed(change.value)} [{_signed(change.low)}, {_signed(change.high)}]"


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


def _without_text(data: bytes) -> bytes:
    """The PNG ``data`` without its text chunks. Each chunk carries its own checksum, so
    what is left is still a PNG."""
    kept, at = [data[:8]], 8
    while at < len(data):
        (size,) = struct.unpack(">I", data[at : at + 4])
        if data[at + 4 : at + 8] != b"tEXt":
            kept.append(data[at : at + 12 + size])
        at += 12 + size
    return b"".join(kept)


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
        "| Platform | `local-cpu` |",
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
    # Its one text names what drew it, so rebuild knows whether it can compare the bytes.
    assert _png_chunks(chart) == [b"IHDR", b"tEXt", b"pHYs", b"IDAT", b"IEND"]
    with Image.open(io.BytesIO(chart)) as image:
        assert image.size == (640, 360)
        assert image.text == {"Software": chart_renderer()}
    assert chart_renderer().startswith(f"matplotlib {matplotlib.__version__} (FreeType ")
    assert chart_renderer().endswith(f"{sys.platform} {platform.machine()}")


@pytest.mark.phase(2)
def test_rebuild_matches_original(runs: Path) -> None:
    root = _store(runs, "mixed", MIXED)
    _store(runs, "base", dict.fromkeys(VAL, RIGHT))
    written = write_run_report("mixed", runs_root=runs, now=AT)
    report, chart = written.report, written.chart
    compared = write_compare_report("base", "mixed", runs_root=runs, now=AT)
    assert compared == root / "eval" / "compare-base.md"
    original, drawn = report.read_bytes(), chart.read_bytes()

    checks = rebuild("mixed", runs_root=runs)
    assert [(check.path, check.diff) for check in checks] == [
        (report, ()),
        (chart, ()),
        (compared, ()),
    ]
    # A rebuild writes nothing.
    assert (report.read_bytes(), chart.read_bytes()) == (original, drawn)

    # Written again later, the report differs from the first in its Generated line only.
    write_run_report("mixed", runs_root=runs, now=LATER)
    again = report.read_bytes()
    changed = [
        (old, new)
        for old, new in zip(original.split(b"\n"), again.split(b"\n"), strict=True)
        if old != new
    ]
    assert changed == [
        (b"Generated: 2026-11-15 09:30:00 UTC", b"Generated: 2026-12-01 18:45:12 UTC")
    ]
    assert [check.diff for check in rebuild("mixed", runs_root=runs)] == [(), (), ()]

    # A number edited in the stored report is a difference.
    report.write_bytes(again.replace(b"| no_sql | 6 | 54.5% |", b"| no_sql | 5 | 54.5% |"))
    run_check, chart_check, compare_check = rebuild("mixed", runs_root=runs)
    assert [line for line in run_check.diff if line.startswith(("-|", "+|"))] == [
        "-| no_sql | 5 | 54.5% |",
        "+| no_sql | 6 | 54.5% |",
    ]
    assert not chart_check.diff and not compare_check.diff

    # Only the time the Generated line holds is masked, so a number typed after it is a
    # difference too.
    stamp = b"Generated: 2026-12-01 18:45:12 UTC"
    claim = stamp + b". Official EX on dev is 87.5% [80.1%, 92.3%]"
    report.write_bytes(again.replace(stamp, claim))
    run_check, chart_check, compare_check = rebuild("mixed", runs_root=runs)
    assert [line for line in run_check.diff if line.startswith(("-Gen", "+Gen"))] == [
        f"-{claim.decode()}",
        "+Generated: (any time)",
    ]
    assert not chart_check.diff and not compare_check.diff
    # So is a second Generated line: only the first is the time of writing.
    report.write_bytes(again.replace(b"\n## By hardness\n", b"\n" + stamp + b"\n## By hardness\n"))
    run_check = rebuild("mixed", runs_root=runs)[0]
    assert [line for line in run_check.diff if line.startswith(("-Gen", "+Gen"))] == [
        f"-{stamp.decode()}"
    ]
    report.write_bytes(again)

    # The chart is compared byte for byte: one drawn for another run's numbers differs.
    _store(runs, "other", dict.fromkeys(VAL, NO_SQL))
    chart.write_bytes(write_run_report("other", runs_root=runs, now=AT).chart.read_bytes())
    run_check, chart_check, compare_check = rebuild("mixed", runs_root=runs)
    assert not run_check.diff and not compare_check.diff
    assert chart_check.path == chart
    other = chart.read_bytes()
    assert chart_check.diff == (
        f"--- {chart}",
        "+++ rebuilt from the stored files",
        f"stored: {len(other):,} bytes, sha256 {hashlib.sha256(other).hexdigest()}",
        f"rebuilt: {len(drawn):,} bytes, sha256 {hashlib.sha256(drawn).hexdigest()}",
    )
    # A missing chart fails the rebuild.
    chart.unlink()
    with pytest.raises(FileNotFoundError, match=r"report shows .*val-hardness\.png, which is"):
        rebuild("mixed", runs_root=runs)
    chart.write_bytes(drawn)

    # So is a score changed after the reports were written: a right answer that official
    # EX now rejects shows in all three.
    _edit_first_score(root, ex_official=False)
    run_check, chart_check, compare_check = rebuild("mixed", runs_root=runs)
    assert "-| official EX | 9 | 20 |" in "\n".join(run_check.diff)
    assert "+| official EX | 8 | 20 |" in "\n".join(run_check.diff)
    assert "+| unclassified | 2 | 16.7% |" in run_check.diff
    assert chart_check.diff
    assert "+| run a right | 8 | 12 |" in compare_check.diff


@pytest.mark.phase(2)
def test_compare_report_has_mcnemar(runs: Path, tmp_path: Path) -> None:
    # Official EX in run a and run b: 8 right in both, 1 only in a, 9 only in b, and 2
    # in neither. Run a fails with 1 syntax error and 10 wrong filters, run b on values.
    _store(runs, "base", dict(zip(VAL, [RIGHT] * 9 + [SYNTAX] + [WHERE] * 10, strict=True)))
    tuned = [RIGHT] * 8 + [VALUE] + [RIGHT] * 9 + [VALUE] * 2
    root = _store(runs, "tuned", dict(zip(VAL, tuned, strict=True)), model="tuned")
    path = write_compare_report("base", "tuned", runs_root=runs, now=AT)
    assert path == root / "eval" / "compare-base.md"
    text = path.read_text()

    comparison = compare(_scores(runs / "base"), _scores(root), clusters=_clusters())
    assert comparison.metric == "ex_official"
    assert comparison.delta.value == pytest.approx(0.4)
    # The test pairs the 12 clusters, not the 20 questions: run a gets more right only
    # where 00018, which b gets wrong, shares 00017's gold query; run b in the 6 clusters
    # from 00019 to 00027; the other 5 tie. The exact binomial test of 1 against 6 is
    # 2 * (1 + 7) / 2**7, where the 10 discordant questions would give 0.0215.
    assert (comparison.mcnemar.wins_a, comparison.mcnemar.wins_b) == (1, 6)
    assert comparison.mcnemar.pvalue == pytest.approx(0.125)
    delta = comparison.delta
    assert text.startswith(
        "# Official EX on val, run b against run a: +40.0 points "
        f"[{_signed(delta.low)}, {_signed(delta.high)}], McNemar p = 0.1250 on clusters\n"
    )
    assert "\n| Run | `base` | `tuned` |\n| Model | `gold` | `tuned` |\n" in text
    links = "[manifest.json](../../base/manifest.json) | [manifest.json](../manifest.json)"
    assert f"| Manifest | {links} |" in text
    assert "checked against them by `forge report rebuild tuned`" in text

    mcnemar = _section(text, "McNemar's test")
    assert "| run a right | 8 | 1 |\n| run a wrong | 9 | 2 |" in mcnemar
    assert "disagree on 10 of 20 questions: run b alone is right on 9, run a alone on 1." in (
        mcnemar
    )
    assert (
        "\nThe test pairs whole clusters, as the intervals draw them: run b gets more right in "
        "6 clusters, run a in 1, and the rest tie.\n"
        "Exact two-sided McNemar test on those clusters: p = 0.1250.\n"
    ) in mcnemar

    accuracy = _section(text, "Execution accuracy")
    rates = f"{_ci(_official(runs / 'base'))} | {_ci(_official(root))}"
    assert f"| official EX | {rates} | {_change(delta)} |" in accuracy

    hardness = _section(text, "By hardness")
    for level in ("easy", "medium", "hard", "extra"):
        on_level = compare(
            [score for score in _scores(runs / "base") if score.hardness == level],
            [score for score in _scores(root) if score.hardness == level],
            clusters=_clusters(),
        )
        row = f"| {level} | 5 | {_pct(on_level.ex_a)} | {_pct(on_level.ex_b)} |"
        assert f"{row} {_change(on_level.delta)} |" in hardness

    classes = _section(text, "By error class")
    for row in (
        "| no_sql | 0 | 0 | +0 |",
        "| syntax | 1 | 0 | -1 |",
        "| where | 10 | 0 | -10 |",
        "| value | 0 | 3 | +3 |",
    ):
        assert f"\n{row}\n" in classes
    assert "unclassified" not in classes

    # Written elsewhere, its links work from there, and it doesn't claim rebuild checks it.
    out = tmp_path / "docs" / "base-vs-tuned.md"
    assert write_compare_report("base", "tuned", out=out, runs_root=runs, now=AT) == out
    elsewhere = out.read_text()
    links = (
        "[manifest.json](../runs/base/manifest.json) | [manifest.json](../runs/tuned/manifest.json)"
    )
    assert f"| Manifest | {links} |" in elsewhere
    assert "forge report rebuild" not in elsewhere
    assert _section(elsewhere, "McNemar's test") == mcnemar


@pytest.mark.phase(2)
def test_compare_report_out_can_name_its_default_path(runs: Path, tmp_path: Path) -> None:
    _store(runs, "base", dict.fromkeys(VAL, RIGHT))
    root = _store(runs, "tuned", MIXED, model="tuned")
    write_run_report("tuned", runs_root=runs, now=AT)
    default = write_compare_report("base", "tuned", runs_root=runs, now=AT)
    text = default.read_text()
    assert "checked against them by `forge report rebuild tuned`" in text
    # Through '..' or a symlink, out still names the file rebuild checks, so it is written
    # as the default is, and its rebuild matches.
    link = tmp_path / "linked-runs"
    link.symlink_to(runs, target_is_directory=True)
    for out in (
        runs / "base" / ".." / "tuned" / "eval" / "compare-base.md",
        link / "tuned" / "eval" / "compare-base.md",
    ):
        default.unlink()
        assert write_compare_report("base", "tuned", out=out, runs_root=runs, now=AT) == out
        assert default.read_text() == text
        assert [(c.path, c.diff) for c in rebuild("tuned", runs_root=runs)][2:] == [(default, ())]
    assert root / "eval" / "compare-base.md" == default


@pytest.mark.phase(2)
def test_compare_with_several_samples_has_no_pvalue(runs: Path) -> None:
    _store(runs, "base", dict.fromkeys(VAL, NO_SQL), samples=2)
    _store(runs, "tuned", dict.fromkeys(VAL, RIGHT), samples=2)
    text = write_compare_report("base", "tuned", runs_root=runs, now=AT).read_text()
    assert text.startswith(
        "# Official EX on val, run b against run a: +100.0 points [+100.0, +100.0]\n"
    )
    assert "| Generation | sampled at temperature 0.7, top-p 1, seed 0, at most 256" in text
    assert "Both runs answered the same 20 questions of val, 2 samples each" in text
    assert "| run a right | 0 | 0 |\n| run a wrong | 40 | 0 |" in text
    assert "The runs disagree on 40 of 40 samples" in text
    assert "There is no p-value: each question has several samples" in text


@pytest.mark.phase(2)
def test_compare_needs_the_same_samples(runs: Path) -> None:
    _store(runs, "base", dict.fromkeys(VAL, RIGHT))
    _store(runs, "fewer", dict.fromkeys(VAL[:10], RIGHT))
    _store(runs, "train", {"spider-train-00000": RIGHT}, split="train")
    with pytest.raises(ValueError, match="scored different samples, so they can't be paired"):
        write_compare_report("base", "fewer", runs_root=runs)
    with pytest.raises(ValueError, match="run base answered val and run train train"):
        write_compare_report("base", "train", runs_root=runs)
    assert not list((runs / "fewer" / "eval").glob("compare-*"))


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


# What a matplotlibrc could change in the chart, were it read.
MATPLOTLIBRC = """\
font.family: monospace
font.size: 14
text.antialiased: False
figure.dpi: 300
savefig.dpi: 50
savefig.transparent: True
"""
# A fresh interpreter's rendering of a stored run's chart: what it read from the
# matplotlibrc, then the chart's hash.
RENDER = """\
import hashlib, sys
from pathlib import Path
import matplotlib
from forge.eval.report import load_run, render_chart
chart = render_chart(load_run(sys.argv[1], runs_root=Path(sys.argv[2])))
print(matplotlib.rcParams["font.family"], hashlib.sha256(chart).hexdigest())
"""


@pytest.mark.phase(2)
def test_chart_is_the_same_bytes_each_time(runs: Path, tmp_path: Path) -> None:
    _store(runs, "mixed", MIXED)
    first = write_run_report("mixed", runs_root=runs, now=AT).chart.read_bytes()
    assert write_run_report("mixed", runs_root=runs, now=LATER).chart.read_bytes() == first
    _store(runs, "other", dict.fromkeys(VAL, RIGHT))
    assert write_run_report("other", runs_root=runs, now=AT).chart.read_bytes() != first
    # Another process draws the same bytes, with other hash seeds and a matplotlibrc that
    # changes matplotlib's defaults, as when rebuild runs where the report was written.
    # Only on one machine: CI's Linux draws other bytes than macOS with the same releases,
    # so a chart names its renderer, and rebuild compares it only where that is the same.
    rc = tmp_path / "matplotlibrc"
    rc.write_text(MATPLOTLIBRC)
    outputs = {
        subprocess.run(
            [sys.executable, "-c", RENDER, "mixed", str(runs)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed, "MATPLOTLIBRC": str(rc)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("1", "2")
    }
    assert outputs == {f"['monospace'] {hashlib.sha256(first).hexdigest()}\n"}


@pytest.mark.phase(2)
def test_rebuild_skips_a_chart_drawn_elsewhere(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store(runs, "mixed", MIXED)
    chart = write_run_report("mixed", runs_root=runs, now=AT).chart
    drawn, here = chart.read_bytes(), chart_renderer()
    # Where another renderer draws, the chart's bytes can't be compared, and the check
    # says why instead of failing.
    with monkeypatch.context() as patched:
        patched.setattr(report_module, "chart_renderer", lambda: "another renderer")
        chart_check = rebuild("mixed", runs_root=runs)[1]
    assert chart_check == Check(
        chart, (), skipped=f"it was drawn with {here}, and this machine draws with another renderer"
    )
    # Back here, it is compared again.
    assert rebuild("mixed", runs_root=runs)[1] == Check(chart, ())

    # A chart that names no renderer, like those drawn before charts named theirs, is
    # compared, and differs from its rebuild. So does one whose Software text isn't a
    # renderer's, as when an image editor saved a change to it.
    bare = _without_text(drawn)
    with Image.open(io.BytesIO(bare)) as image:
        assert image.text == {}
        edited = image.copy()
    ImageDraw.Draw(edited).rectangle((100, 100, 300, 250), fill="red")
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "GIMP 2.10.38")
    saved = io.BytesIO()
    edited.save(saved, format="PNG", pnginfo=info)
    for stored in (bare, saved.getvalue()):
        chart.write_bytes(stored)
        assert rebuild("mixed", runs_root=runs)[1] == Check(
            chart,
            (
                f"--- {chart}",
                "+++ rebuilt from the stored files",
                f"stored: {len(stored):,} bytes, sha256 {hashlib.sha256(stored).hexdigest()}",
                f"rebuilt: {len(drawn):,} bytes, sha256 {hashlib.sha256(drawn).hexdigest()}",
            ),
        )
    with Image.open(io.BytesIO(saved.getvalue())) as image:
        assert image.text == {"Software": "GIMP 2.10.38"}


# SQLite can't parse it, but official EX drops DISTINCT first and runs count(*).
FIRST_DISTINCT = Outcome(False, False, True, "syntax", "SELECT count(DISTINCT *) FROM Aircraft")
RETRY = "once, with the error, for a query that fails to run"


@pytest.mark.phase(2)
def test_report_shows_ex_with_and_without_retry(runs: Path) -> None:
    # Four samples failed to run and were asked again: two were fixed, one failed again,
    # and one lost the official match its first query had. 00015 ran and wasn't retried.
    final = {**dict.fromkeys(VAL, RIGHT), VAL[0]: VALUE, VAL[3]: SYNTAX, VAL[5]: WHERE}
    firsts = {VAL[0]: FIRST_DISTINCT, VAL[1]: SYNTAX, VAL[2]: SCHEMA, VAL[3]: NO_SQL}
    root = _store(runs, "retried", final, first_tries=firsts)
    text = write_run_report("retried", runs_root=runs, now=AT).report.read_text()

    with_retry = _scores(root)
    without = [
        Score(
            example_id=s.example_id,
            hardness=s.hardness,
            valid=False,
            ex_fast=False,
            ex_official=firsts[s.example_id].official,
        )
        if s.example_id in firsts
        else s
        for s in with_retry
    ]
    before, after = (
        summarize(without, clusters=_clusters()),
        summarize(with_retry, clusters=_clusters()),
    )
    assert before.ex_official is not None and after.ex_official is not None
    assert (before.ex_official.hits, after.ex_official.hits) == (16, 17)
    change = compare(without, with_retry, clusters=_clusters()).delta
    assert change.value == pytest.approx(0.05)

    assert f"\n| Retry | {RETRY} |\n" in text
    retry = _section(text, "Retry")
    assert f"\n| valid SQL | {_ci(before.valid)} | {_ci(after.valid)} | |\n" in retry
    assert f"\n| fast EX | {_ci(before.ex_fast)} | {_ci(after.ex_fast)} | |\n" in retry
    official = f"| official EX | {_ci(before.ex_official)} | {_ci(after.ex_official)} |"
    assert f"\n{official} {_change(change)} |\n" in retry
    for row in (
        "| retried: the first query failed to run | 4 |",
        "| the second query ran | 3 |",
        "| right on retry, by official EX | 2 |",
        "| right on the first reply, by official EX | 1 |",
    ):
        assert f"\n{row}\n" in retry
    # The rest of the report is about the replies scored, the second ones where retried.
    assert f"| official EX | 17 | 20 | {_ci(after.ex_official)} |" in text

    failures = _section(text, "Example failures")
    assert (
        "Gold query:\n\n```sql\nSELECT name ,  distance FROM Aircraft\n```\n\n"
        "The first reply has no SQL query.\n\n"
        "Predicted query on retry:\n\n```sql\nSELEC count(*) FROM Aircraft\n```\n"
    ) in failures
    assert (
        "The first reply's query failed to run:\n\n"
        "```sql\nSELECT count(DISTINCT *) FROM Aircraft\n```\n\n"
        "Predicted query on retry:\n\n```sql\nSELECT count(*) FROM Aircraft WHERE aid = 3\n```\n"
    ) in failures
    # 00015 ran, so it has one reply.
    assert (
        "Predicted query:\n\n```sql\nSELECT aid FROM Aircraft WHERE distance > 9\n```" in failures
    )

    # A run that retries nothing has no section, and one without retry says so.
    _store(runs, "base", dict.fromkeys(VAL, RIGHT))
    _store(runs, "calm", dict.fromkeys(VAL, RIGHT), retry=1)
    calm = write_run_report("calm", runs_root=runs, now=AT).report.read_text()
    assert f"\n| Retry | {RETRY} |\n" in calm and "\n## Retry\n" not in calm
    base = write_run_report("base", runs_root=runs, now=AT).report.read_text()
    assert "\n| Retry | none |\n" in base and "\n## Retry\n" not in base

    # A compare report pairs the runs without retry too.
    compared = write_compare_report("base", "retried", runs_root=runs, now=AT).read_text()
    assert f"\n| Retry | none | {RETRY} |\n" in compared
    base_scores = _scores(runs / "base")
    first_change = compare(base_scores, without, clusters=_clusters()).delta
    rates = f"{_ci(_official(runs / 'base'))} | {_ci(before.ex_official)}"
    assert f"\n| official EX without retry | {rates} | {_change(first_change)} |\n" in compared
    assert "gets a change, with retry and without it: each sample of run b" in compared
    plain = write_compare_report("base", "calm", runs_root=runs, now=AT).read_text()
    assert "without retry" not in plain
    assert [check.diff for check in rebuild("retried", runs_root=runs)] == [(), (), ()]


@pytest.mark.phase(2)
def test_report_refuses_retries_the_run_cant_have(runs: Path) -> None:
    firsts = {VAL[0]: SYNTAX}
    _store(runs, "no-retry", dict.fromkeys(VAL, RIGHT), first_tries=firsts, retry=0)
    with pytest.raises(RunCorrupt, match="took 2 replies, but the run never retries"):
        load_run("no-retry", runs_root=runs)
    unjudged = _store(runs, "unjudged", dict.fromkeys(VAL, RIGHT), first_tries=firsts)
    _edit_first_score(unjudged, first_try_ex=None)
    with pytest.raises(RunCorrupt, match="00010 sample 0 has no official verdict on its first try"):
        load_run("unjudged", runs_root=runs)
    stray = _store(runs, "stray", dict.fromkeys(VAL, RIGHT), first_tries={VAL[1]: SYNTAX})
    _edit_first_score(stray, first_try_ex=False)
    with pytest.raises(RunCorrupt, match="00010 sample 0 has a first try's verdict, but took one"):
        load_run("stray", runs_root=runs)
