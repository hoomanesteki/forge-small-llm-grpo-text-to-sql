"""Reports (stage 10): what a run scored, and how two runs differ, from stored files only.

``write_run_report`` writes ``eval/<split>-report.md`` in a run's folder, with the chart
``eval/<split>-hardness.png`` beside it. ``write_compare_report`` writes
``eval/compare-<a>.md`` in run b's folder, or where ``out`` says. Nothing in a report
comes from a run's memory: ``load_run`` reads

- the run's manifest, scores and generations, through ``read_run``, which takes only a
  finished run whose files agree with each other and with the manifest;
- the questions and gold queries of its split, from data whose data manifest hash is
  the manifest's. Other data is refused.

Rates are official EX unless labeled fast (ADR-0008), so every sample needs an official
verdict. Intervals come from ``forge.eval.metrics``, with the questions that share a gold
query drawn together (``gold_clusters``), and McNemar's test in a compare report pairs
those clusters, under the question table. Error classes count the failures of official
EX. A run classifies the failures of fast EX (``forge.eval.scoring``), so a failure that
only official EX finds has no class and counts as ``unclassified``. The report shows
``SHOWN_FAILURES`` failures of each of the ``TOP_CLASSES`` most common classes, picked by
a hash of their ids so they spread over databases and every rendering picks the same
ones. The executor's error message isn't stored, so a failure shows its class.

A run that retried failed queries (``forge.eval.retry``) shows EX with and without
retry, the change between them with its paired interval, and how the retried samples
did. Without retry, each sample keeps its first reply (``without_retry``), so official
EX judges the first reply's query, which is stored as its first try (ADR-0015). A
compare report adds official EX without retry when either run retried, and each failure
shown that was retried shows both of its queries.

A dev run's report shows the model's answers on dev, which belongs to the look the run
logged (ADR-0004): it needs the run's row in ``docs/dev-usage.md`` and adds none.

One line changes between renderings: ``Generated:`` and the time. ``rebuild`` renders a
run's report, and each compare report in its folder, again from the stored files and
compares each with the stored one, that line aside. The chart draws the by-hardness
table; its bytes are the same on every rendering on one machine, so a committed chart
changes only with its numbers. ``rebuild`` doesn't compare it: its pixels can change with
the matplotlib version and aren't known to match between machines, while the table holds
every number it draws.

Each kind of number has one format, so FG-207's I4 check can find every number in the
stored files: a rate as a percentage with one decimal and its 95% interval in brackets
(``45.2% [41.8%, 48.6%]``), a change in percentage points with its sign and interval
(``+5.2 [+1.0, +9.4]``), counts with thousands separators, and a p-value to four
decimals (``< 0.0001`` below that). Constants (the interval level, resamples and seed,
how many failures are shown, and Spider's citation) appear in prose lines only.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from forge.core.types import Example, Generation, GenParams, Hardness, Score, Split
from forge.data.manifest import data_manifest_hash
from forge.data.spider import load_raw, resolve_data_dir
from forge.data.splits import split_examples
from forge.eval.config import check_run_id
from forge.eval.metrics import (
    PERCENTILES,
    RESAMPLES,
    SEED,
    Estimate,
    McNemar,
    Rate,
    Summary,
    compare,
    gold_clusters,
    summarize,
)
from forge.eval.retry import without_retry
from forge.eval.run_files import (
    RunCorrupt,
    StoredRun,
    hold,
    key,
    read_run,
    write_atomic,
)
from forge.eval.taxonomy import ERROR_CLASSES
from forge.ops.dev_usage import read_uses
from forge.ops.paths import dev_usage_path, runs_dir

SHOWN_FAILURES = 5
TOP_CLASSES = 3
# Replies and predicted queries are cut at this many characters: a random model's reply
# can fill pages.
SHOWN_CHARS = 600
UNCLASSIFIED = "unclassified"
GENERATED = "Generated: "
COMPARE_PREFIX = "compare-"

LEVEL = f"{PERCENTILES[1] - PERCENTILES[0]:g}%"
SPIDER = (
    "Questions and gold queries are quoted from Spider 1.0 (Yu et al., EMNLP 2018), "
    "licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)."
)
UNCLASSIFIED_NOTE = (
    "Unclassified failures are those only official EX finds: fast EX matched them, so "
    "the run gave them no class."
)

_ORDER = (*ERROR_CLASSES, UNCLASSIFIED)
_GENERATED_LINE = re.compile(rf"^{GENERATED}.*$", re.MULTILINE)
# What markdown could read as markup in a question, escaped where it shows as text.
_MARKUP = re.compile(r"([\\`*_\[\]<>|~&])")
_BACKTICKS = re.compile(r"`+")

# One series on the light chart surface: text in ink, a hairline grid behind the bars.
_SURFACE = "#fcfcfb"
_BAR = "#2a78d6"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_CHART_INCHES = (6.4, 3.6)
_CHART_DPI = 100
_AXES = (0.09, 0.17, 0.88, 0.6)  # left, bottom, width and height, in figure fractions
_BAR_PX = 24  # bars stay thin, however few levels share the axis


class RunOptions(BaseModel, frozen=True):
    """What a report shows of a run's options, from its manifest's config."""

    backend: str
    split: Split
    limit: int | None
    prompt_version: str
    generation: GenParams
    retry: Literal[0, 1]


@dataclass(frozen=True)
class ReportRun:
    """A finished run with what a report adds to its files."""

    stored: StoredRun
    options: RunOptions
    examples: Mapping[str, Example]  # the questions it scored, by id
    clusters: Mapping[str, str]  # gold_clusters of those questions
    summary: Summary
    official: Rate  # summary.ex_official, which every report leads with

    @property
    def run_id(self) -> str:
        return self.stored.manifest.run_id


@dataclass(frozen=True)
class RunReport:
    """Where ``write_run_report`` put the report and its chart."""

    report: Path
    chart: Path


@dataclass(frozen=True)
class Check:
    """A stored report against its rebuild: their unified diff, empty when they match."""

    path: Path
    diff: tuple[str, ...]


@dataclass(frozen=True)
class _Data:
    """The data reports read questions from, and the hash a run must have been made on."""

    root: Path
    hash: str
    splits: Mapping[Split, Sequence[Example]]


def load_run(
    run_id: str, *, runs_root: Path | None = None, data_dir: Path | None = None
) -> ReportRun:
    """The finished run ``run_id`` with the questions it answered, as a report reads it.

    Raises what ``read_run`` raises, ``ValueError`` for a run made on other data or a dev
    run whose look isn't logged, and ``RunCorrupt`` for scores a finished run can't hold.
    """
    return _load(run_id, _data(data_dir), runs_root or runs_dir())


def write_run_report(
    run_id: str,
    *,
    runs_root: Path | None = None,
    data_dir: Path | None = None,
    now: datetime | None = None,
) -> RunReport:
    """Write the run's report and chart into its ``eval`` folder, from its stored files."""
    run = load_run(run_id, runs_root=runs_root, data_dir=data_dir)
    paths = run.stored.paths
    text = render_run(run, generated=now or datetime.now(UTC))
    chart = render_chart(run)
    with hold(paths):
        write_atomic(paths.chart, chart)
        write_atomic(paths.report, text.encode())
    return RunReport(paths.report, paths.chart)


def write_compare_report(
    a: str,
    b: str,
    *,
    out: Path | None = None,
    runs_root: Path | None = None,
    data_dir: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Write run b against run a to ``out``, by default ``eval/compare-<a>.md`` in b's folder.

    Raises ``ValueError`` unless both runs answered the same samples of one split.
    """
    data, root = _data(data_dir), runs_root or runs_dir()
    run_a, run_b = _load(a, data, root), _load(b, data, root)
    path = out or compare_path(run_b, a)
    text = render_compare(run_a, run_b, out=path, generated=now or datetime.now(UTC))
    # Run b's lock keeps a second writer of its reports from sharing the temporary file.
    with hold(run_b.stored.paths):
        write_atomic(path, text.encode())
    return path


def rebuild(
    run_id: str, *, runs_root: Path | None = None, data_dir: Path | None = None
) -> list[Check]:
    """Render the run's report and each ``compare-<a>.md`` in its ``eval`` folder again,
    and compare each with the stored file, its ``Generated:`` line aside. Writes nothing.
    """
    data, root = _data(data_dir), runs_root or runs_dir()
    run = _load(run_id, data, root)
    report = run.stored.paths.report
    if not report.is_file():
        raise FileNotFoundError(
            f"run {run_id} has no report to rebuild: write it with `forge report run {run_id}`"
        )
    now = datetime.now(UTC)
    checks = [_check(report, render_run(run, generated=now))]
    for path in sorted(report.parent.glob(f"{COMPARE_PREFIX}*.md")):
        a = path.name.removeprefix(COMPARE_PREFIX).removesuffix(".md")
        checks.append(
            _check(path, render_compare(_load(a, data, root), run, out=path, generated=now))
        )
    return checks


def compare_path(b: ReportRun, a: str) -> Path:
    return b.stored.paths.report.parent / f"{COMPARE_PREFIX}{a}.md"


def render_run(run: ReportRun, *, generated: datetime) -> str:
    """The run's report as markdown."""
    summary = run.summary
    lines = [
        f"# Run {run.run_id}: official EX {_rate(run.official)} on {run.options.split}",
        "",
        _generated(generated),
        "",
        f"Written by `forge report run {run.run_id}` from the run's stored files and checked",
        f"against them by `forge report rebuild {run.run_id}`; don't edit it by hand.",
        "",
        SPIDER,
        "",
        "| | |",
        "|---|---|",
        *(f"| {name} | {value} |" for name, value in _header(run, run.stored.paths.report)),
        "",
        "## Execution accuracy",
        "",
        f"| | right | of | rate [{LEVEL} interval] |",
        "|---|---:|---:|---:|",
        f"| valid SQL | {summary.valid.hits:,} | {summary.n:,} | {_rate(summary.valid)} |",
        f"| fast EX | {summary.ex_fast.hits:,} | {summary.n:,} | {_rate(summary.ex_fast)} |",
        f"| official EX | {run.official.hits:,} | {summary.n:,} | {_rate(run.official)} |",
        "",
        *_checker_disagreement(run.stored.scores),
        _method(run),
        "",
        *_retry_section(run),
        "## By hardness",
        "",
        f"| hardness | questions | fast EX [{LEVEL} interval] | official EX [{LEVEL} interval] |",
        "|---|---:|---:|---:|",
    ]
    for level, ex in summary.by_hardness.items():
        official = _rate(_official(ex.ex_official))
        lines.append(f"| {level} | {ex.questions:,} | {_rate(ex.ex_fast)} | {official} |")
    lines += [
        "",
        f"![Official EX by hardness, with {LEVEL} intervals]({run.stored.paths.chart.name})",
        "",
        "## Error classes",
        "",
        *_classes(run),
        "",
        "## Example failures",
        *_examples(run),
    ]
    return "\n".join(lines) + "\n"


def render_compare(a: ReportRun, b: ReportRun, *, out: Path, generated: datetime) -> str:
    """Run b against run a as markdown, with links that work from ``out``.

    Raises ``ValueError`` unless both runs answered the same samples of one split.
    """
    split = a.options.split
    if b.options.split != split:
        raise ValueError(
            f"run {a.run_id} answered {split} and run {b.run_id} {b.options.split}; "
            "compare runs on one split"
        )
    overall = compare(a.stored.scores, b.stored.scores, clusters=a.clusters)
    levels = {
        level: compare(
            _at(a.stored.scores, level), _at(b.stored.scores, level), clusters=a.clusters
        )
        for level in a.summary.by_hardness
    }
    table = overall.mcnemar
    unit = "question" if overall.n == overall.questions else "sample"
    delta = overall.delta
    title = (
        f"# Official EX on {split}, run b against run a: {_points(delta.value)} points "
        f"[{_points(delta.low)}, {_points(delta.high)}]"
    )
    if table.pvalue is not None:
        title += f", McNemar {_p(table.pvalue)} on {table.unit}s"
    written = (
        f"Written by `forge report compare {a.run_id} {b.run_id}` from both runs' stored files"
    )
    if out.absolute() == compare_path(b, a.run_id).absolute():
        written += f" and checked against them by `forge report rebuild {b.run_id}`"
    samples = b.options.generation.n
    retried = _retried(a) or _retried(b)
    changed = "Only official EX, the reported rate, gets a change"
    if retried:
        changed += ", with retry and without it"
    lines = [
        title,
        "",
        _generated(generated),
        "",
        f"{written}; don't edit it by hand.",
        "",
        "| | run a | run b |",
        "|---|---|---|",
        f"| Run | `{a.run_id}` | `{b.run_id}` |",
        *(
            f"| {name} | {value_a} | {value_b} |"
            for (name, value_a), (_, value_b) in zip(
                _header(a, out, shared=False), _header(b, out, shared=False), strict=True
            )
        ),
        "",
        f"Both runs answered the same {_count(overall.questions, 'question')} of {split}, "
        f"{_count(samples, 'sample')} each, on data with manifest hash "
        f"`{b.stored.manifest.data_manifest_hash}`.",
        "",
        "## Execution accuracy",
        "",
        f"| | run a [{LEVEL} interval] | run b [{LEVEL} interval] "
        f"| change in points [{LEVEL} interval] |",
        "|---|---:|---:|---:|",
        f"| valid SQL | {_rate(a.summary.valid)} | {_rate(b.summary.valid)} | |",
        f"| fast EX | {_rate(a.summary.ex_fast)} | {_rate(b.summary.ex_fast)} | |",
        f"| official EX | {_rate(a.official)} | {_rate(b.official)} | {_change(overall.delta)} |",
        *([_without_retry_row(a, b)] if retried else []),
        "",
        f"{changed}: each sample of run b against",
        f"the same one of run a. {_method(a)}",
        "",
        "## McNemar's test",
        "",
        "| official EX | run b right | run b wrong |",
        "|---|---:|---:|",
        f"| run a right | {table.both:,} | {table.only_a:,} |",
        f"| run a wrong | {table.only_b:,} | {table.neither:,} |",
        "",
        f"The runs disagree on {table.only_a + table.only_b:,} of {_count(overall.n, unit)}: "
        f"run b alone is right on {table.only_b:,}, run a alone on {table.only_a:,}.",
        *_mcnemar(table),
        "",
        "## By hardness",
        "",
        f"| hardness | questions | run a | run b | change in points [{LEVEL} interval] |",
        "|---|---:|---:|---:|---:|",
    ]
    for level, change in levels.items():
        lines.append(
            f"| {level} | {change.questions:,} | {_pct(change.ex_a)} | {_pct(change.ex_b)} "
            f"| {_change(change.delta)} |"
        )
    failures_a, failures_b = _failures(a.stored.scores), _failures(b.stored.scores)
    lines += [
        "",
        "## By error class",
        "",
        "Failures of official EX in each run, by class.",
        "",
        "| class | run a | run b | change |",
        "|---|---:|---:|---:|",
    ]
    for name in _class_rows(failures_a, failures_b):
        before, after = failures_a[name], failures_b[name]
        lines.append(f"| {name} | {before:,} | {after:,} | {after - before:+,} |")
    if failures_a[UNCLASSIFIED] or failures_b[UNCLASSIFIED]:
        lines += ["", UNCLASSIFIED_NOTE]
    return "\n".join(lines) + "\n"


def render_chart(run: ReportRun) -> bytes:
    """Official EX at each hardness level with its interval, as PNG bytes that are the same
    on every rendering on one machine."""
    # matplotlib takes about half a second to import, and only the chart needs it.
    import matplotlib.style
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    levels = list(run.summary.by_hardness.items())
    rates = [_official(ex.ex_official) for _, ex in levels]
    values = [100 * rate.value for rate in rates]
    below = [value - 100 * rate.low for value, rate in zip(values, rates, strict=True)]
    above = [100 * rate.high - value for value, rate in zip(values, rates, strict=True)]
    slots = range(len(levels))
    width = _BAR_PX / (_AXES[2] * _CHART_INCHES[0] * _CHART_DPI / len(levels))
    # matplotlib's defaults rather than any matplotlibrc's, so the bytes follow the numbers.
    with matplotlib.style.context("default"):
        figure = Figure(figsize=_CHART_INCHES, dpi=_CHART_DPI, facecolor=_SURFACE)
        FigureCanvasAgg(figure)
        axes = figure.add_axes(_AXES, facecolor=_SURFACE)
        axes.bar(slots, values, width=width, color=_BAR, zorder=2)
        axes.errorbar(
            slots,
            values,
            yerr=[below, above],
            fmt="none",
            ecolor=_INK_SECONDARY,
            elinewidth=1,
            capsize=4,
            capthick=1,
            zorder=3,
        )
        # Each value beside its bar's top, clear of the whisker through it.
        for slot, value in zip(slots, values, strict=True):
            axes.annotate(
                _pct(value / 100),
                (slot + width / 2, value),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=_INK,
            )
        axes.set_xticks(
            list(slots), [f"{level}\n{_count(ex.questions, 'question')}" for level, ex in levels]
        )
        axes.set_xlim(-0.5, len(levels) - 0.5)
        axes.set_ylim(0, 105)
        axes.set_yticks([0, 25, 50, 75, 100])
        axes.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        axes.grid(axis="y", color=_GRID, linewidth=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.spines["bottom"].set_color(_BASELINE)
        axes.tick_params(length=0, labelcolor=_INK_SECONDARY, labelsize=9)
        figure.text(0.09, 0.95, "Official EX by hardness", color=_INK, fontsize=11, va="top")
        figure.text(
            0.09,
            0.88,
            f"{run.options.split}, run {run.run_id}; whiskers are {LEVEL} intervals",
            color=_INK_SECONDARY,
            fontsize=9,
            va="top",
        )
        buffer = io.BytesIO()
        # Without the Software entry the file doesn't name matplotlib's version, so its
        # bytes change only with the drawing.
        figure.savefig(buffer, format="png", dpi=_CHART_DPI, metadata={"Software": None})
    return buffer.getvalue()


def _data(data_dir: Path | None) -> _Data:
    root = resolve_data_dir(data_dir).resolve()
    return _Data(root, data_manifest_hash(root), split_examples(load_raw(root)))


def _load(run_id: str, data: _Data, runs_root: Path) -> ReportRun:
    """A finished run with its questions, once its scores fit the data and each other."""
    check_run_id(run_id)
    stored = read_run(runs_root / run_id)
    manifest = stored.manifest
    try:
        options = RunOptions.model_validate(manifest.config)
    except ValidationError as exc:
        raise RunCorrupt(f"run {run_id}'s config lacks what a report shows: {exc}") from exc
    if manifest.data_manifest_hash != data.hash:
        raise ValueError(
            f"run {run_id} was made on other data: its data manifest hash is "
            f"{manifest.data_manifest_hash[:12]}, and {data.root} hashes to {data.hash[:12]}; "
            "point FORGE_DATA_DIR at the data it used"
        )
    if options.split == "dev":
        _check_dev_use(run_id)
    questions = {example.id: example for example in data.splits[options.split]}
    examples: dict[str, Example] = {}
    for score in stored.scores:
        _check_score(run_id, score, options.retry)
        if score.example_id not in questions:
            raise RunCorrupt(f"run {run_id} scored {score.example_id}, which {options.split} lacks")
        examples[score.example_id] = questions[score.example_id]
    clusters = gold_clusters(examples.values())
    summary = summarize(stored.scores, clusters=clusters)
    return ReportRun(stored, options, examples, clusters, summary, _official(summary.ex_official))


def _check_dev_use(run_id: str) -> None:
    path = dev_usage_path()
    if not any(logged == run_id for _, logged, _ in read_uses(path)):
        raise ValueError(
            f"run {run_id} answered dev questions, but {path} has no row for it; its report "
            "is part of the look the run logged (ADR-0004), so the row must be there"
        )


def _check_score(run_id: str, score: Score, retry: int) -> None:
    """Refuse a score a finished run can't hold, since the report would misread it."""
    sample = f"run {run_id}'s {score.example_id} sample {score.sample_idx}"
    if score.ex_official is None:
        raise RunCorrupt(f"{sample} has no official verdict, though the run finished")
    if score.attempts < 1:
        raise RunCorrupt(f"{sample} took {score.attempts} replies, where every score takes one")
    if score.attempts > 1 + retry:
        raise RunCorrupt(
            f"{sample} took {score.attempts} replies, but the run "
            f"{'retries once' if retry else 'never retries'}"
        )
    if score.attempts > 1 and score.first_try_ex is None:
        raise RunCorrupt(f"{sample} has no official verdict on its first try, though it finished")
    if score.attempts == 1 and score.first_try_ex is not None:
        raise RunCorrupt(f"{sample} has a first try's verdict, but took one reply")
    if score.error_class is not None and score.error_class not in ERROR_CLASSES:
        raise RunCorrupt(f"{sample} has the unknown error class {score.error_class!r}")
    # A run classifies each miss of fast EX and nothing else (forge.eval.scoring).
    if score.ex_fast and score.error_class is not None:
        raise RunCorrupt(f"{sample} matches by fast EX but has the error class {score.error_class}")
    if not score.ex_fast and score.error_class is None:
        raise RunCorrupt(f"{sample} misses by fast EX but has no error class")


def _check(path: Path, rebuilt: str) -> Check:
    stored = path.read_bytes().decode("utf-8")
    diff = difflib.unified_diff(
        _masked(stored).split("\n"),
        _masked(rebuilt).split("\n"),
        fromfile=str(path),
        tofile="rebuilt from the stored files",
        lineterm="",
    )
    return Check(path, tuple(diff))


def _masked(text: str) -> str:
    """``text`` with its first ``Generated:`` line, the one that changes, made constant."""
    return _GENERATED_LINE.sub(f"{GENERATED}(any time)", text, count=1)


def _generated(at: datetime) -> str:
    if at.utcoffset() is None:
        raise ValueError("a report's time must be timezone-aware")
    return f"{GENERATED}{at.astimezone(UTC):%Y-%m-%d %H:%M:%S} UTC"


def _official(rate: Rate | None) -> Rate:
    if rate is None:
        raise RunCorrupt("a score has no official verdict")
    return rate


def _at(scores: Iterable[Score], level: Hardness) -> list[Score]:
    return [score for score in scores if score.hardness == level]


def _split(run: ReportRun) -> str:
    options = run.options
    limit = f" (limit {options.limit:,})" if options.limit is not None else ""
    samples = options.generation.n
    questions = _count(run.summary.questions, "question")
    return f"{options.split}: {questions}{limit}, {_count(samples, 'sample')} each"


def _generation(params: GenParams) -> str:
    tokens = f"at most {params.max_new_tokens:,} new tokens"
    if params.temperature == 0:
        return f"greedy, {tokens}"
    return (
        f"sampled at temperature {params.temperature:g}, top-p {params.top_p:g}, "
        f"seed {params.seed}, {tokens}"
    )


def _header(run: ReportRun, page: Path, *, shared: bool = True) -> list[tuple[str, str]]:
    """What made the run, as table rows for the markdown file at ``page``. Without
    ``shared``, the rows two compared runs have in common are left out."""
    manifest, options = run.stored.manifest, run.options
    adapter = _code(manifest.adapter_ref) if manifest.adapter_ref else "none"
    return [
        ("Model", _code(manifest.model_ref)),
        ("Adapter", adapter),
        ("Backend", _code(options.backend)),
        # A local-mps run is a pilot on the laptop's GPU, and must read as one (ADR-0016).
        ("Platform", _code(manifest.platform)),
        ("Prompt version", _code(options.prompt_version)),
        *([("Split", _split(run))] if shared else []),
        ("Generation", _generation(options.generation)),
        (
            "Retry",
            "once, with the error, for a query that fails to run" if options.retry else "none",
        ),
        ("Git commit", f"`{manifest.git_sha}`"),
        *([("Data manifest hash", f"`{manifest.data_manifest_hash}`")] if shared else []),
        ("Manifest", _link(run.stored.paths.manifest, page)),
    ]


def _method(run: ReportRun) -> str:
    clusters = len(set(run.clusters.values()))
    return (
        f"Intervals are {LEVEL} percentile bootstraps over {RESAMPLES:,} resamples "
        f"(seed {SEED}) that draw whole questions, with the questions that share a gold "
        f"query drawn together: {_count(run.summary.questions, 'question')} in "
        f"{_count(clusters, 'cluster')}."
    )


def _checker_disagreement(scores: Sequence[Score]) -> list[str]:
    only_fast = sum(score.ex_fast and score.ex_official is False for score in scores)
    only_official = sum(score.ex_official is True and not score.ex_fast for score in scores)
    if not only_fast and not only_official:
        return []
    return [
        f"The checkers disagree on {_count(only_fast + only_official, 'sample')}: only fast "
        f"EX matches {only_fast:,}, and only official EX {only_official:,}.",
        "",
    ]


def _retried(run: ReportRun) -> bool:
    return any(score.attempts > 1 for score in run.stored.scores)


def _retry_section(run: ReportRun) -> list[str]:
    """EX with and without retry, when the run retried a sample; nothing otherwise."""
    if not _retried(run):
        return []
    scores = run.stored.scores
    first = [without_retry(score) for score in scores]
    before = summarize(first, clusters=run.clusters)
    change = compare(first, scores, clusters=run.clusters)
    retried = [score for score in scores if score.attempts > 1]
    return [
        "## Retry",
        "",
        "The run asked once more, with the error, for each query that failed to run, and",
        "scored its sample on the second reply. Without retry, every sample keeps its first",
        "reply: a retried one is then neither valid nor a fast match, and official EX",
        "judges its first query. The change pairs each sample with itself.",
        "",
        f"| | without retry [{LEVEL} interval] | with retry [{LEVEL} interval] "
        f"| change in points [{LEVEL} interval] |",
        "|---|---:|---:|---:|",
        f"| valid SQL | {_rate(before.valid)} | {_rate(run.summary.valid)} | |",
        f"| fast EX | {_rate(before.ex_fast)} | {_rate(run.summary.ex_fast)} | |",
        f"| official EX | {_rate(_official(before.ex_official))} | {_rate(run.official)} "
        f"| {_change(change.delta)} |",
        "",
        "| | samples |",
        "|---|---:|",
        f"| retried: the first query failed to run | {len(retried):,} |",
        f"| the second query ran | {sum(score.valid for score in retried):,} |",
        f"| right on retry, by official EX | {sum(s.ex_official is True for s in retried):,} |",
        f"| right on the first reply, by official EX | "
        f"{sum(s.first_try_ex is True for s in retried):,} |",
        "",
    ]


def _without_retry_row(a: ReportRun, b: ReportRun) -> str:
    """The compare table's row for official EX on each run's first replies alone."""
    first_a = [without_retry(score) for score in a.stored.scores]
    first_b = [without_retry(score) for score in b.stored.scores]
    rate_a = _official(summarize(first_a, clusters=a.clusters).ex_official)
    rate_b = _official(summarize(first_b, clusters=b.clusters).ex_official)
    change = compare(first_a, first_b, clusters=a.clusters)
    rates = f"{_rate(rate_a)} | {_rate(rate_b)}"
    return f"| official EX without retry | {rates} | {_change(change.delta)} |"


def _mcnemar(table: McNemar) -> list[str]:
    if table.pvalue is None:
        return [
            "There is no p-value: each question has several samples, which are not "
            "independent pairs. The interval of the change draws whole questions, so it holds."
        ]
    unit = table.unit
    # The questions that share a gold query aren't independent pairs either, so the test
    # pairs what the intervals draw, and says so.
    return [
        f"The test pairs whole {unit}s, as the intervals draw them: run b gets more right in "
        f"{_count(table.wins_b, unit)}, run a in {table.wins_a:,}, and the rest tie.",
        f"Exact two-sided McNemar test on those {unit}s: {_p(table.pvalue)}.",
    ]


def _failures(scores: Iterable[Score]) -> Counter[str]:
    """Failures of official EX, by error class."""
    return Counter(
        score.error_class or UNCLASSIFIED for score in scores if score.ex_official is False
    )


def _class_rows(*counts: Counter[str]) -> list[str]:
    """Every class in the taxonomy's order, then ``unclassified`` where some run has it."""
    return [*ERROR_CLASSES, *([UNCLASSIFIED] if any(c[UNCLASSIFIED] for c in counts) else [])]


def _classes(run: ReportRun) -> list[str]:
    failures = _failures(run.stored.scores)
    total = sum(failures.values())
    if not total:
        return ["None: official EX matches every sample."]
    lines = [
        "Failures of official EX by the class `forge.eval.taxonomy` gives them: the first",
        "class that fits, in this order.",
        "",
        "| class | failures | share of failures |",
        "|---|---:|---:|",
    ]
    for name in _class_rows(failures):
        lines.append(f"| {name} | {failures[name]:,} | {_pct(failures[name] / total)} |")
    if failures[UNCLASSIFIED]:
        lines += ["", UNCLASSIFIED_NOTE]
    return lines


def _examples(run: ReportRun) -> list[str]:
    by_class: dict[str, list[Score]] = {}
    for score in run.stored.scores:
        if score.ex_official is False:
            by_class.setdefault(score.error_class or UNCLASSIFIED, []).append(score)
    if not by_class:
        return ["", "None."]
    # The most failures first, and the taxonomy's order between classes with as many.
    top = sorted(by_class, key=lambda name: (-len(by_class[name]), _ORDER.index(name)))
    lines = [
        "",
        f"Up to {SHOWN_FAILURES} failures of each of the {TOP_CLASSES} most common classes,",
        "picked by a hash of their ids so they spread over databases. The executor's error",
        "message isn't stored, so each shows its class.",
    ]
    for name in top[:TOP_CLASSES]:
        found = by_class[name]
        shown = sorted(sorted(found, key=_pick)[:SHOWN_FAILURES], key=key)
        lines += ["", f"### {name}: {len(shown):,} of {len(found):,} shown"]
        for score in shown:
            lines += _failure(run, score)
    return lines


def _pick(score: Score) -> str:
    return hashlib.sha256(f"{score.example_id}/{score.sample_idx}".encode()).hexdigest()


def _failure(run: ReportRun, score: Score) -> list[str]:
    example = run.examples[score.example_id]
    generation = run.stored.generations[key(score)]
    sample = f", sample {score.sample_idx}" if run.options.generation.n > 1 else ""
    lines = [
        "",
        f"#### {example.id} on {example.db_id} ({score.hardness}{sample})",
        "",
        f"Question: {_escaped(example.question)}",
        "",
        "Gold query:",
        "",
        *_block("sql", example.gold_sql),
        "",
    ]
    reply, query = "The reply", "Predicted query"
    if score.attempts > 1:
        lines += [*_first_try(run.stored.first_tries[key(score)]), ""]
        reply, query = "The reply on retry", "Predicted query on retry"
    if generation.sql is None:
        text, cut = _cut(generation.text)
        lines += [f"{reply} has no SQL query{'. Its start' if cut else ''}:", ""]
        lines += _block("text", text)
    else:
        text, cut = _cut(generation.sql)
        lines += [f"{query}{', cut short' if cut else ''}:", "", *_block("sql", text)]
    return [*lines, "", f"Error: {_error(score)}"]


def _first_try(first: Generation) -> list[str]:
    """A retried sample's first reply: the query that failed to run, or that it had none."""
    if first.sql is None:
        return ["The first reply has no SQL query."]
    text, cut = _cut(first.sql)
    return [
        f"The first reply's query failed to run{'. Its start' if cut else ''}:",
        "",
        *_block("sql", text),
    ]


def _error(score: Score) -> str:
    if score.error_class is None:
        return f"{UNCLASSIFIED}. Fast EX matches it and official EX doesn't."
    if score.error_class == "no_sql":
        return "no_sql. There is no query to execute."
    if not score.valid:
        return f"{score.error_class}. The query failed to execute."
    return f"{score.error_class}. The query ran, and official EX finds no match."


def _cut(text: str) -> tuple[str, bool]:
    return (text, False) if len(text) <= SHOWN_CHARS else (text[:SHOWN_CHARS], True)


def _printable(text: str, *, keep: str = "\n\t") -> str:
    """``text`` with each character that doesn't print, other than ``keep``, escaped."""
    return "".join(c if c.isprintable() or c in keep else repr(c)[1:-1] for c in text)


def _block(language: str, text: str) -> list[str]:
    """A fenced code block that nothing in ``text`` can close early."""
    clean = _printable(text)
    longest = max((len(run) for run in _BACKTICKS.findall(clean)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{language}", *clean.split("\n"), fence]


def _escaped(text: str) -> str:
    """``text`` on one line, as markdown that shows it as it is."""
    return _MARKUP.sub(r"\\\1", _printable(" ".join(text.split())))


def _code(text: str) -> str:
    """``text`` as inline code that a table cell can hold."""
    clean = _printable(text, keep="")
    ticks = "`" * (max((len(run) for run in _BACKTICKS.findall(clean)), default=0) + 1)
    pad = " " if clean.startswith("`") or clean.endswith("`") else ""
    return f"{ticks}{pad}{clean}{pad}{ticks}".replace("|", "\\|")


def _link(target: Path, page: Path) -> str:
    """A link to ``target`` that works from the markdown file at ``page``."""
    relative = Path(os.path.relpath(target.absolute(), page.absolute().parent)).as_posix()
    if any(c in relative for c in " ()<>"):
        relative = f"<{relative}>"
    return f"[{target.name}]({relative})"


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def _pct(share: float) -> str:
    return f"{100 * share:.1f}%"


def _rate(rate: Rate) -> str:
    return f"{_pct(rate.value)} [{_pct(rate.low)}, {_pct(rate.high)}]"


def _points(change: float) -> str:
    # Adding 0.0 turns a rounded -0.0 into 0.0, so no change shows as +0.0.
    return f"{round(100 * change, 1) + 0.0:+.1f}"


def _change(change: Estimate) -> str:
    return f"{_points(change.value)} [{_points(change.low)}, {_points(change.high)}]"


def _p(pvalue: float) -> str:
    return "p < 0.0001" if pvalue < 0.0001 else f"p = {pvalue:.4f}"
