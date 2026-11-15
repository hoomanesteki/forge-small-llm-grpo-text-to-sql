"""Reports (stage 10): what a run scored, from its stored files only.

``write_run_report`` writes ``eval/<split>-report.md`` in a run's folder, with the chart
``eval/<split>-hardness.png`` beside it. Nothing in a report comes from a run's memory:
``load_run`` reads

- the run's manifest, scores and generations, through ``read_run``, which takes only a
  finished run whose files agree with each other and with the manifest;
- the questions and gold queries of its split, from data whose data manifest hash is
  the manifest's. Other data is refused.

Rates are official EX unless labeled fast (ADR-0008), so every sample needs an official
verdict. Intervals come from ``forge.eval.metrics``, with the questions that share a gold
query drawn together (``gold_clusters``). Error classes count the failures of official
EX. A run classifies the failures of fast EX (``forge.eval.scoring``), so a failure that
only official EX finds has no class and counts as ``unclassified``. The report shows
``SHOWN_FAILURES`` failures of each of the ``TOP_CLASSES`` most common classes, picked by
a hash of their ids so they spread over databases and every rendering picks the same
ones. The executor's error message isn't stored, so a failure shows its class.

A dev run's report shows the model's answers on dev, which belongs to the look the run
logged (ADR-0004): it needs the run's row in ``docs/dev-usage.md`` and adds none.

One line changes between renderings: ``Generated:`` and the time. The chart draws the
by-hardness table; its bytes are the same on every rendering on one machine, so a
committed chart changes only with its numbers.

Each kind of number has one format, so FG-207's I4 check can find every number in the
stored files: a rate as a percentage with one decimal and its 95% interval in brackets
(``45.2% [41.8%, 48.6%]``), and counts with thousands separators. Constants (the
interval level, resamples and seed, how many failures are shown, and Spider's citation)
appear in prose lines only.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from forge.core.types import Example, GenParams, Score, Split
from forge.data.manifest import data_manifest_hash
from forge.data.spider import load_raw, resolve_data_dir
from forge.data.splits import split_examples
from forge.eval.config import check_run_id
from forge.eval.metrics import (
    PERCENTILES,
    RESAMPLES,
    SEED,
    Rate,
    Summary,
    gold_clusters,
    summarize,
)
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


def render_run(run: ReportRun, *, generated: datetime) -> str:
    """The run's report as markdown."""
    summary = run.summary
    lines = [
        f"# Run {run.run_id}: official EX {_rate(run.official)} on {run.options.split}",
        "",
        _generated(generated),
        "",
        f"Written by `forge report run {run.run_id}` from the run's stored files; don't edit",
        "it by hand.",
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
        _check_score(run_id, score)
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


def _check_score(run_id: str, score: Score) -> None:
    """Refuse a score a finished run can't hold, since the report would misread it."""
    sample = f"run {run_id}'s {score.example_id} sample {score.sample_idx}"
    if score.ex_official is None:
        raise RunCorrupt(f"{sample} has no official verdict, though the run finished")
    if score.error_class is not None and score.error_class not in ERROR_CLASSES:
        raise RunCorrupt(f"{sample} has the unknown error class {score.error_class!r}")
    # A run classifies each miss of fast EX and nothing else (forge.eval.scoring).
    if score.ex_fast and score.error_class is not None:
        raise RunCorrupt(f"{sample} matches by fast EX but has the error class {score.error_class}")
    if not score.ex_fast and score.error_class is None:
        raise RunCorrupt(f"{sample} misses by fast EX but has no error class")


def _generated(at: datetime) -> str:
    if at.utcoffset() is None:
        raise ValueError("a report's time must be timezone-aware")
    return f"{GENERATED}{at.astimezone(UTC):%Y-%m-%d %H:%M:%S} UTC"


def _official(rate: Rate | None) -> Rate:
    if rate is None:
        raise RunCorrupt("a score has no official verdict")
    return rate


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


def _header(run: ReportRun, page: Path) -> list[tuple[str, str]]:
    """What made the run, as table rows for the markdown file at ``page``."""
    manifest, options = run.stored.manifest, run.options
    adapter = _code(manifest.adapter_ref) if manifest.adapter_ref else "none"
    return [
        ("Model", _code(manifest.model_ref)),
        ("Adapter", adapter),
        ("Backend", _code(options.backend)),
        ("Prompt version", _code(options.prompt_version)),
        ("Split", _split(run)),
        ("Generation", _generation(options.generation)),
        ("Git commit", f"`{manifest.git_sha}`"),
        ("Data manifest hash", f"`{manifest.data_manifest_hash}`"),
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
    if generation.sql is None:
        text, cut = _cut(generation.text)
        lines += [f"The reply has no SQL query{'. Its start' if cut else ''}:", ""]
        lines += _block("text", text)
    else:
        text, cut = _cut(generation.sql)
        lines += [f"Predicted query{', cut short' if cut else ''}:", "", *_block("sql", text)]
    return [*lines, "", f"Error: {_error(score)}"]


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
