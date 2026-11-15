"""Invariant I4 for a run's report: every number in it traces to the run's stored files.

``untraced`` reads what a run stored straight from its folder (the manifest, and the
scores, generations and first tries the manifest lists) and the questions of its split
from data whose hash is the manifest's. From them it recomputes what a report shows:
counts and rates by adding up the scores, intervals by summarizing those scores as
``forge.eval.metrics`` does. Then it reads the report a line at a time. A line that
reports something must hold the recomputed numbers, in order, besides the identifiers it
names (the run, the model, the commit, an example); a table row is known by its section
and first cell. The method's constants (the interval level, the resamples and seed, how
many failures are shown) are allowed where the report states them, and so is Spider's
citation. Question lines and code blocks quote the data and the stored replies: each must
be a verbatim quote, and its numbers are then the data's. Every other line must hold no
digit. The ``Generated:`` line, the time the report was written, is the one line left
out, as a rebuild leaves it out.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from forge.core.types import Example, Generation, RunManifest, Score
from forge.data.manifest import data_manifest_hash
from forge.data.spider import load_raw
from forge.data.splits import split_examples
from forge.eval.metrics import (
    RESAMPLES,
    SEED,
    Estimate,
    HardnessEx,
    Summary,
    compare,
    gold_clusters,
    summarize,
)
from forge.eval.report import LEVEL, SHOWN_CHARS, SHOWN_FAILURES, SPIDER, TOP_CLASSES, UNCLASSIFIED
from forge.eval.taxonomy import ERROR_CLASSES

M = TypeVar("M", bound=BaseModel)
Key = tuple[str, int]  # (example_id, sample_idx)
# What a line must hold: the identifiers it names, and its numbers in order.
Expected = tuple[Sequence[str], Sequence[str]]
NOTHING: Expected = ((), ())

# A number as a report writes one: a count with thousands separators, a decimal, a
# percentage, or a change with its sign. Not digits inside a word, such as the 1 of p1.
NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?")
GENERATED = re.compile(r"Generated: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC")
SEPARATOR = re.compile(r"\|[-:|]+\|")
FENCE = re.compile(r"(`{3,})(?:sql|text)")
FAILURE = re.compile(r"#### (\S+) on (\S+) \((\w+)(?:, sample (\d+))?\)")
LINK = re.compile(r"\[manifest\.json\]\(<?([^)>]+)>?\)")
# What a question line escapes: the characters markdown could read as markup.
ESCAPE = re.compile(r"\\([\\`*_\[\]<>|~&])")


@dataclass(frozen=True)
class Stored:
    """A run's files as its folder holds them, with the questions it answered."""

    run_dir: Path
    manifest: RunManifest
    scores: tuple[Score, ...]
    generations: dict[Key, Generation]
    first_tries: dict[Key, Generation]
    examples: dict[str, Example]


def untraced(text: str, run_dir: Path, *, data_dir: Path) -> list[str]:
    """What in ``text``, the report of the run in ``run_dir``, that run's stored files don't
    give: one message a finding, and none when every number traces."""
    return _Trace(read_stored(run_dir, data_dir)).check(text.split("\n"))


def read_stored(run_dir: Path, data_dir: Path) -> Stored:
    """The run's manifest and the files it lists, with its questions from ``data_dir``.

    Raises ``ValueError`` when the data isn't what the run was made on.
    """
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_bytes())
    if data_manifest_hash(data_dir) != manifest.data_manifest_hash:
        raise ValueError(f"{data_dir} is not the data run {manifest.run_id} was made on")
    outputs = manifest.outputs
    scores = tuple(_lines(run_dir / outputs["scores"], Score))
    generations = _lines(run_dir / outputs["generations"], Generation)
    firsts = (
        _lines(run_dir / outputs["first_tries"], Generation) if "first_tries" in outputs else []
    )
    questions = {e.id: e for e in split_examples(load_raw(data_dir))[manifest.config["split"]]}
    return Stored(
        run_dir,
        manifest,
        scores,
        {_key(generation): generation for generation in generations},
        {_key(first): first for first in firsts},
        {score.example_id: questions[score.example_id] for score in scores},
    )


class _Trace:
    """One pass over a report's lines, against what ``stored`` gives."""

    def __init__(self, stored: Stored) -> None:
        self.stored = stored
        self.rows, self.prose = _expected(stored)
        self.failures = _failures(stored.scores)
        self.problems: list[str] = []
        self.section = ""
        self.failure: Score | None = None  # the failure the lines below show
        # Each example class heading: its class, the failures it says it shows, and those
        # shown under it.
        self.headings: list[tuple[str, int, list[Score]]] = []
        self.generated = 0

    def check(self, lines: Sequence[str]) -> list[str]:
        i = 0
        while i < len(lines):
            fence = FENCE.fullmatch(lines[i])
            if fence is None:
                following = lines[i + 1] if i + 1 < len(lines) else ""
                self._line(f"line {i + 1}", lines[i], following)
                i += 1
                continue
            try:
                end = lines.index(fence.group(1), i + 1)
            except ValueError:
                self.problems.append(f"line {i + 1}: a code block that never closes")
                break
            self._quote(f"line {i + 1}", "\n".join(lines[i + 1 : end]))
            i = end + 1
        for name, stated, shown in self.headings:
            if len(shown) != stated:
                self.problems.append(
                    f"{name}: says {stated} failures are shown, and shows {len(shown)}"
                )
        if self.generated != 1:
            self.problems.append(f"{self.generated} Generated lines, where a report has one")
        return self.problems

    def _line(self, where: str, line: str, following: str) -> None:
        if line.startswith("#"):
            self.failure = None
        if line.startswith("## "):
            self.section = line.removeprefix("## ")
        if line.startswith("Generated: "):
            self.generated += 1
            if not GENERATED.fullmatch(line):
                self.problems.append(f"{where}: not the time of writing: {line!r}")
        elif line == SPIDER:
            return
        elif line.startswith("|"):
            self._row(where, line, following)
        elif line.startswith("#### "):
            self._failure(where, line)
        elif line.startswith("### "):
            name = line.removeprefix("### ").split(":")[0]
            found = self.failures[name]
            stated = min(SHOWN_FAILURES, found)
            self.headings.append((name, stated, []))
            self._expect(where, line, _says(f"{stated:,}", f"{found:,}"))
        elif line.startswith("Question: "):
            self._question(where, line.removeprefix("Question: "))
        else:
            prose = (e for prefix, e in self.prose.items() if line.startswith(prefix))
            self._expect(where, line, next(prose, NOTHING))

    def _row(self, where: str, line: str, following: str) -> None:
        if SEPARATOR.fullmatch(line):
            return
        if SEPARATOR.fullmatch(following):
            # A header row: the one number a header holds is the interval level.
            self._expect(where, line, _says(*[LEVEL] * len(NUMBER.findall(line))))
            return
        label = line.split("|")[1].strip()
        expected = self.rows.get((self.section, label))
        if expected is None:
            self.problems.append(f"{where}: a row the stored files don't give: {line!r}")
            return
        self._expect(where, line, expected)
        if (self.section, label) == ("", "Manifest"):
            link = LINK.search(line)
            run_dir = self.stored.run_dir
            target = (run_dir / "eval" / link.group(1)).resolve() if link else None
            if target != (run_dir / "manifest.json").resolve():
                self.problems.append(f"{where}: the link isn't to the run's manifest: {line!r}")

    def _failure(self, where: str, line: str) -> None:
        match = FAILURE.fullmatch(line)
        if match is None or not self.headings:
            self.problems.append(f"{where}: no failure of the run reads {line!r}")
            return
        example_id, db_id, hardness, sample = match.groups()
        sample_key = (example_id, int(sample or 0))
        score = next((s for s in self.stored.scores if _key(s) == sample_key), None)
        name, _, shown = self.headings[-1]
        if (
            score is None
            or score.ex_official is not False
            or (score.error_class or UNCLASSIFIED) != name
            or self.stored.examples[example_id].db_id != db_id
            or score.hardness != hardness
        ):
            self.problems.append(f"{where}: no {name} failure of the run reads {line!r}")
            return
        self.failure = score
        shown.append(score)
        several = self.stored.manifest.config["generation"]["n"] > 1
        sample_idx = (str(score.sample_idx),) if several else ()
        self._expect(where, line, ((example_id, db_id), sample_idx))

    def _question(self, where: str, text: str) -> None:
        if self.failure is None:
            self.problems.append(f"{where}: a question outside any failure")
            return
        question = self.stored.examples[self.failure.example_id].question
        if ESCAPE.sub(r"\1", text) != _printable(" ".join(question.split())):
            self.problems.append(f"{where}: not the question of {self.failure.example_id}")

    def _quote(self, where: str, body: str) -> None:
        """A code block must quote the failure's gold query or one of its stored replies."""
        score = self.failure
        if score is None:
            self.problems.append(f"{where}: a code block outside any failure")
            return
        reply = self.stored.generations[_key(score)]
        first = self.stored.first_tries.get(_key(score))
        # Replies show cut short; gold queries in full.
        replies = (reply.sql, reply.text, first.sql if first is not None else None)
        quoted = [
            self.stored.examples[score.example_id].gold_sql,
            *(text[:SHOWN_CHARS] for text in replies if text is not None),
        ]
        if body not in {_printable(text) for text in quoted}:
            self.problems.append(f"{where}: a code block that quotes nothing of {score.example_id}")

    def _expect(self, where: str, line: str, expected: Expected) -> None:
        identifiers, numbers = expected
        rest = line
        for name in identifiers:
            if name not in rest:
                self.problems.append(f"{where}: {name!r} isn't in {line!r}")
            rest = rest.replace(name, " ")
        found = NUMBER.findall(rest)
        want = [token for number in numbers for token in NUMBER.findall(number)]
        if found != want:
            self.problems.append(f"{where}: {found} where the stored files give {want}: {line!r}")
        elif any(char.isdigit() for char in NUMBER.sub(" ", rest)):
            self.problems.append(f"{where}: digits that are no number: {line!r}")


def _expected(stored: Stored) -> tuple[dict[tuple[str, str], Expected], dict[str, Expected]]:
    """What each table row, by section and first cell, must hold, and each line that
    starts with one of the prefixes."""
    manifest, scores = stored.manifest, stored.scores
    config, generation = manifest.config, manifest.config["generation"]
    n, questions = len(scores), len({score.example_id for score in scores})
    clusters = gold_clusters(stored.examples.values())
    summary = summarize(scores, clusters=clusters)
    valid, fast, official = _hits(scores)
    limit = () if config["limit"] is None else (f"{config['limit']:,}",)
    sampled = (
        ()
        if generation["temperature"] == 0
        else (f"{generation['temperature']:g}", f"{generation['top_p']:g}", str(generation["seed"]))
    )
    rows: dict[tuple[str, str], Expected] = {
        ("", "Model"): _names(manifest.model_ref),
        ("", "Adapter"): _names(*filter(None, [manifest.adapter_ref])),
        ("", "Backend"): _names(config["backend"]),
        ("", "Prompt version"): _names(config["prompt_version"]),
        ("", "Split"): _says(f"{questions:,}", *limit, f"{generation['n']:,}"),
        ("", "Generation"): _says(*sampled, f"{generation['max_new_tokens']:,}"),
        ("", "Retry"): NOTHING,
        ("", "Git commit"): _names(manifest.git_sha),
        ("", "Data manifest hash"): _names(manifest.data_manifest_hash),
        ("", "Manifest"): NOTHING,
        ("Execution accuracy", "valid SQL"): _says(
            *_right_of(valid, n), *_rate(valid, n, summary.valid)
        ),
        ("Execution accuracy", "fast EX"): _says(
            *_right_of(fast, n), *_rate(fast, n, summary.ex_fast)
        ),
        ("Execution accuracy", "official EX"): _says(
            *_right_of(official, n), *_rate(official, n, _official(summary))
        ),
    }
    for level, ex in summary.by_hardness.items():
        at = [score for score in scores if score.hardness == level]
        _, fast_at, official_at = _hits(at)
        rows["By hardness", level] = _says(
            f"{len({score.example_id for score in at}):,}",
            *_rate(fast_at, len(at), ex.ex_fast),
            *_rate(official_at, len(at), _official(ex)),
        )
    failures = _failures(scores)
    total = sum(failures.values())
    for name in (*ERROR_CLASSES, UNCLASSIFIED):
        share = (_pct(failures[name] / total),) if total else ()
        rows["Error classes", name] = _says(f"{failures[name]:,}", *share)
    rows.update(_retry_rows(scores, clusters, summary))
    only_fast = sum(score.ex_fast and score.ex_official is False for score in scores)
    only_official = sum(score.ex_official is True and not score.ex_fast for score in scores)
    groups = len(set(clusters.values()))
    prose: dict[str, Expected] = {
        "# Run ": ((manifest.run_id,), _rate(official, n, _official(summary))),
        "Written by ": _names(manifest.run_id),
        "against them by ": _names(manifest.run_id),
        "The checkers disagree on ": _says(
            f"{only_fast + only_official:,}", f"{only_fast:,}", f"{only_official:,}"
        ),
        "Intervals are ": _says(
            LEVEL, f"{RESAMPLES:,}", str(SEED), f"{questions:,}", f"{groups:,}"
        ),
        "![": _says(LEVEL),
        "Up to ": _says(str(SHOWN_FAILURES), str(TOP_CLASSES)),
    }
    return rows, prose


def _retry_rows(
    scores: Sequence[Score], clusters: dict[str, str], summary: Summary
) -> dict[tuple[str, str], Expected]:
    """The rows of the retry section, which a run has when it retried a sample."""
    retried = [score for score in scores if score.attempts > 1]
    if not retried:
        return {}
    # Without retry, a retried sample has only its first reply, whose query failed to run:
    # not valid, no fast match, and the official verdict on that query.
    first = [
        Score(
            example_id=score.example_id,
            sample_idx=score.sample_idx,
            hardness=score.hardness,
            valid=False,
            ex_fast=False,
            ex_official=score.first_try_ex,
        )
        if score.attempts > 1
        else score
        for score in scores
    ]
    before = summarize(first, clusters=clusters)
    change = compare(first, scores, clusters=clusters).delta
    n = len(scores)
    (valid_a, fast_a, official_a), (valid_b, fast_b, official_b) = _hits(first), _hits(scores)
    return {
        ("Retry", "valid SQL"): _says(
            *_rate(valid_a, n, before.valid), *_rate(valid_b, n, summary.valid)
        ),
        ("Retry", "fast EX"): _says(
            *_rate(fast_a, n, before.ex_fast), *_rate(fast_b, n, summary.ex_fast)
        ),
        ("Retry", "official EX"): _says(
            *_rate(official_a, n, _official(before)),
            *_rate(official_b, n, _official(summary)),
            _points((official_b - official_a) / n),
            _points(change.low),
            _points(change.high),
        ),
        ("Retry", "retried: the first query failed to run"): _says(f"{len(retried):,}"),
        ("Retry", "the second query ran"): _says(f"{sum(s.valid for s in retried):,}"),
        ("Retry", "right on retry, by official EX"): _says(
            f"{sum(s.ex_official is True for s in retried):,}"
        ),
        ("Retry", "right on the first reply, by official EX"): _says(
            f"{sum(s.first_try_ex is True for s in retried):,}"
        ),
    }


def _names(*identifiers: str) -> Expected:
    return identifiers, ()


def _says(*numbers: str) -> Expected:
    return (), numbers


def _hits(scores: Iterable[Score]) -> tuple[int, int, int]:
    """How many scores ran, matched by fast EX, and matched by official EX."""
    valid = fast = official = 0
    for score in scores:
        valid += score.valid
        fast += score.ex_fast
        official += score.ex_official is True
    return valid, fast, official


def _failures(scores: Iterable[Score]) -> Counter[str]:
    """Failures of official EX by error class; one only official EX finds has none."""
    return Counter(
        score.error_class or UNCLASSIFIED for score in scores if score.ex_official is False
    )


def _official(summary: Summary | HardnessEx) -> Estimate:
    if summary.ex_official is None:
        raise ValueError("a finished run has an official verdict on every sample")
    return summary.ex_official


def _right_of(hits: int, n: int) -> tuple[str, str]:
    return f"{hits:,}", f"{n:,}"


def _rate(hits: int, n: int, interval: Estimate) -> tuple[str, str, str]:
    return _pct(hits / n), _pct(interval.low), _pct(interval.high)


def _pct(share: float) -> str:
    return f"{100 * share:.1f}%"


def _points(change: float) -> str:
    return f"{round(100 * change, 1) + 0.0:+.1f}"


def _printable(text: str) -> str:
    """``text`` as a report prints it: what doesn't print, but newlines and tabs, escaped."""
    return "".join(c if c.isprintable() or c in "\n\t" else repr(c)[1:-1] for c in text)


def _key(item: Generation | Score) -> Key:
    return (item.example_id, item.sample_idx)


def _lines(path: Path, model: type[M]) -> list[M]:
    return [model.model_validate_json(line) for line in path.read_bytes().splitlines()]
