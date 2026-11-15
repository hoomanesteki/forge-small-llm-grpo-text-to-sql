"""Scores in aggregate (stage 9): rates with intervals and paired comparisons.

Every interval is a 95% percentile bootstrap over 1,000 resamples from
``numpy.random.default_rng(0)``, seeded afresh for each interval. An interval then
depends only on the scores it covers, so a report rebuilt from stored scores gets the
same numbers, and the intervals over one set of questions share their resamples.

A resample draws whole questions, since the samples of one question are not
independent.

``compare`` counts a sample as right by official EX when every score of both runs has
an official verdict, and by fast EX otherwise (``ex_metric``). Reported numbers are
official (ADR-0008), and a run stopped before its official batch can still be compared.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, get_args

import numpy as np
from pydantic import BaseModel

from forge.core.types import Hardness, Score
from forge.eval.run_files import Key, key

Metric = Literal["ex_fast", "ex_official"]
Outcome = Literal["valid", "ex_fast", "ex_official"]

LEVELS: tuple[Hardness, ...] = get_args(Hardness)
RESAMPLES = 1000
SEED = 0
PERCENTILES = (2.5, 97.5)  # a 95% interval
SHOWN_SAMPLES = 3


class Estimate(BaseModel, frozen=True):
    """A point estimate with its 95% bootstrap interval."""

    value: float
    low: float
    high: float


class Rate(Estimate, frozen=True):
    """A share of the samples: ``hits`` of the ``n`` beside it."""

    hits: int


class HardnessEx(BaseModel, frozen=True):
    """Execution match on the questions of one hardness level."""

    n: int  # scores
    questions: int
    ex_fast: Rate
    ex_official: Rate | None


class Summary(BaseModel, frozen=True):
    """What ``summarize`` returns."""

    n: int  # scores
    questions: int
    valid: Rate
    ex_fast: Rate
    # None unless every score has an official verdict: a run stopped before its official
    # batch has verdicts only for its timeouts, which are all misses.
    ex_official: Rate | None
    by_hardness: dict[Hardness, HardnessEx]  # the levels present, easy to extra


class McNemar(BaseModel, frozen=True):
    """Paired outcomes as a 2x2 table, with the exact McNemar test on them."""

    both: int  # right in a and in b
    only_a: int  # right in a, wrong in b
    only_b: int  # wrong in a, right in b
    neither: int  # wrong in both
    # Two-sided, with each question an independent pair. None when a question has
    # several samples, which are not independent.
    pvalue: float | None


class Comparison(BaseModel, frozen=True):
    """What ``compare`` returns: run b against run a on the same samples."""

    metric: Metric
    n: int  # pairs
    questions: int
    ex_a: float
    ex_b: float
    delta: Estimate  # ex_b - ex_a, with a paired bootstrap interval
    mcnemar: McNemar


def summarize(scores: Iterable[Score]) -> Summary:
    """The valid rate and EX, fast and official, with EX by hardness too; each with its
    interval.

    Raises ``ValueError`` for no scores, a sample scored twice, or a question scored at
    two hardness levels.
    """
    questions = _questions(scores)
    official = all(score.ex_official is not None for group in questions.values() for score in group)
    levels = {
        level: {qid: group for qid, group in questions.items() if group[0].hardness == level}
        for level in LEVELS
    }
    return Summary(
        n=sum(len(group) for group in questions.values()),
        questions=len(questions),
        valid=_rate(questions, "valid"),
        ex_fast=_rate(questions, "ex_fast"),
        ex_official=_rate(questions, "ex_official") if official else None,
        by_hardness={
            level: HardnessEx(
                n=sum(len(group) for group in groups.values()),
                questions=len(groups),
                ex_fast=_rate(groups, "ex_fast"),
                ex_official=_rate(groups, "ex_official") if official else None,
            )
            for level, groups in levels.items()
            if groups
        },
    )


def compare(scores_a: Iterable[Score], scores_b: Iterable[Score]) -> Comparison:
    """Run b against run a, each sample paired with the same (example id, sample) in the other.

    Raises ``ValueError`` unless both runs scored the same samples, and for scores
    ``summarize`` refuses.
    """
    a = _questions(scores_a)
    a_scores = [score for group in a.values() for score in group]
    b = {key(score): score for group in _questions(scores_b).values() for score in group}
    a_keys = {key(score) for score in a_scores}
    if a_keys != b.keys():
        raise ValueError(_mismatch(a_keys, set(b)))
    metric = ex_metric([*a_scores, *b.values()])
    table: Counter[tuple[bool, bool]] = Counter()
    gains: list[int] = []
    sizes: list[int] = []
    for group in a.values():
        pairs = [(_hit(score, metric), _hit(b[key(score)], metric)) for score in group]
        table.update(pairs)
        gains.append(sum(int(right_b) - int(right_a) for right_a, right_b in pairs))
        sizes.append(len(pairs))
    n = len(a_scores)
    both, only_a = table[True, True], table[True, False]
    only_b, neither = table[False, True], table[False, False]
    low, high = _interval(gains, sizes)
    return Comparison(
        metric=metric,
        n=n,
        questions=len(a),
        ex_a=(both + only_a) / n,
        ex_b=(both + only_b) / n,
        delta=Estimate(value=(only_b - only_a) / n, low=low, high=high),
        mcnemar=McNemar(
            both=both,
            only_a=only_a,
            only_b=only_b,
            neither=neither,
            pvalue=_mcnemar_pvalue(both, only_a, only_b, neither) if n == len(a) else None,
        ),
    )


def ex_metric(scores: Iterable[Score]) -> Metric:
    """Official EX when every score has an official verdict, else fast EX."""
    return "ex_official" if all(score.ex_official is not None for score in scores) else "ex_fast"


def _questions(scores: Iterable[Score]) -> dict[str, list[Score]]:
    """Scores by question, in example id order, so the order they come in can't move an
    interval."""
    groups: dict[str, list[Score]] = {}
    seen: set[Key] = set()
    for score in scores:
        if key(score) in seen:
            raise ValueError(f"{score.example_id} sample {score.sample_idx} is scored twice")
        seen.add(key(score))
        group = groups.setdefault(score.example_id, [])
        if group and group[0].hardness != score.hardness:
            raise ValueError(
                f"{score.example_id} is scored as both {group[0].hardness} and {score.hardness}"
            )
        group.append(score)
    if not groups:
        raise ValueError("there are no scores to aggregate")
    return {example_id: groups[example_id] for example_id in sorted(groups)}


def _hit(score: Score, outcome: Outcome) -> bool:
    if outcome == "valid":
        return score.valid
    if outcome == "ex_fast":
        return score.ex_fast
    return score.ex_official is True


def _rate(questions: Mapping[str, Sequence[Score]], outcome: Outcome) -> Rate:
    hits = [sum(_hit(score, outcome) for score in group) for group in questions.values()]
    sizes = [len(group) for group in questions.values()]
    low, high = _interval(hits, sizes)
    return Rate(hits=sum(hits), value=sum(hits) / sum(sizes), low=low, high=high)


def _interval(numerators: Sequence[int], denominators: Sequence[int]) -> tuple[float, float]:
    """The bootstrap interval of the summed numerators over the summed denominators, one of
    each a question, drawing questions with replacement."""
    totals = np.array(list(zip(numerators, denominators, strict=True)), dtype=np.int64)
    picks = np.random.default_rng(SEED).integers(0, len(totals), size=(RESAMPLES, len(totals)))
    drawn = totals[picks].sum(axis=1)
    low, high = np.percentile(drawn[:, 0] / drawn[:, 1], PERCENTILES)
    return float(low), float(high)


def _mcnemar_pvalue(both: int, only_a: int, only_b: int, neither: int) -> float:
    # statsmodels brings in pandas and scipy, about a second, and only compare needs it.
    from statsmodels.stats.contingency_tables import mcnemar

    return float(mcnemar([[both, only_a], [only_b, neither]], exact=True).pvalue)


def _mismatch(a: set[Key], b: set[Key]) -> str:
    parts = []
    for name, extra in (("a", a - b), ("b", b - a)):
        if extra:
            shown = ", ".join(f"{i} sample {j}" for i, j in sorted(extra)[:SHOWN_SAMPLES])
            more = ", ..." if len(extra) > SHOWN_SAMPLES else ""
            parts.append(f"{len(extra)} only in run {name} ({shown}{more})")
    return "the runs scored different samples, so they can't be paired: " + "; ".join(parts)
