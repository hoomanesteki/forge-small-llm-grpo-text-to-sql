"""Scores in aggregate (stage 9): rates with intervals, paired comparisons and pass@k.

Every interval is a 95% percentile bootstrap over 1,000 resamples from
``numpy.random.default_rng(0)``, seeded afresh for each interval. An interval then
depends only on the scores it covers, so a report rebuilt from stored scores gets the
same numbers, and the intervals over one set of questions share their resamples.

A resample draws whole questions, since the samples of one question are not
independent. Nor are the questions Spider asks twice in other words, which share a gold
query: given ``clusters`` (``gold_clusters``), a resample draws whole clusters instead.

``compare`` and ``pass_at_k`` count a sample as right by official EX when every score
has an official verdict, and by fast EX otherwise (``ex_metric``). Reported numbers are
official (ADR-0008), and a run stopped before its official batch can still be compared.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, get_args

import numpy as np
from pydantic import BaseModel

from forge.core.types import Example, Hardness, Score
from forge.eval.run_files import Key, key

Metric = Literal["ex_fast", "ex_official"]
Outcome = Literal["valid", "ex_fast", "ex_official"]
# Example id to the name of its cluster; questions with one name are drawn together.
Clusters = Mapping[str, str]
# A question's share of a ratio: its example id, numerator and denominator.
Part = tuple[str, int, int]

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
    # Two-sided, with each question an independent pair; clusters widen only the delta's
    # interval. None when a question has several samples, which are not independent.
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


def summarize(scores: Iterable[Score], *, clusters: Clusters | None = None) -> Summary:
    """The valid rate and EX, fast and official, with EX by hardness too; each with its
    interval.

    Raises ``ValueError`` for no scores, a sample scored twice, a question scored at two
    hardness levels, or a scored question that ``clusters`` leaves out.
    """
    questions = _questions(scores, clusters)
    official = all(score.ex_official is not None for group in questions.values() for score in group)
    levels = {
        level: {qid: group for qid, group in questions.items() if group[0].hardness == level}
        for level in LEVELS
    }
    return Summary(
        n=sum(len(group) for group in questions.values()),
        questions=len(questions),
        valid=_rate(questions, "valid", clusters),
        ex_fast=_rate(questions, "ex_fast", clusters),
        ex_official=_rate(questions, "ex_official", clusters) if official else None,
        by_hardness={
            level: HardnessEx(
                n=sum(len(group) for group in groups.values()),
                questions=len(groups),
                ex_fast=_rate(groups, "ex_fast", clusters),
                ex_official=_rate(groups, "ex_official", clusters) if official else None,
            )
            for level, groups in levels.items()
            if groups
        },
    )


def compare(
    scores_a: Iterable[Score], scores_b: Iterable[Score], *, clusters: Clusters | None = None
) -> Comparison:
    """Run b against run a, each sample paired with the same (example id, sample) in the other.

    Raises ``ValueError`` unless both runs scored the same samples, and for scores
    ``summarize`` refuses.
    """
    a = _questions(scores_a, clusters)
    a_scores = [score for group in a.values() for score in group]
    b = {key(score): score for group in _questions(scores_b).values() for score in group}
    a_keys = {key(score) for score in a_scores}
    if a_keys != b.keys():
        raise ValueError(_mismatch(a_keys, set(b)))
    metric = ex_metric([*a_scores, *b.values()])
    table: Counter[tuple[bool, bool]] = Counter()
    gains: list[Part] = []
    for example_id, group in a.items():
        pairs = [(_hit(score, metric), _hit(b[key(score)], metric)) for score in group]
        table.update(pairs)
        gain = sum(int(right_b) - int(right_a) for right_a, right_b in pairs)
        gains.append((example_id, gain, len(pairs)))
    n = len(a_scores)
    both, only_a = table[True, True], table[True, False]
    only_b, neither = table[False, True], table[False, False]
    low, high = _interval(gains, clusters)
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


def pass_at_k(scores: Iterable[Score], k: int) -> float:
    """The chance that at least one of k samples is right, averaged over questions.

    Each question with n samples, c of them right, gives the unbiased estimate
    1 - C(n - c, k) / C(n, k) of Chen et al. (2021). Raises ``ValueError`` when k is
    below 1 or a question has fewer than k samples, and for scores ``summarize`` refuses.
    """
    if k < 1:
        raise ValueError(f"pass@k needs k of at least 1, got {k}")
    questions = _questions(scores)
    metric = ex_metric(score for group in questions.values() for score in group)
    estimates: list[float] = []
    for example_id, group in questions.items():
        n = len(group)
        if n < k:
            raise ValueError(f"pass@{k} needs {k} samples of each question; {example_id} has {n}")
        right = sum(_hit(score, metric) for score in group)
        estimates.append(1 - math.comb(n - right, k) / math.comb(n, k))
    return math.fsum(estimates) / len(estimates)


def ex_metric(scores: Iterable[Score]) -> Metric:
    """Official EX when every score has an official verdict, else fast EX."""
    return "ex_official" if all(score.ex_official is not None for score in scores) else "ex_fast"


def gold_clusters(examples: Iterable[Example]) -> dict[str, str]:
    """Each example's cluster: the first example, by id, with its database and gold query.

    Spider asks most questions twice in other words, and the two share a gold query.
    """
    first: dict[tuple[str, str], str] = {}
    return {
        example.id: first.setdefault((example.db_id, example.gold_sql), example.id)
        for example in sorted(examples, key=lambda example: example.id)
    }


def _questions(scores: Iterable[Score], clusters: Clusters | None = None) -> dict[str, list[Score]]:
    """Scores by question, in example id order so the order they come in can't move an
    interval, once the scores and ``clusters`` are known to fit together."""
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
    if clusters is not None:
        unplaced = sorted(groups.keys() - clusters.keys())
        if unplaced:
            raise ValueError(
                f"no cluster for {len(unplaced)} of the scored questions, such as {unplaced[0]}"
            )
    return {example_id: groups[example_id] for example_id in sorted(groups)}


def _hit(score: Score, outcome: Outcome) -> bool:
    if outcome == "valid":
        return score.valid
    if outcome == "ex_fast":
        return score.ex_fast
    return score.ex_official is True


def _rate(
    questions: Mapping[str, Sequence[Score]], outcome: Outcome, clusters: Clusters | None
) -> Rate:
    parts = [
        (example_id, sum(_hit(score, outcome) for score in group), len(group))
        for example_id, group in questions.items()
    ]
    hits = sum(hit for _, hit, _ in parts)
    low, high = _interval(parts, clusters)
    return Rate(hits=hits, value=hits / sum(size for _, _, size in parts), low=low, high=high)


def _interval(parts: Iterable[Part], clusters: Clusters | None) -> tuple[float, float]:
    """The bootstrap interval of the parts' summed numerators over their summed
    denominators, drawing questions, or their clusters, with replacement."""
    units: dict[str, tuple[int, int]] = {}
    for example_id, numerator, denominator in parts:
        unit = example_id if clusters is None else clusters[example_id]
        unit_numerator, unit_denominator = units.get(unit, (0, 0))
        units[unit] = (unit_numerator + numerator, unit_denominator + denominator)
    totals = np.array([units[unit] for unit in sorted(units)], dtype=np.int64)
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
