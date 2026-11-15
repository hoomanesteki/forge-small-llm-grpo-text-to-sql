"""Scores in aggregate (stage 9): the valid rate and EX, with intervals.

Every interval is a 95% percentile bootstrap over 1,000 resamples from
``numpy.random.default_rng(0)``, seeded afresh for each interval. An interval then
depends only on the scores it covers, so a report rebuilt from stored scores gets the
same numbers, and the intervals over one set of questions share their resamples.

A resample draws whole questions, since the samples of one question are not
independent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, get_args

import numpy as np
from pydantic import BaseModel

from forge.core.types import Hardness, Score
from forge.eval.run_files import Key, key

Outcome = Literal["valid", "ex_fast", "ex_official"]

LEVELS: tuple[Hardness, ...] = get_args(Hardness)
RESAMPLES = 1000
SEED = 0
PERCENTILES = (2.5, 97.5)  # a 95% interval


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
