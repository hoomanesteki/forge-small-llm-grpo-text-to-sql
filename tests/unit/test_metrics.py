"""Scores in aggregate: rates with bootstrap intervals."""

from __future__ import annotations

import math
import random

import pytest

from forge.core.types import Hardness, Score
from forge.eval.metrics import Rate, summarize


def _score(
    question: int,
    *,
    fast: bool,
    official: bool | None = None,
    hardness: Hardness = "easy",
    sample: int = 0,
    valid: bool = True,
) -> Score:
    return Score(
        example_id=f"spider-dev-{question:05d}",
        sample_idx=sample,
        hardness=hardness,
        valid=valid,
        ex_fast=fast,
        ex_official=official,
    )


@pytest.mark.phase(2)
def test_ex_by_hardness() -> None:
    scores = [
        _score(0, fast=True, official=True),
        _score(1, fast=True, official=True),
        _score(2, fast=True, official=False),
        _score(3, fast=False, official=False),
        _score(4, fast=True, official=True, hardness="medium"),
        _score(5, fast=False, official=False, hardness="medium"),
        _score(6, fast=False, official=False, hardness="extra"),
        _score(7, fast=False, official=False, hardness="extra", valid=False),
        _score(8, fast=False, official=False, hardness="extra", valid=False),
    ]
    summary = summarize(scores)
    assert (summary.n, summary.questions) == (9, 9)
    assert (summary.valid.hits, summary.valid.value) == (7, 7 / 9)
    assert (summary.ex_fast.hits, summary.ex_fast.value) == (4, 4 / 9)
    assert summary.ex_official is not None
    assert (summary.ex_official.hits, summary.ex_official.value) == (3, 3 / 9)

    # Only the levels with questions, in the order easy to extra.
    assert list(summary.by_hardness) == ["easy", "medium", "extra"]
    counts = {
        level: (row.n, row.questions, row.ex_fast.hits, row.ex_official.hits)
        for level, row in summary.by_hardness.items()
        if row.ex_official is not None
    }
    assert counts == {"easy": (4, 4, 3, 2), "medium": (2, 2, 1, 1), "extra": (3, 3, 0, 0)}
    assert summary.by_hardness["easy"].ex_fast.value == 0.75
    assert summary.by_hardness["extra"].ex_fast == Rate(value=0.0, low=0.0, high=0.0, hits=0)

    # A level's intervals come from its own questions alone, and the input order is moot.
    easy = summarize(scores[:4])
    assert summary.by_hardness["easy"].ex_fast == easy.ex_fast
    assert summary.by_hardness["easy"].ex_official == easy.ex_official
    assert summarize(reversed(scores)) == summary

    # One score without an official verdict leaves official EX out everywhere.
    partial = summarize([*scores[:-1], scores[-1].model_copy(update={"ex_official": None})])
    assert partial.ex_official is None
    assert all(row.ex_official is None for row in partial.by_hardness.values())
    assert partial.ex_fast == summary.ex_fast


@pytest.mark.phase(2)
def test_bootstrap_ci_contains_estimate() -> None:
    for n in (1, 2, 3, 8, 30):
        for hits in range(n + 1):
            rate = summarize([_score(q, fast=q < hits) for q in range(n)]).ex_fast
            assert (rate.hits, rate.value) == (hits, hits / n)
            assert 0.0 <= rate.low <= rate.value <= rate.high <= 1.0

    # A 95% interval: about 1.96 standard errors either side of a half on 400 questions,
    # where a 90% one would be 0.008 narrower at each end.
    scores = [_score(q, fast=q % 2 == 0) for q in range(400)]
    rate = summarize(scores).ex_fast
    half_width = 1.96 * math.sqrt(0.5 * 0.5 / 400)
    assert rate.low == pytest.approx(0.5 - half_width, abs=0.005)
    assert rate.high == pytest.approx(0.5 + half_width, abs=0.005)
    # Pinned, since reports rebuilt from stored scores must repeat their numbers: another
    # seed, resample count or percentile method would show here.
    assert (rate.low, rate.high) == (0.45, 0.5475625)
    shuffled = list(scores)
    random.Random(0).shuffle(shuffled)
    assert summarize(shuffled).ex_fast == rate


@pytest.mark.phase(2)
def test_bootstrap_resamples_whole_questions() -> None:
    single = [_score(q, fast=q % 3 == 0, hardness="hard") for q in range(30)]
    # Each question answered twice, the same way: twice the samples, no more evidence.
    double = [*single, *(score.model_copy(update={"sample_idx": 1}) for score in single)]
    one, two = summarize(single), summarize(double)
    assert (two.n, two.questions) == (60, 30)
    assert (two.by_hardness["hard"].n, two.by_hardness["hard"].questions) == (60, 30)
    assert two.ex_fast == one.ex_fast.model_copy(update={"hits": 20})
    assert one.ex_fast.low < one.ex_fast.value < one.ex_fast.high


@pytest.mark.phase(2)
def test_summarize_refuses_bad_scores() -> None:
    with pytest.raises(ValueError, match="no scores"):
        summarize([])
    score = _score(0, fast=True)
    with pytest.raises(ValueError, match="spider-dev-00000 sample 0 is scored twice"):
        summarize([score, score])
    with pytest.raises(ValueError, match="scored as both easy and hard"):
        summarize([score, _score(0, fast=True, hardness="hard", sample=1)])
