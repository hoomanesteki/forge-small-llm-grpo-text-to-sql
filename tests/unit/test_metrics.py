"""Scores in aggregate: rates with bootstrap intervals, paired comparisons and pass@k."""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

import pytest

from forge.core.types import Hardness, Score
from forge.eval.metrics import Rate, compare, ex_metric, pass_at_k, summarize


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


def _runs(*, both: int, only_a: int, only_b: int, neither: int) -> tuple[list[Score], list[Score]]:
    """Two runs over the same questions whose official verdicts make the given table."""
    outcomes = (
        [(True, True)] * both
        + [(True, False)] * only_a
        + [(False, True)] * only_b
        + [(False, False)] * neither
    )
    # Fast EX says the opposite throughout, so counting the wrong one can't go unseen.
    a = [_score(q, fast=not right, official=right) for q, (right, _) in enumerate(outcomes)]
    b = [_score(q, fast=not right, official=right) for q, (_, right) in enumerate(outcomes)]
    return a, b


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

    a, b = _runs(both=20, only_a=3, only_b=9, neither=8)
    delta = compare(a, b).delta
    assert delta.value == pytest.approx(6 / 40)
    assert delta.low <= delta.value <= delta.high


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
def test_mcnemar_known_table() -> None:
    a, b = _runs(both=5, only_a=1, only_b=9, neither=5)
    # With no difference between the runs each of the 10 discordant pairs favors b with
    # chance 1/2, so the two-sided exact p is twice the chance of at most 1 favoring a.
    reference = 2 * sum(math.comb(10, i) for i in range(2)) / 2**10
    assert reference == 0.021484375

    result = compare(a, b)
    assert result.metric == "ex_official"
    table = result.mcnemar
    assert (table.both, table.only_a, table.only_b, table.neither) == (5, 1, 9, 5)
    assert table.pvalue == pytest.approx(reference, rel=1e-12)
    # Two-sided: swapping the runs mirrors the table and keeps p.
    swapped = compare(b, a).mcnemar
    assert (swapped.only_a, swapped.only_b) == (9, 1)
    assert swapped.pvalue == pytest.approx(reference, rel=1e-12)

    # 0 against 5 gives 2 / 2**5; an even split can't reject anything.
    one_sided = compare(*_runs(both=3, only_a=0, only_b=5, neither=2)).mcnemar
    assert one_sided.pvalue == pytest.approx(2 / 2**5, rel=1e-12)
    assert compare(*_runs(both=4, only_a=3, only_b=3, neither=0)).mcnemar.pvalue == 1.0


@pytest.mark.phase(2)
def test_compare_pairs_samples() -> None:
    # b fixes 3 of a's 30 misses and breaks nothing, so no resample can favor a.
    a, b = _runs(both=30, only_a=0, only_b=3, neither=27)
    result = compare(a, b)
    assert (result.n, result.questions) == (60, 60)
    assert (result.ex_a, result.ex_b) == (30 / 60, 33 / 60)
    assert result.delta.value == pytest.approx(3 / 60)
    assert 0.0 <= result.delta.low <= result.delta.value <= result.delta.high
    # Pairing sees what two separate intervals can't: the change is far tighter than either.
    alone = summarize(a).ex_official
    assert alone is not None
    assert result.delta.high - result.delta.low < (alone.high - alone.low) / 2

    same = compare(a, list(reversed(a)))
    assert same.delta.model_dump() == {"value": 0.0, "low": 0.0, "high": 0.0}
    assert same.mcnemar.pvalue == 1.0

    # Several samples per question: the change keeps its interval, but McNemar's test,
    # which needs independent pairs, has no p-value.
    twice_a = [*a, *(score.model_copy(update={"sample_idx": 1}) for score in a)]
    twice_b = [*b, *(score.model_copy(update={"sample_idx": 1}) for score in b)]
    sampled = compare(twice_a, twice_b)
    assert (sampled.n, sampled.questions, sampled.mcnemar.only_b) == (120, 60, 6)
    assert sampled.mcnemar.pvalue is None
    assert sampled.delta == result.delta


@pytest.mark.phase(2)
def test_compare_refuses_mismatched_samples() -> None:
    a = [_score(q, fast=True) for q in range(5)]
    b = [_score(q, fast=True) for q in range(1, 7)]
    with pytest.raises(
        ValueError,
        match=r"1 only in run a \(spider-dev-00000 sample 0\); 2 only in run b "
        r"\(spider-dev-00005 sample 0, spider-dev-00006 sample 0\)",
    ):
        compare(a, b)
    # The same questions with other samples don't pair either.
    with pytest.raises(ValueError, match="5 only in run a"):
        compare(a, [score.model_copy(update={"sample_idx": 1}) for score in a])
    with pytest.raises(ValueError, match="spider-dev-00001 sample 0 is scored twice"):
        compare(a, [*a, a[1]])


@pytest.mark.phase(2)
def test_official_ex_counts_when_every_score_has_it() -> None:
    a = [_score(q, fast=q == 0, official=q != 0) for q in range(4)]
    b = [_score(q, fast=True, official=True) for q in range(4)]
    assert ex_metric(a) == "ex_official"
    assert pass_at_k(a, 1) == 3 / 4
    assert compare(a, b).ex_a == 3 / 4

    # Without a verdict for every score, of either run, fast EX counts instead.
    b[2] = b[2].model_copy(update={"ex_official": None})
    assert ex_metric(b) == "ex_fast"
    result = compare(a, b)
    assert (result.metric, result.ex_a) == ("ex_fast", 1 / 4)
    a[3] = a[3].model_copy(update={"ex_official": None})
    assert pass_at_k(a, 1) == 1 / 4


@pytest.mark.phase(2)
def test_pass_at_k_unbiased() -> None:
    # Four questions of 5 samples each, with 0, 1, 2 and 5 of them right.
    right = [0, 1, 2, 5]
    scores = [
        _score(q, fast=sample < c, hardness="medium", sample=sample)
        for q, c in enumerate(right)
        for sample in range(5)
    ]
    for k in range(1, 6):
        # The estimator is unbiased for the share of k-sample draws, without replacement,
        # that hold a right answer; count every draw.
        draws = [
            Fraction(
                sum(any(s < c for s in draw) for draw in itertools.combinations(range(5), k)),
                math.comb(5, k),
            )
            for c in right
        ]
        assert draws == [1 - Fraction(math.comb(5 - c, k), math.comb(5, k)) for c in right]
        assert pass_at_k(scores, k) == pytest.approx(float(sum(draws) / len(draws)), rel=1e-12)

    # pass@1 is the share of right samples, pass@5 the share of questions with one.
    assert pass_at_k(scores, 1) == pytest.approx(8 / 20)
    assert pass_at_k(scores, 5) == 3 / 4
    assert pass_at_k(reversed(scores), 2) == pass_at_k(scores, 2)
    with pytest.raises(ValueError, match="pass@6 needs 6 samples of each question"):
        pass_at_k(scores, 6)
    with pytest.raises(ValueError, match="k of at least 1"):
        pass_at_k(scores, 0)


@pytest.mark.phase(2)
def test_summarize_refuses_bad_scores() -> None:
    with pytest.raises(ValueError, match="no scores"):
        summarize([])
    score = _score(0, fast=True)
    with pytest.raises(ValueError, match="spider-dev-00000 sample 0 is scored twice"):
        summarize([score, score])
    with pytest.raises(ValueError, match="scored as both easy and hard"):
        summarize([score, _score(0, fast=True, hardness="hard", sample=1)])
