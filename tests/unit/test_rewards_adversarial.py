"""Tricky completions and what GRPO's rewards pay for them (FG-502).

``tests/fixtures/adversarial_completions.yaml`` lists replies written to fool a checker,
each with the rewards it should get, and ``test_adversarial_cases`` scores them all.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from forge.core.types import ExecResult
from forge.eval.agreement import GOLD_TIMEOUT_MS
from forge.eval.gold_cache import MAX_ROWS
from forge.sql.executor import SqliteExecutor
from forge.sql.sandbox import SandboxPool
from forge.train.rewards import Rewards

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SPIDER_MINI = FIXTURES / "spider_mini"
CASES = FIXTURES / "adversarial_completions.yaml"

COMPONENTS = ("execution", "validity", "format", "guard")
CASE_KEYS = {
    "name",
    "db_id",
    "question",
    "gold",
    "completion",
    "expected",
    "audit_candidate",
    "note",
}
# The completions the story and the product owner asked for by kind.
REQUIRED = {
    "aliased_gold",
    "columns_reordered",
    "select_star_superset",
    "limit_1_luck",
    "empty_matches_empty",
    "correct_query_in_prose",
    "two_code_blocks_last_right",
    "think_block_draft_then_answer",
    "very_long_reply",
    "comment_with_instructions",
    "union_constant_trick",
    "literal_in_other_case",
    "comment_only_on_empty_gold",
    "empty_string",
    "semicolon_alone",
}
# FG-601's detectors, one of which each audit candidate's note names.
DETECTORS = (
    "constant_query",
    "empty_matches_empty",
    "limit_luck",
    "superset_projection",
    "fast_official_disagree",
    "format_without_sql",
    "comment_payload",
    "length_drift",
    "literal_hallucination",
)

Scores = tuple[float, float, float, float]


@dataclass(frozen=True)
class Case:
    """One completion of the fixture file, its gold query and the rewards it should get."""

    name: str
    db_id: str
    question: str
    gold: str
    completion: str
    expected: dict[str, float]
    audit_candidate: bool
    note: str


def _load_cases() -> list[Case]:
    """The fixture file's cases, each checked for the fields the file's header lists."""
    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"]
    loaded = [_case(raw) for raw in cases]
    names = [case.name for case in loaded]
    assert len(set(names)) == len(names), "case names must be unique"
    return loaded


def _case(raw: Mapping[str, object]) -> Case:
    """A case from its mapping in the file, with its padding put in."""
    name = raw.get("name")
    keys = set(raw)
    assert keys - {"pad"} == CASE_KEYS, f"{name}: fields {sorted(keys)}"
    assert isinstance(name, str) and re.fullmatch(r"[a-z0-9_]+", name), name
    expected = raw["expected"]
    assert isinstance(expected, dict), name
    assert set(expected) == {*COMPONENTS, "total"}, f"{name}: expected {sorted(expected)}"
    assert all(type(value) is float for value in expected.values()), f"{name}: floats only"
    assert type(raw["audit_candidate"]) is bool, name
    text = raw["completion"]
    assert isinstance(text, str), name
    if "pad" in raw:
        pad = raw["pad"]
        assert isinstance(pad, dict) and set(pad) == {"text", "times"}, name
        assert isinstance(pad["text"], str) and type(pad["times"]) is int, name
        assert text.count("{pad}") == 1, f"{name}: the completion needs one {{pad}}"
        text = text.replace("{pad}", pad["text"] * pad["times"])
    fields = {key: raw[key] for key in ("db_id", "question", "gold", "note")}
    assert all(isinstance(value, str) and value for value in fields.values()), name
    return Case(
        name=name,
        completion=text,
        expected=expected,
        audit_candidate=raw["audit_candidate"],
        **fields,
    )


@pytest.fixture(scope="module")
def pool() -> Iterator[SandboxPool]:
    with SandboxPool(2, data_dir=SPIDER_MINI) as pool:
        yield pool


def _batch(items: Sequence[tuple[str, str, str, str]]) -> dict[str, Any]:
    """TRL's arguments to a reward function for ``(example id, db id, gold query, reply)``
    items: conversational completions, and the dataset's columns as keywords."""
    return {
        "prompts": [[{"role": "user", "content": "q"}]] * len(items),
        "completions": [[{"role": "assistant", "content": reply}] for *_, reply in items],
        "completion_ids": None,
        "example_id": [example_id for example_id, _, _, _ in items],
        "db_id": [db_id for _, db_id, _, _ in items],
        "gold_sql": [gold for _, _, gold, _ in items],
    }


def _score(rewards: Rewards, batch: Mapping[str, Any]) -> list[Scores]:
    """Each completion's four rewards, the functions called as TRL calls them."""
    outputs = [fn(**batch) for fn in rewards.functions]
    return [(e, v, f, g) for e, v, f, g in zip(*outputs, strict=True)]


def _total(rewards: Rewards, scores: Scores) -> float:
    """The reward TRL trains on: the functions' rewards by their weights."""
    return sum(weight * value for weight, value in zip(rewards.weights, scores, strict=True))


def _gold(cases: Sequence[Case]) -> dict[str, ExecResult]:
    """Each case's gold result by case name, run in process as trusted SQL is."""
    results: dict[str, ExecResult] = {}
    with SqliteExecutor(SPIDER_MINI) as executor:
        for case in cases:
            result = executor.run(
                case.db_id, case.gold, timeout_ms=GOLD_TIMEOUT_MS, max_rows=MAX_ROWS
            )
            assert result.ok, f"{case.name}: the gold query fails: {result.error}"
            results[case.name] = result
    return results


@pytest.mark.phase(5)
def test_adversarial_cases(pool: SandboxPool) -> None:
    cases = _load_cases()
    assert len(cases) >= 20
    missing = REQUIRED - {case.name for case in cases}
    assert not missing, missing

    rewards = Rewards(_gold(cases), pool)
    items = [(case.name, case.db_id, case.gold, case.completion) for case in cases]
    scores = _score(rewards, _batch(items))

    wrong = []
    for case, got in zip(cases, scores, strict=True):
        expected = tuple(case.expected[name] for name in COMPONENTS)
        total = _total(rewards, got)
        if got != pytest.approx(expected) or total != pytest.approx(case.expected["total"]):
            wrong.append(
                f"{case.name}: expected {expected} = {case.expected['total']}, "
                f"got {got} = {total:.4g}"
            )
    assert not wrong, "\n".join(wrong)

    # The file's totals are its components by the weights, so neither can drift alone.
    for case in cases:
        components = tuple(case.expected[name] for name in COMPONENTS)
        weighted = _total(rewards, components)
        assert weighted == pytest.approx(case.expected["total"]), case.name
    # An audit candidate is a reply the rewards pay for, flagged for the detector that
    # looks for its kind in real samples.
    audit = [case for case in cases if case.audit_candidate]
    assert audit
    for case in audit:
        assert case.expected["total"] > 0, case.name
        assert any(f"FG-601's {detector}" in case.note for detector in DETECTORS), case.name
