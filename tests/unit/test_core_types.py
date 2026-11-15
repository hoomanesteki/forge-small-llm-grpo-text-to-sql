from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from forge.core.types import (
    Example,
    ExecResult,
    Generation,
    GenParams,
    Prompt,
    RunManifest,
    Score,
)


def _example(**overrides: Any) -> Example:
    fields: dict[str, Any] = {
        "id": "spider-train-00042",
        "db_id": "concert_singer",
        "question": "How many singers do we have?",
        "gold_sql": "SELECT count(*) FROM singer",
        "hardness": "easy",
        "split": "train",
    }
    return Example(**{**fields, **overrides})


def _manifest(**overrides: Any) -> RunManifest:
    fields: dict[str, Any] = {
        "run_id": "2026-11-03-qwen3-4b-grpo-a1b2c3d",
        "kind": "grpo",
        "git_sha": "a1b2c3d" + "0" * 33,
        "config": {"lr": 5e-6, "lora": {"r": 16}},
        "config_hash": "0" * 64,
        "data_manifest_hash": "unset",
        "model_ref": "Qwen/Qwen3-4B",
        "platform": "kaggle",
        "seed": 42,
        "started_at": datetime(2026, 11, 3, 9, 30, tzinfo=UTC),
    }
    return RunManifest(**{**fields, **overrides})


@pytest.mark.phase(1)
def test_example_is_frozen() -> None:
    example = _example()
    with pytest.raises(ValidationError, match="frozen"):
        example.question = "How many concerts?"
    assert example.question == "How many singers do we have?"

    # A changed copy is the only way to vary an example, and it leaves the original alone.
    harder = example.model_copy(update={"hardness": "hard"})
    assert (example.hardness, harder.hardness) == ("easy", "hard")

    # Frozen examples hash by value, so they can key caches and dedupe sets.
    assert hash(example) == hash(_example())
    assert len({example, _example(), harder}) == 2


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (Prompt(example_id="e1", messages=(), prompt_version="p1"), "prompt_version", "p2"),
        (GenParams(), "temperature", 0.7),
    ],
)
def test_other_frozen_contracts_reject_assignment(model: BaseModel, field: str, value: Any) -> None:
    before = getattr(model, field)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(model, field, value)
    assert getattr(model, field) == before


@pytest.mark.phase(1)
def test_result_records_stay_mutable() -> None:
    # Runners fill these in as work completes, so the contract leaves them unfrozen.
    gen = Generation(
        example_id="e1",
        model_ref="Qwen/Qwen3-4B",
        text="```sql\nSELECT 1\n```",
        sql=None,
        tokens_in=120,
        tokens_out=9,
        latency_ms=41.5,
    )
    gen.sql = "SELECT 1"
    score = Score(example_id="e1", hardness="easy", valid=True, ex_fast=True)
    score.ex_official = False
    assert (gen.sql, score.ex_official) == ("SELECT 1", False)


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    "build",
    [
        lambda: _example(hardness="very hard"),
        lambda: _example(split="test"),
        lambda: ExecResult(ok=False, error_kind="crash"),
        lambda: _manifest(kind="train"),
        lambda: _manifest(platform="aws"),
        lambda: _manifest(status="done"),
    ],
    ids=["hardness", "split", "error_kind", "kind", "platform", "status"],
)
def test_literal_fields_reject_unknown_values(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()


@pytest.mark.phase(1)
def test_gen_params_defaults_are_greedy() -> None:
    assert GenParams().model_dump() == {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 256,
        "n": 1,
        "seed": 0,
    }


@pytest.mark.phase(1)
def test_exec_result_rows_survive_a_json_round_trip() -> None:
    # Gold results get cached as JSON; rows must come back as tuples to compare equal.
    result = ExecResult(ok=True, columns=["name", "age"], rows=[("Joe", 52), ("Rose", None)])
    restored = ExecResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.rows is not None
    assert all(isinstance(row, tuple) for row in restored.rows)


@pytest.mark.phase(1)
def test_run_manifest_round_trips_and_keeps_outputs_separate() -> None:
    first = _manifest()
    restored = RunManifest.model_validate_json(first.model_dump_json())
    assert restored == first
    assert restored.started_at.tzinfo is not None
    assert (restored.status, restored.outputs) == ("running", {})

    first.outputs["metrics"] = "runs/x/metrics.jsonl"
    assert _manifest().outputs == {}


@pytest.mark.phase(2)
def test_scores_stored_before_retry_read_as_one_reply() -> None:
    # Lines from runs made before FG-206 lack the retry fields (ADR-0015).
    line = '{"example_id":"e1","hardness":"easy","valid":true,"ex_fast":true,"ex_official":true}'
    old = Score.model_validate_json(line)
    assert (old.attempts, old.first_try_ex) == (1, None)
    retried = old.model_copy(update={"attempts": 2, "first_try_ex": False})
    assert Score.model_validate_json(retried.model_dump_json()) == retried
