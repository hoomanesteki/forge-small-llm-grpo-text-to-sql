from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from forge.core.hashing import canonical_json, sha256_hex
from forge.core.types import Example, RunManifest


class _Schedule(BaseModel):
    lr_by_step: dict[int, float]


@pytest.mark.phase(0)
def test_rejects_types_without_a_stable_encoding() -> None:
    class Opaque:
        pass

    with pytest.raises(TypeError):
        canonical_json({"x": Opaque()})
    with pytest.raises(TypeError):
        canonical_json({"x": {1, "a"}})


@pytest.mark.phase(0)
def test_encodes_paths_dates_and_sets_stably() -> None:
    a = canonical_json({"p": Path("data/spider"), "d": date(2026, 11, 14), "s": {"b", "a"}})
    assert a == b'{"d":"2026-11-14","p":"data/spider","s":["a","b"]}'


@pytest.mark.phase(1)
def test_canonical_json_stable() -> None:
    a = {"lora": {"r": 16, "alpha": 32}, "lr": 5e-6, "rewards": [{"w": 1.0, "name": "exec"}]}
    b = {"rewards": [{"name": "exec", "w": 1.0}], "lr": 5e-6, "lora": {"alpha": 32, "r": 16}}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a) == (
        b'{"lora":{"alpha":32,"r":16},"lr":5e-06,"rewards":[{"name":"exec","w":1.0}]}'
    )
    assert sha256_hex(canonical_json(a)) == sha256_hex(canonical_json(b))

    # Lists are ordered data (a curriculum, LoRA targets), so their order must count.
    assert canonical_json({"x": [1, 2]}) != canonical_json({"x": [2, 1]})
    # Non-ASCII text stays as UTF-8 rather than escapes, so hashes match what is on disk.
    assert canonical_json({"q": "café"}) == '{"q":"café"}'.encode()

    # A contract model hashes the same as its fields in any order.
    example = Example(
        id="spider-train-00042",
        db_id="concert_singer",
        question="How many singers do we have?",
        gold_sql="SELECT count(*) FROM singer",
        hardness="easy",
        split="train",
    )
    shuffled = dict(reversed(list(example.model_dump().items())))
    assert canonical_json(example) == canonical_json(shuffled)


@pytest.mark.phase(1)
def test_model_hashes_like_its_own_dump() -> None:
    manifest = RunManifest(
        run_id="2026-11-14-qwen3-4b-eval-a1b2c3d",
        kind="eval",
        git_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        config={"data": Path("data/spider"), "on": date(2026, 11, 14)},
        config_hash="0" * 64,
        data_manifest_hash="1" * 64,
        model_ref="Qwen/Qwen3-4B",
        platform="local-cpu",
        seed=0,
        started_at=datetime(2026, 11, 14, 9, 30, tzinfo=UTC),
    )

    encoded = canonical_json(manifest)
    assert encoded == canonical_json(manifest.model_dump())
    assert canonical_json({"run": manifest}) == canonical_json({"run": manifest.model_dump()})
    # A datetime reads as isoformat gives it wherever it sits, never with pydantic's "Z".
    assert b'"started_at":"2026-11-14T09:30:00+00:00"' in encoded
    assert b'"config":{"data":"data/spider","on":"2026-11-14"}' in encoded


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    "obj",
    [
        {"lr_schedule": {2: 1e-4, 10: 1e-5}},
        {"buckets": [{1.5: "a", 10.0: "b"}], "flags": {True: 1, None: 0}},
        # Mixed keys sort like any others once they are strings.
        {10: "ten", "9": "nine", 1: "one"},
        {"schedule": _Schedule(lr_by_step={500: 1e-4, 2000: 1e-5})},
    ],
)
def test_canonical_json_survives_a_json_round_trip(obj: dict[Any, Any]) -> None:
    # A manifest stores the config as this JSON read back, with every key a string, so
    # both must give the same bytes: keys sort as the strings JSON writes.
    encoded = canonical_json(obj)
    assert canonical_json(json.loads(encoded)) == encoded


@pytest.mark.phase(1)
def test_int_keys_sort_as_their_json_strings() -> None:
    assert canonical_json({2: "a", 10: "b"}) == b'{"10":"b","2":"a"}'
    assert canonical_json({2: "a", 10: "b"}) == canonical_json({"2": "a", "10": "b"})


@pytest.mark.phase(1)
def test_keys_that_json_writes_alike_are_refused() -> None:
    # Both would be written as "1", and reading the JSON back would keep only one of them.
    with pytest.raises(TypeError, match="'1'"):
        canonical_json({"x": {1: "int", "1": "str"}})
