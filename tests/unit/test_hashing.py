from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from forge.core.hashing import canonical_json, sha256_hex
from forge.core.types import Example


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
