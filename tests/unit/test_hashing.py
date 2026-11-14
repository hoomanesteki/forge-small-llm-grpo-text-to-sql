from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from forge.core.hashing import canonical_json


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
