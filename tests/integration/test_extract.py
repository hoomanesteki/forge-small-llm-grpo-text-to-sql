"""The fallback line rule never drops a real query: every gold query, on one line, comes back."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.core.types import Example
from forge.data.spider import load_raw
from forge.sql.extract import extract_sql


def _not_extracted(examples: list[Example]) -> list[str]:
    """Ids of gold queries that a reply without a code block would lose."""
    lost = []
    for example in examples:
        line = " ".join(example.gold_sql.split())
        reply = f"Here is the query:\n{line}\nIt answers the question."
        if extract_sql(reply) != line.removesuffix(";").rstrip():
            lost.append(example.id)
    return lost


@pytest.mark.phase(1)
def test_fixture_golds_survive_the_fallback(spider_mini: Path) -> None:
    examples = load_raw()
    assert len(examples) == 60
    assert _not_extracted(examples) == []


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_every_spider_gold_survives_the_fallback() -> None:
    # Gold SQL only, no model, so the dev golds here are not a dev use (ADR-0004).
    examples = load_raw()
    assert len(examples) == 9693
    assert _not_extracted(examples) == []
