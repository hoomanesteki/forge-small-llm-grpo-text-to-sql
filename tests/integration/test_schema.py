"""Normalizing a schema loses nothing: each statement rebuilds the table it came from."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from forge.data.schema import SAMPLE_PREFIX, schema_tables, schema_text
from forge.data.spider import resolve_data_dir


def _databases(root: Path) -> list[str]:
    paths = root.glob("database/*/*.sqlite")
    return sorted(path.parent.name for path in paths if path.stem == path.parent.name)


def _shape(conn: sqlite3.Connection, table: str) -> dict[str, list[tuple[object, ...]]]:
    """Columns (types, defaults, NOT NULL, key order), foreign keys and key indexes."""
    quoted = '"' + table.replace('"', '""') + '"'
    # Explicit CREATE INDEX statements (origin "c") are not part of the table's DDL, and
    # the seq column only numbers the list, so neither counts.
    indexes = [row[1:] for row in conn.execute(f"PRAGMA index_list({quoted})") if row[3] != "c"]
    return {
        "columns": conn.execute(f"PRAGMA table_xinfo({quoted})").fetchall(),
        "foreign_keys": conn.execute(f"PRAGMA foreign_key_list({quoted})").fetchall(),
        "indexes": sorted(indexes),
    }


def _assert_schemas_rebuild(root: Path) -> int:
    databases = _databases(root)
    assert databases, f"no databases under {root}"
    rebuilt = 0
    for db_id in databases:
        tables = schema_tables(db_id, "p1", data_dir=root)
        source = root / "database" / db_id / f"{db_id}.sqlite"
        with (
            closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as orig,
            closing(sqlite3.connect(":memory:")) as copy,
        ):
            for table in tables:
                copy.execute(table.create)
            for table in tables:
                assert _shape(copy, table.name) == _shape(orig, table.name), (db_id, table.name)
                rebuilt += 1
        # p2 is p1 plus its sample lines; reading every database's rows must not fail.
        p2 = schema_text(db_id, "p2", data_dir=root).splitlines()
        assert [line for line in p2 if not line.startswith(SAMPLE_PREFIX)] == [
            table.create for table in tables
        ]
    return rebuilt


@pytest.mark.phase(1)
def test_fixture_schemas_rebuild_the_same_tables(spider_mini: Path) -> None:
    assert _assert_schemas_rebuild(spider_mini) == 26


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_every_spider_schema_rebuilds_the_same_tables() -> None:
    # All 166 databases; sqlite_sequence in three of them is left out.
    assert _assert_schemas_rebuild(resolve_data_dir()) == 873
