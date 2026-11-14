"""Schema text: normalized CREATE TABLE statements, p2 sample rows, caching, read-only access."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from forge.data import schema
from forge.data.schema import (
    SAMPLE_PREFIX,
    normalize_sql,
    render_schema,
    schema_tables,
    schema_text,
)
from forge.data.spider import DataNotFound

DEPARTMENT_MANAGEMENT_P1 = [
    'CREATE TABLE "department" ("Department_ID" int, "Name" text, "Creation" text, '
    '"Ranking" int, "Budget_in_Billions" real, "Num_Employees" real, '
    'PRIMARY KEY ("Department_ID"))',
    'CREATE TABLE "head" ("head_ID" int, "name" text, "born_state" text, "age" real, '
    'PRIMARY KEY ("head_ID"))',
    'CREATE TABLE "management" ("department_ID" int, "head_ID" int, "temporary_acting" text, '
    'PRIMARY KEY ("Department_ID","head_ID"), '
    'FOREIGN KEY ("Department_ID") REFERENCES `department`("Department_ID"), '
    'FOREIGN KEY ("head_ID") REFERENCES `head`("head_ID"))',
]

AIRCRAFT = (
    "CREATE TABLE aircraft(aid number(9,0) primary key, name varchar2(30), distance number(6,0))"
)


def _make_db(root: Path, db_id: str, *statements: str) -> Path:
    path = root / "database" / db_id / f"{db_id}.sqlite"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as conn:
        for statement in statements:
            conn.execute(statement)
    conn.close()
    return path


@pytest.mark.phase(1)
def test_schema_includes_pk_fk(spider_mini: Path) -> None:
    assert schema_text("department_management", "p1").splitlines() == DEPARTMENT_MANAGEMENT_P1

    # flight_1 declares its keys inline and in lower case; they come through as written.
    flight = schema_text("flight_1", "p1").splitlines()
    assert flight[0] == (
        "CREATE TABLE flight(flno number(4,0) primary key, origin varchar2(20), "
        "destination varchar2(20), distance number(6,0), departure_date date, "
        "arrival_date date, price number(7,2), aid number(9,0), "
        'foreign key("aid") references `aircraft`("aid"))'
    )
    assert flight[1] == AIRCRAFT
    assert flight[3] == (
        "CREATE TABLE certificate(eid number(9,0), aid number(9,0), primary key(eid,aid), "
        'foreign key("eid") references `employee`("eid"), '
        'foreign key("aid") references `aircraft`("aid"))'
    )


@pytest.mark.phase(1)
def test_p2_adds_sample_rows(spider_mini: Path) -> None:
    p1 = schema_text("flight_1", "p1")
    p2 = schema_text("flight_1", "p2")

    assert SAMPLE_PREFIX.strip() not in p1
    lines = p2.splitlines()
    # p2 is p1 with sample lines added under each table, and nothing else changed.
    assert [line for line in lines if not line.startswith(SAMPLE_PREFIX)] == p1.splitlines()
    start = lines.index(AIRCRAFT) + 1
    assert lines[start : start + 4] == [
        "-- sample: (1, 'Boeing 747-400', 8430)",
        "-- sample: (2, 'Boeing 737-800', 3383)",
        "-- sample: (3, 'Airbus A340-300', 7120)",
        "CREATE TABLE employee(eid number(9,0) primary key, name varchar2(30), "
        "salary number(10,2))",
    ]
    tables = schema_tables("flight_1", "p2")
    assert [len(table.samples) for table in tables] == [3, 3, 3, 3]
    assert all(not table.samples for table in schema_tables("flight_1", "p1"))
    # geo has no rows, so p2 adds nothing there.
    assert schema_text("geo", "p2") == schema_text("geo", "p1")


@pytest.mark.phase(1)
def test_sample_values_are_cut_and_kept_on_one_line(tmp_path: Path) -> None:
    _make_db(
        tmp_path,
        "odd",
        "CREATE TABLE notes (id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT, raw BLOB)",
        "INSERT INTO notes (body, raw) VALUES ('it''s\n' || replace(hex(zeroblob(30)), '0', 'x'),"
        " X'00ff')",
        "INSERT INTO notes (body, raw) VALUES (NULL, NULL)",
        "INSERT INTO notes (body, raw) VALUES (CAST(X'41ff42' AS TEXT), NULL)",
        "INSERT INTO notes (body, raw) VALUES ('fourth row', NULL)",
    )

    tables = schema_tables("odd", "p2", data_dir=tmp_path)

    # AUTOINCREMENT adds SQLite's own sqlite_sequence table, which is not part of the schema.
    assert [table.name for table in tables] == ["notes"]
    assert tables[0].samples == (
        # 'it''s xxx...': the newline becomes a space, and the value is cut at 50 characters
        # before its quote is doubled.
        "-- sample: (1, 'it''s " + "x" * 45 + "...', X'00ff')",
        "-- sample: (2, NULL, NULL)",
        # Text that is not valid UTF-8 is shown with a replacement character.
        "-- sample: (3, 'A�B', NULL)",
    )


@pytest.mark.phase(1)
def test_normalize_sql_removes_comments_and_collapses_whitespace() -> None:
    sql = (
        "CREATE TABLE  t (\n"
        "\tid int PRIMARY KEY, -- the key, (unbalanced\n"
        "  note text DEFAULT 'a  --  b' /* two\n lines */ ,\n"
        '  "two  words" int\n'
        ",  x number(4,0) DEFAULT -1\n"
        ")"
    )
    assert normalize_sql(sql) == (
        "CREATE TABLE t (id int PRIMARY KEY, note text DEFAULT 'a  --  b', "
        '"two  words" int, x number(4,0) DEFAULT -1)'
    )
    with pytest.raises(ValueError, match="unterminated"):
        normalize_sql("CREATE TABLE t (a text DEFAULT 'open)")
    with pytest.raises(ValueError, match="unterminated"):
        normalize_sql("CREATE TABLE t (a text /* open)")


@pytest.mark.phase(1)
def test_render_schema_can_drop_samples_and_columns(spider_mini: Path) -> None:
    tables = schema_tables("flight_1", "p2")

    assert render_schema(tables) == schema_text("flight_1", "p2")
    assert render_schema(tables, samples=False) == schema_text("flight_1", "p1")
    assert render_schema(tables, samples=False, stubs={"flight", "employee"}).splitlines() == [
        "CREATE TABLE flight (...)",
        AIRCRAFT,
        "CREATE TABLE employee (...)",
        tables[3].create,
    ]


@pytest.mark.phase(1)
def test_schema_is_cached_and_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fresh data directories, so no other test has filled their cache entries.
    db_id = "department_management"
    source = Path(__file__).resolve().parents[1] / "fixtures" / "spider_mini" / "database"
    first_dir, other_dir = tmp_path / "a", tmp_path / "b"
    for root in (first_dir, other_dir):
        (root / "database" / db_id).mkdir(parents=True)
        shutil.copy(source / db_id / f"{db_id}.sqlite", root / "database" / db_id)

    opened: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real_connect = sqlite3.connect

    def spy(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        opened.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(schema.sqlite3, "connect", spy)

    first = schema_text(db_id, "p1", data_dir=first_dir)
    assert schema_text(db_id, "p1", data_dir=first_dir) is first
    assert schema_text(db_id, "p1", data_dir=first_dir / "database" / "..") is first
    schema_tables(db_id, "p1", data_dir=first_dir)
    assert len(opened) == 1

    schema_text(db_id, "p2", data_dir=first_dir)
    assert len(opened) == 2

    # Another data directory is another entry, and FORGE_DATA_DIR is read on each call.
    monkeypatch.setenv("FORGE_DATA_DIR", str(other_dir))
    assert schema_text(db_id, "p1") == first
    assert len(opened) == 3

    for args, kwargs in opened:
        assert args[0].startswith("file:///")
        assert args[0].endswith(f"/{db_id}.sqlite?mode=ro&immutable=1")
        assert kwargs == {"uri": True}


@pytest.mark.phase(1)
def test_schema_errors(spider_mini: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown prompt version 'p3'"):
        schema_text("flight_1", "p3")
    with pytest.raises(DataNotFound, match="forge data fetch"):
        schema_text("no_such_db", "p1")
    _make_db(tmp_path, "empty")
    with pytest.raises(ValueError, match="has no tables"):
        schema_text("empty", "p1", data_dir=tmp_path)
