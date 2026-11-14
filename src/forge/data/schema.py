"""Schema text for prompts: each table's ``CREATE TABLE`` statement, one line per table.

The statements are the database's own, read from ``sqlite_master``, so column types,
primary keys and foreign keys all come through as their authors wrote them. Comments
are removed and whitespace collapsed (``normalize_sql``). Prompt version ``p2`` follows
each table with up to three of its rows as ``-- sample:`` comments.

Results are cached per (data directory, database, version), so building thousands of
prompts reads each database once.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Collection, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, get_args

from forge.data.spider import DATA_DIR_ENV, DataNotFound, db_path, resolve_data_dir

PromptVersion = Literal["p1", "p2"]
PROMPT_VERSIONS: tuple[PromptVersion, ...] = get_args(PromptVersion)

SAMPLE_ROWS = 3
SAMPLE_CHARS = 50
SAMPLE_PREFIX = "-- sample: "

# Statements are split into whitespace and comments (``gap``), quoted strings and names,
# the punctuation whose spacing gets tidied, and everything else. ``bad`` is a quote or
# block comment that never closes.
_TOKEN = re.compile(
    r"""
    (?P<gap>\s+|--[^\n]*|/\*.*?\*/)
    |(?P<quoted>'(?:[^']|'')*'|"(?:[^"]|"")*"|`(?:[^`]|``)*`)
    |(?P<punct>[(),])
    |(?P<word>(?:[^\s(),'"`/-]|-(?!-)|/(?!\*))+)
    |(?P<bad>.)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass(frozen=True)
class Table:
    """One table of a schema, ready to print."""

    name: str
    create: str  # its CREATE TABLE statement, on one line
    stub: str  # the same statement with its column list cut to "(...)"
    samples: tuple[str, ...]  # "-- sample:" lines, p2 only


def check_version(version: str) -> PromptVersion:
    if version not in PROMPT_VERSIONS:
        raise ValueError(
            f"unknown prompt version {version!r}; expected one of {', '.join(PROMPT_VERSIONS)}"
        )
    return version


def schema_text(db_id: str, version: str, *, data_dir: Path | None = None) -> str:
    """The schema as it appears in a prompt of this version."""
    return _schema_text(_root(data_dir), db_id, check_version(version))


def schema_tables(db_id: str, version: str, *, data_dir: Path | None = None) -> tuple[Table, ...]:
    """The schema's tables in creation order, for prompts that must leave parts out."""
    return _tables(_root(data_dir), db_id, check_version(version))


def render_schema(
    tables: Sequence[Table], *, samples: bool = True, stubs: Collection[str] = ()
) -> str:
    """One line per table, each followed by its sample rows unless ``samples`` is False.

    Tables named in ``stubs`` are shown without their columns or samples.
    """
    lines: list[str] = []
    for table in tables:
        if table.name in stubs:
            lines.append(table.stub)
            continue
        lines.append(table.create)
        if samples:
            lines.extend(table.samples)
    return "\n".join(lines)


def normalize_sql(sql: str) -> str:
    """The statement on one line: comments removed and each run of whitespace one space.

    No space is kept after ``(`` or before ``)`` and ``,``. Quoted strings and names are
    left exactly as they are. Comments have to go: on one line, a ``--`` comment would
    swallow the rest of the statement.
    """
    parts: list[str] = []
    gap = False
    for match in _tokens(sql):
        if match.lastgroup == "gap":
            gap = True
            continue
        text = match.group()
        if gap and parts and parts[-1] != "(" and text not in (")", ","):
            parts.append(" ")
        parts.append(text)
        gap = False
    return "".join(parts)


def _tokens(sql: str) -> Iterator[re.Match[str]]:
    for match in _TOKEN.finditer(sql):
        if match.lastgroup == "bad":
            raise ValueError(f"unterminated quote or comment at offset {match.start()}: {sql!r}")
        yield match


def _stub(statement: str) -> str:
    for match in _tokens(statement):
        if match.lastgroup == "punct" and match.group() == "(":
            return statement[: match.start()].rstrip() + " (...)"
    raise ValueError(f"no column list in {statement!r}")


def _root(data_dir: Path | None) -> Path:
    # Resolved, so the same directory reached by two paths shares one cache entry.
    return resolve_data_dir(data_dir).resolve()


@cache
def _schema_text(root: Path, db_id: str, version: PromptVersion) -> str:
    return render_schema(_tables(root, db_id, version))


@cache
def _tables(root: Path, db_id: str, version: PromptVersion) -> tuple[Table, ...]:
    path = db_path(db_id, root)
    if not path.is_file():
        raise DataNotFound(
            f"database/{db_id}/{db_id}.sqlite not found in {root}; run `forge data fetch`, "
            f"or point {DATA_DIR_ENV} at a Spider data directory"
        )
    tables: list[Table] = []
    with closing(_connect_read_only(path)) as conn:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY rowid"
        ).fetchall()
        for name, sql in rows:
            # SQLite's own bookkeeping, such as sqlite_sequence for AUTOINCREMENT.
            if name.startswith("sqlite_"):
                continue
            create = normalize_sql(sql)
            samples = _sample_lines(conn, name) if version == "p2" else ()
            tables.append(Table(name=name, create=create, stub=_stub(create), samples=samples))
    if not tables:
        raise ValueError(f"{path} has no tables")
    return tuple(tables)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    # Read-only and immutable, so SQLite never writes, locks or journals a Spider file.
    # FG-106 moves this into forge.sql.executor, which then opens every connection.
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    # A few Spider rows are not valid UTF-8; they show with replacement characters.
    conn.text_factory = _decode
    return conn


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _sample_lines(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    quoted = '"' + table.replace('"', '""') + '"'
    rows = conn.execute(f"SELECT * FROM {quoted} LIMIT {SAMPLE_ROWS}").fetchall()
    return tuple(SAMPLE_PREFIX + "(" + ", ".join(map(_literal, row)) + ")" for row in rows)


def _literal(value: object) -> str:
    """A value as SQL would write it, on one line and cut to ``SAMPLE_CHARS``."""
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return "X" + _quoted(value.hex())
    if isinstance(value, str):
        # Line breaks inside a value would end the comment it sits in.
        return _quoted(" ".join(value.split()))
    return _cut(str(value))


def _quoted(text: str) -> str:
    return "'" + _cut(text).replace("'", "''") + "'"


def _cut(text: str) -> str:
    return text if len(text) <= SAMPLE_CHARS else text[:SAMPLE_CHARS] + "..."
