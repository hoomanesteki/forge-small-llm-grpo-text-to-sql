"""Run model-written SQL on Spider's SQLite databases without risk to them or to the caller.

Every connection to a Spider database is made by ``connect_readonly`` (CONTRIBUTING
rule 9). It is read-only and immutable, so SQLite never writes, locks or journals the
file, and its authorizer is default-deny: only ``SQLITE_SELECT``, ``SQLITE_READ``,
``SQLITE_FUNCTION`` and ``SQLITE_RECURSIVE`` pass, so writes, DDL, ``ATTACH``, ``VACUUM``,
``PRAGMA`` and transactions fail as they are prepared, before anything runs.

``SqliteExecutor.run`` adds the rest and never raises for bad SQL: a deadline checked
from a progress handler, a row cap, and one statement per call. Each failure comes back
as an ``ExecResult`` with an ``ErrorKind``:

- ``denied``: the authorizer refused the statement, there was more than one statement,
  or the statement returns no columns, so it isn't a query. Statements are counted here
  before SQLite sees them. ``SELECT 1;`` and trailing comments are one statement; a stray
  ``;`` after the first statement makes two, as it does for ``sqlite3``. A ``CREATE
  TRIGGER`` body counts as several, which changes nothing because a trigger is a write.
  A write that names a missing table fails with ``no_such_table`` before the authorizer
  is asked.
- ``timeout``: the deadline passed while the query ran or its rows were fetched.
- ``syntax``, ``no_such_table``, ``no_such_column``: from SQLite's error message.
- ``empty_sql``: nothing but whitespace, comments and semicolons.
- ``other``: any other error, such as an unknown function or a ``?`` placeholder.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from forge.core.types import ErrorKind, ExecResult
from forge.data.spider import DATA_DIR_ENV, DataNotFound, db_path, resolve_data_dir

DEFAULT_TIMEOUT_MS = 2000
DEFAULT_MAX_ROWS = 10_000
# SQLite VM instructions between deadline checks. 1,000 take microseconds, so a query
# stops within a millisecond of its deadline, and checking costs next to nothing.
PROGRESS_STEPS = 1000

ALLOWED_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)

MULTIPLE_STATEMENTS = "more than one statement; only a single statement may run"
NOT_A_QUERY = "not a query; only statements that return rows may run"
EMPTY_SQL = "no SQL"

# How sqlite3 sees a string of SQL: whitespace and comments (``gap``), statement ends,
# and everything else, with quoted strings and names kept whole so a ``;`` inside them
# doesn't end anything. SQLite runs an unclosed block comment to the end of the input,
# and an unclosed quote is an error SQLite reports itself, so both run to the end here.
# Whitespace is the five characters sqlite3 skips between statements.
_LEXEME = re.compile(
    r"""
    (?P<gap>[ \t\n\f\r]+|--[^\n]*|/\*.*?(?:\*/|\Z))
    |(?P<end>;)
    |(?P<text>'(?:[^']|'')*'?|"(?:[^"]|"")*"?|`(?:[^`]|``)*`?|\[[^\]]*\]?
        |[^ \t\n\f\r;'"`\[/-]+|[/-])
    """,
    re.VERBOSE | re.DOTALL,
)

# Whole-message patterns, so a name quoted inside a message can't change its kind.
_ERROR_KINDS: tuple[tuple[re.Pattern[str], ErrorKind], ...] = (
    (re.compile(r"not authorized|authorization denied"), "denied"),
    # sqlite3's own refusal of a second statement, should one ever get past
    # statement_count.
    (re.compile(r"You can only execute one statement at a time\."), "denied"),
    (re.compile(r"no such table: .*", re.DOTALL), "no_such_table"),
    (re.compile(r"no such column: .*", re.DOTALL), "no_such_column"),
    (
        re.compile(r'near ".*": syntax error|unrecognized token: .*|incomplete input', re.DOTALL),
        "syntax",
    ),
)


class SqlExecutor(Protocol):
    """Runs one query on one database and reports the outcome; never raises for bad SQL."""

    def run(
        self,
        db_id: str,
        sql: str,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> ExecResult: ...


def connect_readonly(path: Path) -> sqlite3.Connection:
    """Open a Spider database so that nothing done through the connection can change it."""
    # Checked here because a read-only open of a missing file only fails at the first
    # query, where it would look like an error in the SQL.
    if not path.is_file():
        raise DataNotFound(
            f"{path} not found; run `forge data fetch`, or point {DATA_DIR_ENV} at a Spider "
            "data directory"
        )
    # immutable=1 also skips locking and change detection, which is safe because nothing
    # writes Spider files while they are in use.
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    conn.set_authorizer(_authorize)
    # A few Spider rows are not valid UTF-8; they read with replacement characters.
    conn.text_factory = _decode
    return conn


def statement_count(sql: str) -> int:
    """How many statements ``sqlite3`` would find in ``sql``: 0 when there is nothing to run.

    Semicolons before the first statement are skipped, as SQLite skips them. After the
    first statement only whitespace and comments may follow, so a stray ``;`` there
    counts as a statement of its own.
    """
    count = 0
    in_statement = False
    for match in _LEXEME.finditer(sql):
        kind = match.lastgroup
        if kind == "gap":
            continue
        if kind == "end":
            if in_statement:
                in_statement = False
            elif count:
                count += 1
        elif not in_statement:
            count += 1
            in_statement = True
    return count


def classify_error(message: str) -> ErrorKind:
    """The ``ErrorKind`` of a SQLite error message, ``other`` when no pattern fits."""
    for pattern, kind in _ERROR_KINDS:
        if pattern.fullmatch(message):
            return kind
    return "other"


class SqliteExecutor:
    """``SqlExecutor`` over ``database/<db_id>/<db_id>.sqlite`` in a Spider data directory.

    Connections are opened on first use and kept, one per database, for the life of the
    executor; ``close`` (or leaving a ``with`` block) closes them.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        # Absolute, so pool workers find the same files whatever their working directory.
        self.data_dir = resolve_data_dir(data_dir).resolve()
        self._connections: dict[str, sqlite3.Connection] = {}

    def run(
        self,
        db_id: str,
        sql: str,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> ExecResult:
        """Run one statement; a missing database raises ``DataNotFound``."""
        _check_limits(timeout_ms, max_rows)
        conn = self._connection(db_id)
        start = time.perf_counter()
        statements = statement_count(sql)
        if statements == 0:
            return ExecResult(ok=False, error=EMPTY_SQL, error_kind="empty_sql", ms=_ms(start))
        if statements > 1:
            return ExecResult(
                ok=False, error=MULTIPLE_STATEMENTS, error_kind="denied", ms=_ms(start)
            )

        deadline = start + timeout_ms / 1000
        timed_out = False

        def past_deadline() -> bool:
            nonlocal timed_out
            timed_out = time.perf_counter() >= deadline
            return timed_out  # true aborts the statement with "interrupted"

        conn.set_progress_handler(past_deadline, PROGRESS_STEPS)
        cursor: sqlite3.Cursor | None = None
        try:
            cursor = conn.execute(sql)
            if cursor.description is None:
                # SQLite runs DROP ... IF EXISTS on a missing object as a no-op without
                # asking the authorizer; reported as ok, it would pass for an empty result.
                return ExecResult(ok=False, error=NOT_A_QUERY, error_kind="denied", ms=_ms(start))
            rows = cursor.fetchmany(max_rows + 1)
            columns = [column[0] for column in cursor.description]
        except (sqlite3.Error, sqlite3.Warning) as exc:
            # Python 3.11's sqlite3 raises Warning, not Error, for a second statement.
            if timed_out:
                return ExecResult(
                    ok=False,
                    error=f"timed out after {timeout_ms} ms",
                    error_kind="timeout",
                    ms=_ms(start),
                )
            return ExecResult(
                ok=False, error=str(exc), error_kind=classify_error(str(exc)), ms=_ms(start)
            )
        except ValueError as exc:
            # The string can't be handed to SQLite at all, such as a lone surrogate, which
            # has no UTF-8 encoding.
            return ExecResult(ok=False, error=str(exc), error_kind="other", ms=_ms(start))
        finally:
            if cursor is not None:
                cursor.close()
            conn.set_progress_handler(None, 0)
        return ExecResult(
            ok=True,
            columns=columns,
            rows=rows[:max_rows],
            truncated=len(rows) > max_rows,
            ms=_ms(start),
        )

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _connection(self, db_id: str) -> sqlite3.Connection:
        conn = self._connections.get(db_id)
        if conn is None:
            conn = self._connections[db_id] = connect_readonly(db_path(db_id, self.data_dir))
        return conn


def _authorize(
    action: int, arg1: str | None, arg2: str | None, db_name: str | None, source: str | None
) -> int:
    # Default deny: an action that a newer SQLite adds stays refused until listed here.
    return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _check_limits(timeout_ms: int, max_rows: int) -> None:
    if timeout_ms <= 0:
        raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")
    if max_rows < 0:
        raise ValueError(f"max_rows must be zero or more, got {max_rows}")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
