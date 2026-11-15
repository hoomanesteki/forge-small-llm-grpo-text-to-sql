"""Run model-written SQL on Spider's SQLite databases without risk to them or to the caller.

Every connection to a Spider database is made by ``connect_readonly`` (CONTRIBUTING
rule 9). It is read-only and immutable, so SQLite never writes, locks or journals the
file, and its authorizer is default-deny: only ``SQLITE_SELECT``, ``SQLITE_READ``,
``SQLITE_FUNCTION`` and ``SQLITE_RECURSIVE`` pass, so writes, DDL, ``ATTACH``, ``VACUUM``,
``PRAGMA`` and transactions fail as they are prepared, before anything runs. Its
``LIMITS``, measured against what Spider needs, keep the work in one row small: no string,
blob or table row over 4 kB, LIKE and GLOB patterns of at most 100 characters, SQL text of
at most 8 kB, at most 200 columns, and no attached databases, which also stops ``VACUUM
INTO``.

``SqliteExecutor.run`` adds the rest and never raises for bad SQL: one statement per
call, and only a query; a deadline; a row cap; and a byte budget for the rows it keeps.
Each failure comes back as an ``ExecResult`` with an ``ErrorKind``:

- ``denied``: there was more than one statement, the statement is not a query, or the
  authorizer refused it. Statements are counted and classified here, before SQLite sees
  them. ``SELECT 1;`` and trailing comments are one statement; a stray ``;`` after the
  first statement makes two, as it does for ``sqlite3``. A ``CREATE TRIGGER`` body counts
  as several, which changes nothing because a trigger is a write. A statement is a query
  when its keyword (``statement_keyword``) is ``SELECT`` or ``VALUES``; any other
  statement keyword is denied, even when SQLite would fail first on a missing name. One
  that returns no columns is denied too.
- ``timeout``: the deadline passed while the query ran or its rows were fetched. A
  progress handler checks it every ``PROGRESS_STEPS`` VM steps, and a watchdog thread
  interrupts SQLite at the deadline; the watchdog covers queries that run too few steps
  for the handler to fire. SQLite only looks at either between rows, so the work inside
  one row runs to the end. For ``instr``, ``replace``, ``LIKE`` and ``GLOB`` the limits
  hold that to a few hundredths of a second (measured), but other string functions are
  slower: one row of ``ltrim`` calls with a large character set, or ``unhex`` with a
  large ignore set, can run a second or two. So this deadline is soft and its overrun is
  not bounded in general; only a ``SandboxPool`` kill is a hard bound.
- ``syntax``, ``no_such_table``, ``no_such_column``: from SQLite's error message.
- ``empty_sql``: nothing but whitespace, comments and semicolons.
- ``other``: any other error, such as an unknown function, a ``?`` placeholder, a value
  or SQL text over a limit, or rows that take more than ``MAX_RESULT_BYTES`` of memory.

So the deadline of ``run`` is soft, and ``run`` is for trusted SQL: gold queries, schema
reads, tests. Untrusted SQL, anything a model wrote, goes through ``run_many`` or a
``forge.sql.sandbox.SandboxPool``, whose worker processes are killed at a hard deadline
(ADR-0014).
"""

from __future__ import annotations

import re
import sqlite3
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Protocol, Self

from forge.core.types import ErrorKind, ExecResult
from forge.data.spider import DATA_DIR_ENV, DataNotFound, db_path, resolve_data_dir

DEFAULT_TIMEOUT_MS = 2000
DEFAULT_MAX_ROWS = 10_000
# SQLite VM instructions between deadline checks. 1,000 ordinary ones take microseconds,
# and checking costs next to nothing; the watchdog in run covers steps that are slow.
PROGRESS_STEPS = 1000

ALLOWED_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)
# Measured on the full Spider data (ADR-0014). Its longest gold query is 577 bytes and its
# longest LIKE pattern 42 characters; its widest table has 48 columns and its widest gold
# result 19. Its largest value is 213 bytes, but SQLite reads each CREATE statement in
# sqlite_master through the length limit too, whenever it prepares a query, and soccer_1's
# is 1,191 bytes: every gold query runs at that limit, and soccer_1's fail one byte under
# it. The cost of instr, LIKE, GLOB or replace grows with the product of their inputs'
# lengths, so for those the length and SQL limits hold one row to a few hundredths of a
# second, where FG-191's 100 kB allowed 18 s. Not every function is that cheap (ltrim with
# a large character set, unhex with a large ignore set, run a second or two in one row), so
# the in-process deadline is soft; the SandboxPool kill is the hard bound (ADR-0014). With
# no attached database allowed, ATTACH and VACUUM INTO fail on any connection.
MAX_VALUE_BYTES = 4_096
LIMITS: Mapping[int, int] = MappingProxyType(
    {
        sqlite3.SQLITE_LIMIT_LENGTH: MAX_VALUE_BYTES,
        sqlite3.SQLITE_LIMIT_SQL_LENGTH: 8_192,
        sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH: 100,
        sqlite3.SQLITE_LIMIT_COLUMN: 200,
        sqlite3.SQLITE_LIMIT_ATTACHED: 0,
    }
)
# The memory the rows of one result may take, counted by row_size.
MAX_RESULT_BYTES = 64 * 2**20
# The most memory one value can take: a str holds 4 bytes a character once one of them
# is past U+FFFF, and SQLite counts at least one byte a character.
MAX_VALUE_SIZE = 4 * MAX_VALUE_BYTES + sys.getsizeof("\U00010000")

MULTIPLE_STATEMENTS = "more than one statement; only a single statement may run"
NOT_A_QUERY = "not a query; only statements that return rows may run"
NOT_A_QUERY_KEYWORD = "{keyword} is not a query; only SELECT and VALUES statements may run"
EMPTY_SQL = "no SQL"
RESULT_TOO_BIG = f"result too big: its rows take more than {MAX_RESULT_BYTES // 2**20} MiB"

QUERY_KEYWORDS = frozenset({"SELECT", "VALUES"})
# Every keyword a SQLite statement can start with.
STATEMENT_KEYWORDS = frozenset(
    {
        *("ALTER", "ANALYZE", "ATTACH", "BEGIN", "COMMIT", "CREATE", "DELETE", "DETACH"),
        *("DROP", "END", "EXPLAIN", "INSERT", "PRAGMA", "REINDEX", "RELEASE", "REPLACE"),
        *("ROLLBACK", "SAVEPOINT", "SELECT", "UPDATE", "VACUUM", "VALUES", "WITH"),
    }
)
# The statements a WITH clause can lead into.
_WITH_BODIES = frozenset({"SELECT", "VALUES", "INSERT", "REPLACE", "UPDATE", "DELETE"})

# How sqlite3 sees a string of SQL: whitespace, comments, statement ends, parentheses,
# quoted strings and names (kept whole, so a ``;`` inside them doesn't end anything), and
# other words. SQLite runs an unclosed block comment to the end of the input, and an
# unclosed quote is an error SQLite reports itself, so both run to the end here. SQLite
# reads whitespace as one of `` \t\n\f\r`` followed by any of those or ``\v``; a ``\v``
# anywhere else is an unrecognized token.
_LEXEME = re.compile(
    r"""
    (?P<space>[ \t\n\f\r][ \t\n\v\f\r]*)
    |(?P<comment>--[^\n]*|/\*.*?(?:\*/|\Z))
    |(?P<end>;)
    |(?P<open>\()
    |(?P<close>\))
    |(?P<quoted>'(?:[^']|'')*'?|"(?:[^"]|"")*"?|`(?:[^`]|``)*`?|\[[^\]]*\]?)
    |(?P<word>[^ \t\n\f\r;'"`\[()/-]+|[/-])
    """,
    re.VERBOSE | re.DOTALL,
)
_GAPS = frozenset({"space", "comment"})
# SQLite identifiers: ASCII letters, digits, ``_`` and ``$``, and any non-ASCII character.
_IDENTIFIER = re.compile("[A-Za-z_\u0080-\U0010ffff][A-Za-z0-9_$\u0080-\U0010ffff]*")

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
    for limit, value in LIMITS.items():
        conn.setlimit(limit, value)
    # A few Spider rows are not valid UTF-8; they read with replacement characters.
    conn.text_factory = _decode
    return conn


def statement_count(sql: str) -> int:
    """How many statements ``sqlite3`` would find in ``sql``: 0 when there is nothing to run.

    Semicolons before the first statement are skipped, as SQLite skips them. After the
    first statement only whitespace and comments may follow, so a stray ``;`` there
    counts as a statement of its own, and so does a ``\v``, which Python's check there
    never skips.
    """
    count = 0
    in_statement = False
    for match in _LEXEME.finditer(sql):
        kind = match.lastgroup
        if kind == "end":
            if in_statement:
                in_statement = False
            elif count:
                count += 1
            continue
        after_first = count > 0 and not in_statement
        if kind == "comment" or (kind == "space" and not (after_first and "\v" in match.group())):
            continue
        if not in_statement:
            count += 1
            in_statement = True
    return count


def strip_leading(sql: str) -> str:
    """``sql`` from its first statement on: the whitespace, comments and semicolons SQLite
    skips before it are gone."""
    for match in _LEXEME.finditer(sql):
        if match.lastgroup not in _GAPS and match.lastgroup != "end":
            return sql[match.start() :]
    return ""


def statement_keyword(sql: str) -> str | None:
    """The upper-case keyword that says what the first statement in ``sql`` does.

    That is the statement's first word or, after a ``WITH`` clause, the first ``SELECT``,
    ``VALUES``, ``INSERT``, ``REPLACE``, ``UPDATE`` or ``DELETE`` outside the parentheses
    of its common table expressions. ``None`` when the first word is no statement keyword,
    which SQLite reports as a syntax error, or nothing follows the ``WITH`` clause.
    """
    lexemes = (m for m in _LEXEME.finditer(strip_leading(sql)) if m.lastgroup not in _GAPS)
    first = next(lexemes, None)
    if first is None or first.lastgroup != "word":
        return None
    word = _IDENTIFIER.match(first.group())
    keyword = word.group().upper() if word is not None else None
    if keyword not in STATEMENT_KEYWORDS:
        return None
    if keyword != "WITH":
        return keyword
    depth = 0
    for match in lexemes:
        kind = match.lastgroup
        if kind == "end":
            break
        if kind == "open":
            depth += 1
        elif kind == "close":
            depth -= 1
        elif kind == "word" and depth <= 0:
            for name in _IDENTIFIER.finditer(match.group()):
                if name.group().upper() in _WITH_BODIES:
                    return name.group().upper()
    return None


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
        check_run_args(timeout_ms, max_rows)
        conn = self._connection(db_id)
        start = time.perf_counter()
        statements = statement_count(sql)
        if statements == 0:
            return ExecResult(ok=False, error=EMPTY_SQL, error_kind="empty_sql", ms=_ms(start))
        if statements > 1:
            return ExecResult(
                ok=False, error=MULTIPLE_STATEMENTS, error_kind="denied", ms=_ms(start)
            )
        keyword = statement_keyword(sql)
        if keyword is not None and keyword not in QUERY_KEYWORDS:
            # Decided here because SQLite resolves names before it asks the authorizer,
            # so a write on a missing table would come back as no_such_table.
            error = NOT_A_QUERY_KEYWORD.format(keyword=keyword)
            return ExecResult(ok=False, error=error, error_kind="denied", ms=_ms(start))

        deadline = start + timeout_ms / 1000
        # Both abort the statement with SQLITE_INTERRUPT. The progress handler runs every
        # PROGRESS_STEPS steps, which slow steps can stretch far past the deadline, so the
        # watchdog interrupts at the deadline itself. SQLite drops an interrupt that comes
        # before the statement starts, and then the progress handler still stops it.
        conn.set_progress_handler(lambda: time.perf_counter() >= deadline, PROGRESS_STEPS)
        watchdog = threading.Timer(timeout_ms / 1000, conn.interrupt)
        watchdog.start()
        cursor: sqlite3.Cursor | None = None
        try:
            cursor = conn.execute(sql)
            if cursor.description is None:
                # SQLite runs DROP ... IF EXISTS on a missing object as a no-op without
                # asking the authorizer; reported as ok, it would pass for an empty result.
                return ExecResult(ok=False, error=NOT_A_QUERY, error_kind="denied", ms=_ms(start))
            columns = [column[0] for column in cursor.description]
            rows = _fetch(cursor, len(columns), max_rows + 1)
            if rows is None:
                return ExecResult(ok=False, error=RESULT_TOO_BIG, error_kind="other", ms=_ms(start))
        except (sqlite3.Error, sqlite3.Warning) as exc:
            # Python 3.11's sqlite3 raises Warning, not Error, for a second statement, and
            # errors sqlite3 raises before calling SQLite carry no error code.
            if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
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
            # Joined, so a late interrupt can't land on the next query; one that already
            # fired is cleared when SQLite starts the next statement.
            watchdog.cancel()
            watchdog.join()
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

    def run_many(
        self,
        items: Sequence[tuple[str, str]],
        workers: int,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> list[ExecResult]:
        """Run ``(db_id, sql)`` pairs in a ``SandboxPool`` of ``workers`` processes.

        Results keep their order. This is the path for untrusted SQL: each query has a
        hard deadline, ``timeout_ms`` plus ``forge.sql.sandbox.KILL_GRACE_MS``, after
        which its worker is killed. The pool lives for this call only; code that runs
        batches again and again keeps its own ``SandboxPool``.
        """
        # Imported here because the sandbox builds on this module.
        from forge.sql.sandbox import SandboxPool

        check_run_args(timeout_ms, max_rows)
        with SandboxPool(workers, data_dir=self.data_dir) as pool:
            return pool.run_many(items, timeout_ms=timeout_ms, max_rows=max_rows)

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


def row_size(row: tuple[Any, ...]) -> int:
    """The bytes a fetched row holds: the tuple, its values and its slot in the list.

    A shared object such as ``None`` or a small int counts each time, so this errs high.
    """
    return sys.getsizeof(row) + 8 + sum(map(sys.getsizeof, row))


def _fetch(cursor: sqlite3.Cursor, width: int, limit: int) -> list[tuple[Any, ...]] | None:
    """Up to ``limit`` rows of ``width`` values, or ``None`` once they pass the budget."""
    # No row is bigger than this, so a batch holds at most a quarter of the budget and the
    # rows held never pass 1.25 times it, however big their values are.
    largest = sys.getsizeof((None,) * width) + 8 + width * MAX_VALUE_SIZE
    batch = max(1, MAX_RESULT_BYTES // (4 * largest))
    rows: list[tuple[Any, ...]] = []
    size = 0
    while len(rows) < limit:
        chunk = cursor.fetchmany(min(batch, limit - len(rows)))
        if not chunk:
            break
        size += sum(map(row_size, chunk))
        if size > MAX_RESULT_BYTES:
            return None
        rows.extend(chunk)
    return rows


def check_run_args(timeout_ms: int, max_rows: int) -> None:
    """Raise ``ValueError`` for a timeout or row cap that ``run`` can't honor."""
    if timeout_ms <= 0:
        raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")
    if max_rows < 0:
        raise ValueError(f"max_rows must be zero or more, got {max_rows}")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
