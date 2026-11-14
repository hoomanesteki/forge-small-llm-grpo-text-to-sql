"""Open Spider's SQLite databases so that nothing done through a connection can change them.

Every connection to a Spider database is made by ``connect_readonly`` (CONTRIBUTING
rule 9). It is read-only and immutable, so SQLite never writes, locks or journals the
file, and its authorizer is default-deny: only ``SQLITE_SELECT``, ``SQLITE_READ``,
``SQLITE_FUNCTION`` and ``SQLITE_RECURSIVE`` pass, so writes, DDL, ``ATTACH``, ``VACUUM``,
``PRAGMA`` and transactions fail as they are prepared, before anything runs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from forge.data.spider import DATA_DIR_ENV, DataNotFound

ALLOWED_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)


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


def _authorize(
    action: int, arg1: str | None, arg2: str | None, db_name: str | None, source: str | None
) -> int:
    # Default deny: an action that a newer SQLite adds stays refused until listed here.
    return sqlite3.SQLITE_OK if action in ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")
