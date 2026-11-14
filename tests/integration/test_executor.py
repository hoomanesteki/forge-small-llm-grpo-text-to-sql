"""Read-only connections to real Spider databases."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from forge.data.spider import DataNotFound
from forge.sql.executor import connect_readonly

DB = "department_management"


def _db_file(root: Path, db_id: str = DB) -> Path:
    return root / "database" / db_id / f"{db_id}.sqlite"


@pytest.mark.phase(1)
def test_connect_readonly_guards_on_two_layers(spider_mini: Path, tmp_path: Path) -> None:
    path = _db_file(spider_mini)
    with closing(connect_readonly(path)) as conn:
        assert conn.execute("SELECT count(*) FROM head").fetchall() == [(10,)]
        # The authorizer refuses writes as they are prepared...
        with pytest.raises(sqlite3.DatabaseError, match=r"^not authorized$"):
            conn.execute("DELETE FROM head")
        # ...and without it, the read-only URI still refuses them as they run.
        conn.set_authorizer(None)
        with pytest.raises(sqlite3.OperationalError, match="readonly database"):
            conn.execute("DELETE FROM head")
        assert conn.execute("SELECT count(*) FROM head").fetchall() == [(10,)]

    # A missing file is reported up front, not as a failing query later.
    with pytest.raises(DataNotFound, match=r"gone\.sqlite not found; run `forge data fetch`"):
        connect_readonly(tmp_path / "gone.sqlite")
