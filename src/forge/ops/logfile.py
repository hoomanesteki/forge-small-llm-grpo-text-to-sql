"""Locked appends to the markdown logs (compute ledger, dev usage).

Both logs are append-only tables that several processes may write at once, for
example parallel ``forge runs pull`` calls. Each write takes an exclusive lock, reads
the current content, runs an optional check (the dev budget), then appends one line,
so rows are never lost and the budget can't be exceeded by a race.
"""

from __future__ import annotations

import fcntl
from collections.abc import Callable
from pathlib import Path


def locked_append(
    path: Path,
    line: str,
    *,
    preamble: str,
    check: Callable[[str], None] | None = None,
) -> str:
    """Append ``line`` under an exclusive lock and return the file content before it."""
    if "\n" in line or "\r" in line:
        raise ValueError("a log row must be a single line")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            before = fh.read()
            if check is not None:
                check(before)
            prefix = ""
            if not before:
                prefix = preamble
            elif not before.endswith("\n"):
                prefix = "\n"
            # Mode a+ always writes at the end, whatever the read position.
            fh.write(prefix + line + "\n")
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return before
