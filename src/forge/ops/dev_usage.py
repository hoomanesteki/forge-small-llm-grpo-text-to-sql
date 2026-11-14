"""The dev split budget: at most six looks at Spider dev, each logged with a reason.

Every look at the reported split is a chance to tune against it, so the log is
append-only and the seventh request is refused outright.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from forge.ops.logfile import locked_append
from forge.ops.paths import dev_usage_path

MAX_DEV_USES = 6
HEADER = "| date | run_id | reason |\n|---|---|---|\n"


class DevBudgetExceeded(RuntimeError):
    pass


def _parse(text: str) -> list[tuple[str, str, str]]:
    uses = []
    for line in text.splitlines():
        if line.startswith("| 2"):
            d, run_id, reason = (c.strip() for c in line.strip().strip("|").split("|", 2))
            uses.append((d, run_id, reason))
    return uses


def read_uses(path: Path | None = None) -> list[tuple[str, str, str]]:
    path = path or dev_usage_path()
    return _parse(path.read_text(encoding="utf-8")) if path.is_file() else []


def count_dev_uses(path: Path | None = None) -> int:
    return len(read_uses(path))


def log_dev_use(
    run_id: str, reason: str, *, path: Path | None = None, today: date | None = None
) -> int:
    """Record one dev evaluation and return how many have been used, this one included."""
    if not reason or not reason.strip():
        raise ValueError("a dev evaluation needs a non-empty reason")
    if not run_id.strip():
        raise ValueError("a dev evaluation needs a run id")
    clean_id = " ".join(run_id.split())
    if "|" in clean_id:
        raise ValueError(f"run_id must not contain '|': {run_id!r}")
    clean_reason = " ".join(reason.replace("|", "/").split())
    day = (today or date.today()).isoformat()

    def within_budget(before: str) -> None:
        used = len(_parse(before))
        if used >= MAX_DEV_USES:
            raise DevBudgetExceeded(f"dev split already used {used} of {MAX_DEV_USES} times")

    before = locked_append(
        path or dev_usage_path(),
        f"| {day} | {clean_id} | {clean_reason} |",
        preamble="# Dev split usage\n\n" + HEADER,
        check=within_budget,
    )
    return len(_parse(before)) + 1
