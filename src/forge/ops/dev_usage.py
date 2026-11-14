"""The dev split budget: at most six looks at Spider dev, each logged with a reason.

Every look at the reported split is a chance to tune against it, so the log is
append-only and the seventh request is refused outright.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from forge.ops.paths import dev_usage_path

MAX_DEV_USES = 6
HEADER = "| date | run_id | reason |\n|---|---|---|\n"


class DevBudgetExceeded(RuntimeError):
    pass


def read_uses(path: Path | None = None) -> list[tuple[str, str, str]]:
    path = path or dev_usage_path()
    if not path.is_file():
        return []
    uses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| 2"):
            d, run_id, reason = (c.strip() for c in line.strip().strip("|").split("|", 2))
            uses.append((d, run_id, reason))
    return uses


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
    path = path or dev_usage_path()
    used = count_dev_uses(path)
    if used >= MAX_DEV_USES:
        raise DevBudgetExceeded(f"dev split already used {used} of {MAX_DEV_USES} times")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Dev split usage\n\n" + HEADER, encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    day = (today or date.today()).isoformat()
    clean = reason.strip().replace("|", "/").replace("\n", " ")
    path.write_text(text + f"| {day} | {run_id} | {clean} |\n", encoding="utf-8")
    return used + 1
