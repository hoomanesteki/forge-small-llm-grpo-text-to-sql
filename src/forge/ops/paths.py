"""Where the repo's append-only logs live."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """``FORGE_ROOT`` if set, else the nearest parent holding ``docs/PLAN.md``, else cwd."""
    if os.environ.get("FORGE_ROOT"):
        return Path(os.environ["FORGE_ROOT"])
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "docs" / "PLAN.md").is_file():
            return candidate
    return here


def ledger_path() -> Path:
    return Path(os.environ.get("FORGE_LEDGER", repo_root() / "docs" / "compute-ledger.md"))


def dev_usage_path() -> Path:
    return Path(os.environ.get("FORGE_DEV_USAGE", repo_root() / "docs" / "dev-usage.md"))
