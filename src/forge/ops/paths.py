"""Where the repo's append-only logs, local runs and caches live."""

from __future__ import annotations

import os
from pathlib import Path

# Logs the CLI appends to. They describe runs rather than produce them, so a change to
# them alone does not make the working tree dirty for manifest purposes.
APPEND_ONLY_LOGS = ("docs/compute-ledger.md", "docs/dev-usage.md")


class RepoRootNotFound(RuntimeError):
    pass


def repo_root() -> Path:
    """``FORGE_ROOT`` if set, else the nearest parent of cwd holding ``docs/PLAN.md``.

    There is no silent fallback to cwd: writing a stray ledger in some other folder
    would lose GPU hours from the real one.
    """
    if os.environ.get("FORGE_ROOT"):
        return Path(os.environ["FORGE_ROOT"])
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "docs" / "PLAN.md").is_file():
            return candidate
    raise RepoRootNotFound(
        f"no Forge repo above {here}; run inside the repo or set FORGE_ROOT "
        "(or FORGE_LEDGER / FORGE_DEV_USAGE for the log files)"
    )


def ledger_path() -> Path:
    if os.environ.get("FORGE_LEDGER"):
        return Path(os.environ["FORGE_LEDGER"])
    return repo_root() / APPEND_ONLY_LOGS[0]


def dev_usage_path() -> Path:
    if os.environ.get("FORGE_DEV_USAGE"):
        return Path(os.environ["FORGE_DEV_USAGE"])
    return repo_root() / APPEND_ONLY_LOGS[1]


def runs_dir() -> Path:
    """Local run folders, ``runs/<run_id>/``: ``FORGE_RUNS_DIR`` if set, else ``runs``."""
    if os.environ.get("FORGE_RUNS_DIR"):
        return Path(os.environ["FORGE_RUNS_DIR"])
    return repo_root() / "runs"


def cache_dir() -> Path:
    """Results worth keeping between runs: ``FORGE_CACHE_DIR`` if set, else ``cache``."""
    if os.environ.get("FORGE_CACHE_DIR"):
        return Path(os.environ["FORGE_CACHE_DIR"])
    return repo_root() / "cache"
