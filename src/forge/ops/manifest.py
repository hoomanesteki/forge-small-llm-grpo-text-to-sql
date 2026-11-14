"""Run manifests: which code, config, data and model produced a number.

A manifest from uncommitted code can't be reproduced, so building one on a dirty
tree is refused unless the caller passes ``allow_dirty``. When allowed, the SHA gets
a ``-dirty`` suffix, so a gate check that the SHA is on dev fails loudly.

FG-102 adds the ``RunManifest`` model and run ids; until then the builder returns a
plain dict with the same fields.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.core.hashing import canonical_json, sha256_hex
from forge.data.manifest import data_manifest_hash
from forge.ops.paths import APPEND_ONLY_LOGS


class DirtyTreeError(RuntimeError):
    pass


def _git(args: list[str], repo: Path | None) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_sha(repo: Path | None = None, *, allow_dirty: bool = False) -> str:
    sha = _git(["rev-parse", "HEAD"], repo)
    # Untracked files count (whatever status.showUntrackedFiles says): a new module that
    # isn't committed still changes results. The append-only logs don't.
    excludes = [f":(exclude){log}" for log in APPEND_ONLY_LOGS]
    status = _git(["status", "--porcelain", "--untracked-files=all", "--", ".", *excludes], repo)
    if status:
        if not allow_dirty:
            raise DirtyTreeError(
                "working tree has uncommitted changes; commit first or pass --allow-dirty"
            )
        return f"{sha}-dirty"
    return sha


def config_hash(config: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(config))


def build_manifest(
    kind: str,
    config: dict[str, Any],
    *,
    run_id: str,
    model_ref: str,
    seed: int,
    platform: str = "local-cpu",
    adapter_ref: str | None = None,
    parent_run: str | None = None,
    gpu: str | None = None,
    allow_dirty: bool = False,
    repo: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "kind": kind,
        "git_sha": git_sha(repo, allow_dirty=allow_dirty),
        "config": config,
        "config_hash": config_hash(config),
        "data_manifest_hash": data_manifest_hash(),
        "model_ref": model_ref,
        "adapter_ref": adapter_ref,
        "parent_run": parent_run,
        "platform": platform,
        "gpu": gpu,
        "seed": seed,
        "started_at": (now or datetime.now(UTC)).isoformat(),
        "ended_at": None,
        "gpu_hours": 0.0,
        "cost_usd": 0.0,
        "outputs": {},
        "status": "running",
    }
