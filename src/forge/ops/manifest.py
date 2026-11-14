"""Run manifests: which code, config, data and model produced a number.

A manifest from uncommitted code can't be reproduced, so building one on a dirty
tree is refused unless the caller passes ``allow_dirty``. When allowed, the SHA gets
a ``-dirty`` suffix, so a gate check that the SHA is on dev fails loudly.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from forge.core.hashing import canonical_json, sha256_hex
from forge.core.ids import RunKind, make_run_id
from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.ops.paths import APPEND_ONLY_LOGS

# Mirrors RunManifest.platform; a unit test keeps the two in step.
Platform = Literal["local-cpu", "kaggle", "colab", "modal"]


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
    kind: RunKind,
    config: dict[str, Any],
    *,
    model_ref: str,
    seed: int,
    run_id: str | None = None,
    platform: Platform = "local-cpu",
    adapter_ref: str | None = None,
    parent_run: str | None = None,
    gpu: str | None = None,
    allow_dirty: bool = False,
    repo: Path | None = None,
    now: datetime | None = None,
) -> RunManifest:
    """Describe a run starting now. Pass ``run_id`` only to keep an id, as a resumed run does."""
    if now is not None and now.tzinfo is None:
        # Kaggle runs in UTC and laptops don't, so a naive time can't be compared.
        raise ValueError("now must be timezone-aware")
    started_at = now or datetime.now(UTC)
    sha = git_sha(repo, allow_dirty=allow_dirty)
    if run_id is None:
        run_id = make_run_id(kind, model_ref, sha, on=started_at.astimezone(UTC).date())
    return RunManifest(
        run_id=run_id,
        kind=kind,
        git_sha=sha,
        config=config,
        config_hash=config_hash(config),
        data_manifest_hash=data_manifest_hash(),
        model_ref=model_ref,
        adapter_ref=adapter_ref,
        parent_run=parent_run,
        platform=platform,
        gpu=gpu,
        seed=seed,
        started_at=started_at,
    )
