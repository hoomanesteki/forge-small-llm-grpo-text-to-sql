"""Run manifests: which code, config, data and model produced a number.

A manifest from uncommitted code can't be reproduced, so building one on a dirty
tree is refused unless the caller passes ``allow_dirty``. When allowed, the SHA gets
a ``-dirty`` suffix, so a gate check that the SHA is on dev fails loudly.
"""

from __future__ import annotations

import json
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
Platform = Literal["local-cpu", "local-mps", "kaggle", "colab", "modal"]


class DirtyTreeError(RuntimeError):
    pass


class GitError(RuntimeError):
    """git can't name the commit: no checkout it can read holds the folder, or git is missing."""


def _git(args: list[str], where: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(where), *args], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise GitError(f"can't run git in {where}: {exc}") from exc


def _output(args: list[str], where: Path) -> str:
    done = _git(args, where)
    if done.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {where}: {_one_line(done.stderr)}")
    return done.stdout.strip()


def _one_line(text: str) -> str:
    return " ".join(text.split())


def git_sha(repo: Path | None = None, *, allow_dirty: bool = False) -> str:
    """HEAD's commit in the checkout that holds ``repo`` (by default the current directory).

    The dirty check covers that whole checkout, from its top, whichever of its folders
    ``repo`` is. Raises ``GitError`` when no checkout holds it.
    """
    where = Path.cwd() if repo is None else repo
    found = _git(["rev-parse", "--show-toplevel"], where)
    if found.returncode != 0:
        raise GitError(
            f"{where} is not in a git checkout ({_one_line(found.stderr)}); a run records its "
            "commit, so run it inside a Forge checkout or set FORGE_ROOT to one"
        )
    top = Path(found.stdout.strip())
    sha = _output(["rev-parse", "HEAD"], top)
    # Untracked files count (whatever status.showUntrackedFiles says): a new module that
    # isn't committed still changes results. The append-only logs don't. Pathspecs are
    # anchored at the top, so where git starts can't narrow the check or move the logs.
    excludes = [f":(top,exclude){log}" for log in APPEND_ONLY_LOGS]
    status = _output(["status", "--porcelain", "--untracked-files=all", "--", ":/", *excludes], top)
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
    data_dir: Path | None = None,
) -> RunManifest:
    """Describe a run starting now. Pass ``run_id`` only to keep an id, as a resumed run does.

    ``data_dir`` is the data the run reads, which its data manifest hash describes; the
    default is ``FORGE_DATA_DIR``'s, as everywhere else.
    """
    # Kaggle runs in UTC and laptops don't, so a naive time can't be compared. A tzinfo
    # without an offset is naive too: astimezone would read it in the machine's zone.
    if now is not None and now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    started_at = now or datetime.now(UTC)
    sha = git_sha(repo, allow_dirty=allow_dirty)
    if run_id is None:
        run_id = make_run_id(kind, model_ref, sha, on=started_at.astimezone(UTC).date())
    # The manifest keeps the config as hashed, in a copy of its own, so editing the
    # caller's nested dicts later can't change it or break its config_hash.
    snapshot: dict[str, Any] = json.loads(canonical_json(config))
    return RunManifest(
        run_id=run_id,
        kind=kind,
        git_sha=sha,
        config=snapshot,
        config_hash=config_hash(snapshot),
        data_manifest_hash=data_manifest_hash(data_dir),
        model_ref=model_ref,
        adapter_ref=adapter_ref,
        parent_run=parent_run,
        platform=platform,
        gpu=gpu,
        seed=seed,
        started_at=started_at,
    )
