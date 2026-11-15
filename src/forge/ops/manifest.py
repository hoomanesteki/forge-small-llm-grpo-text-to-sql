"""Run manifests: which code, config, data and model produced a number.

A manifest from uncommitted code can't be reproduced, so building one on a dirty
tree is refused unless the caller passes ``allow_dirty``. When allowed, the SHA gets
a ``-dirty`` suffix, so a gate check that the SHA is on dev fails loudly.

A Kaggle kernel runs the wheel from the code dataset and has no checkout to ask, so
there ``code_sha`` takes the commit the dataset was built from, which the kernel's entry
script passes in ``FORGE_CODE_SHA``. Inside a checkout the variable is never read.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from forge.core.hashing import canonical_json, sha256_hex
from forge.core.ids import RunKind, make_run_id
from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.ops.paths import RepoRootNotFound, dev_usage_path, ledger_path, repo_root

# Mirrors RunManifest.platform; a unit test keeps the two in step.
Platform = Literal["local-cpu", "local-mps", "kaggle", "colab", "modal"]

# Set by kaggle/entry.py from the code dataset's GIT_SHA (forge.ops.kaggle).
CODE_SHA_ENV = "FORGE_CODE_SHA"
# A full commit: publish-code refuses a dirty tree, so a dataset never names a -dirty one.
_COMMIT = re.compile(r"[0-9a-f]{40}")


class DirtyTreeError(RuntimeError):
    """The code's commit can't be named: the tree has changes no commit holds."""


class GitError(RuntimeError):
    """git can't name the commit: no checkout it can read holds the folder, or git is missing."""


class NoCommitError(GitError):
    """No checkout to ask, and no ``FORGE_CODE_SHA`` to say which commit the code is."""


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
    ``repo`` is, except the append-only logs where ``forge.ops.paths`` puts them, so a
    Forge root below the top still leaves them out. Raises ``GitError`` when no checkout
    holds ``repo``, or when the Forge root sits in that checkout without being tracked
    by it.
    """
    where = Path.cwd() if repo is None else repo
    found = _git(["rev-parse", "--show-toplevel"], where)
    if found.returncode != 0:
        raise GitError(
            f"{where} is not in a git checkout ({_one_line(found.stderr)}); a run records its "
            "commit, so run it inside a Forge checkout or set FORGE_ROOT to one"
        )
    top = Path(found.stdout.strip())
    _require_tracked_forge_root(top)
    sha = _output(["rev-parse", "HEAD"], top)
    # Untracked files count (whatever status.showUntrackedFiles says): a new module that
    # isn't committed still changes results. The append-only logs don't. Pathspecs are
    # anchored at the top, so where git starts can't narrow the check or move the logs.
    excludes = _log_excludes(top)
    status = _output(["status", "--porcelain", "--untracked-files=all", "--", ":/", *excludes], top)
    if status:
        if not allow_dirty:
            raise DirtyTreeError(
                "working tree has uncommitted changes; commit first or pass --allow-dirty"
            )
        return f"{sha}-dirty"
    return sha


def _log_excludes(top: Path) -> list[str]:
    """Pathspecs that leave out each append-only log the CLI writes (``ledger_path``,
    ``dev_usage_path``) that is in the checkout at ``top``."""
    excludes = []
    for log_path in (ledger_path, dev_usage_path):
        try:
            log = log_path()
        except RepoRootNotFound:
            # With no Forge root to put it in, the CLI can't have appended to this log.
            continue
        rel = _path_in_checkout(log, top)
        if rel is not None:
            excludes.append(f":(top,exclude,literal){rel}")
    return excludes


def _require_tracked_forge_root(top: Path) -> None:
    """Refuse a Forge root that sits in the checkout at ``top`` without being part of it.

    A FORGE_ROOT copied into some other repository, untracked or ignored there, would
    otherwise record that repository's commit as the code that ran.
    """
    try:
        root = repo_root()
    except RepoRootNotFound:
        return
    plan = root / "docs" / "PLAN.md"
    if not plan.is_file():
        return
    rel = _path_in_checkout(plan, top)
    if rel is None:
        # The Forge root is in another checkout, or in none; this one's commit stands.
        return
    tracked = _git(["ls-files", "--error-unmatch", "--", f":(top,literal){rel}"], top)
    if tracked.returncode != 0:
        raise GitError(
            f"{root} holds docs/PLAN.md, but the checkout at {top} doesn't track it, so that "
            "checkout's commit isn't the code that runs; run inside a Forge checkout or set "
            "FORGE_ROOT to one"
        )


def _path_in_checkout(path: Path, top: Path) -> str | None:
    """``path`` relative to the top of the checkout at ``top``, spelled as git spells it,
    or None when that checkout doesn't hold it.

    git names the folder from the file system, so a path typed in other letter case or
    through a symlink still gets the form its pathspecs match.
    """
    absolute = Path(os.path.abspath(path))
    missing = [absolute.name]
    folder = absolute.parent
    # A log may not exist yet; git can only start in a folder that does.
    while not folder.is_dir():
        if folder.parent == folder:
            return None
        missing.insert(0, folder.name)
        folder = folder.parent
    found = _git(["rev-parse", "--show-toplevel", "--show-prefix"], folder)
    if found.returncode != 0:
        return None
    lines = found.stdout.split("\n")
    if not lines[0] or not os.path.samefile(lines[0], top):
        return None
    prefix = lines[1] if len(lines) > 1 else ""
    return prefix + "/".join(missing)


def code_sha(
    repo: Path | None = None, *, allow_dirty: bool = False, env: Mapping[str, str] | None = None
) -> str:
    """The commit a run's code comes from.

    In a git checkout this is ``git_sha``, whatever ``FORGE_CODE_SHA`` says, so the variable
    can never stand in for the code that is really there. Outside one it is
    ``FORGE_CODE_SHA``, which must be a full commit. Raises ``NoCommitError`` when neither
    names one, and ``ValueError`` when the variable isn't a full commit.
    """
    if _in_checkout(repo):
        return git_sha(repo, allow_dirty=allow_dirty)
    where = Path.cwd() if repo is None else repo
    sha = (os.environ if env is None else env).get(CODE_SHA_ENV, "")
    if not sha:
        raise NoCommitError(
            f"{where} is not in a git checkout, and {CODE_SHA_ENV} doesn't name the commit "
            "the code was built from; a run records its commit, so run it inside a Forge "
            "checkout or set FORGE_ROOT to one"
        )
    if not _COMMIT.fullmatch(sha):
        raise ValueError(f"{CODE_SHA_ENV} must be a full 40-character commit, got {sha!r}")
    return sha


def _in_checkout(repo: Path | None) -> bool:
    where = Path.cwd() if repo is None else repo
    try:
        done = _git(["rev-parse", "--is-inside-work-tree"], where)
    except GitError:  # no git on the machine
        return False
    return done.returncode == 0 and done.stdout.strip() == "true"


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
    default is ``FORGE_DATA_DIR``'s, as everywhere else. The commit is ``code_sha``'s.
    """
    # Kaggle runs in UTC and laptops don't, so a naive time can't be compared. A tzinfo
    # without an offset is naive too: astimezone would read it in the machine's zone.
    if now is not None and now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    started_at = now or datetime.now(UTC)
    sha = code_sha(repo, allow_dirty=allow_dirty)
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
