"""Run ids: ``YYYY-MM-DD-<model-slug>-<kind>-<sha7>``.

Ids name folders in the run store, so they keep to ``[a-z0-9-]``, sort by date, and
say at a glance which model, which job and which commit made the run.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Literal, get_args

# Mirrors RunManifest.kind; a unit test keeps the two in step.
RunKind = Literal["eval", "sft", "grpo", "export", "audit"]
RUN_KINDS: tuple[str, ...] = get_args(RunKind)

# Full or abbreviated SHA-1/SHA-256, plus the suffix forge.ops.manifest adds on a dirty tree.
_GIT_SHA = re.compile(r"[0-9a-f]{7,64}(-dirty)?")
_NOT_SLUG = re.compile(r"[^a-z0-9]+")
# The shape huggingface_hub accepts for a repo id: an optional namespace and a name, each
# of word characters, '-' and '.', starting and ending with a word character.
_HUB_REPO_ID = re.compile(r"(?:\w(?:[\w.-]*\w)?/)?\w(?:[\w.-]*\w)?")


def _model_slug(model_ref: str) -> str:
    repo, at, _ = model_ref.partition("@")
    # Only a hub ref, "<repo id>@<revision>", carries a revision, which may hold slashes
    # (refs/pr/1). In a local path such as ./runs/a@b/ckpt the '@' belongs to a directory
    # name. "@main" names a revision of no model, so it gets no slug.
    is_hub_ref = bool(at) and (not repo or _HUB_REPO_ID.fullmatch(repo) is not None)
    name = (repo if is_hub_ref else model_ref).rstrip("/").rsplit("/", 1)[-1]
    slug = _NOT_SLUG.sub("-", name.lower()).strip("-")
    if not slug:
        raise ValueError(f"model_ref {model_ref!r} has no letters or digits to name a run by")
    return slug


def make_run_id(kind: RunKind, model_ref: str, git_sha: str, *, on: date | None = None) -> str:
    """Build a run id such as ``2026-11-03-qwen3-4b-grpo-a1b2c3d``.

    ``on`` defaults to today in UTC, so the date doesn't depend on where the run starts.
    """
    if kind not in RUN_KINDS:
        raise ValueError(f"kind must be one of {', '.join(RUN_KINDS)}, got {kind!r}")
    if not _GIT_SHA.fullmatch(git_sha):
        raise ValueError(
            f"git_sha must be 7 to 64 lowercase hex characters, optionally ending in -dirty, "
            f"got {git_sha!r}"
        )
    day = on if on is not None else datetime.now(UTC).date()
    # strftime rather than isoformat, which would append the time for a datetime.
    return f"{day:%Y-%m-%d}-{_model_slug(model_ref)}-{kind}-{git_sha[:7]}"
