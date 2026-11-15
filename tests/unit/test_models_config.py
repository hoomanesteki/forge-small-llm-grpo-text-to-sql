"""configs/models.yaml: the models tests and smoke runs load by name."""

from __future__ import annotations

import re

import pytest


@pytest.mark.phase(2)
def test_tiny_model_is_pinned_to_a_commit(tiny_model_ref: str) -> None:
    repo, at, revision = tiny_model_ref.partition("@")

    # A random-weight model from TRL's test org (plan section 5.7), at a full commit sha:
    # the org rewrites these repos in place, so a branch would move under the tests.
    assert repo.startswith("trl-internal-testing/")
    assert at
    assert re.fullmatch(r"[0-9a-f]{40}", revision), revision
