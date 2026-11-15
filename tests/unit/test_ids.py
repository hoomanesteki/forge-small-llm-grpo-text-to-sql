from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import get_args

import pytest

from forge.core import ids
from forge.core.ids import RUN_KINDS, make_run_id
from forge.core.types import RunManifest

RUN_ID = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+-(eval|sft|grpo|export|audit)-[0-9a-f]{7}$")
SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
DAY = date(2026, 11, 3)


@pytest.mark.phase(1)
def test_run_id_format() -> None:
    assert make_run_id("grpo", "Qwen/Qwen3-4B", SHA, on=DAY) == "2026-11-03-qwen3-4b-grpo-a1b2c3d"

    refs = [
        "Qwen/Qwen3-4B",
        "hoomanesteki/forge-qwen3-4b-sft@3f2a9c1",
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "runs/2026-11-03-qwen3-4b-sft-a1b2c3d/checkpoints/best/",
        "Org/Model_v2 (8-bit)",
    ]
    for kind in RUN_KINDS:
        for ref in refs:
            for sha in (SHA, f"{SHA}-dirty", SHA[:7]):
                run_id = make_run_id(kind, ref, sha, on=DAY)
                assert RUN_ID.fullmatch(run_id), run_id
    assert RUN_ID.fullmatch(make_run_id("eval", "Qwen/Qwen3-4B", SHA))


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("model_ref", "slug"),
    [
        ("Qwen/Qwen3-4B", "qwen3-4b"),
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "qwen2-5-coder-1-5b-instruct"),
        ("hoomanesteki/forge-qwen3-4b-sft@3f2a9c1", "forge-qwen3-4b-sft"),
        # Hub repo ids can't contain '@', so everything after the first one is the revision.
        ("hoomanesteki/forge-qwen3-4b-sft@refs/pr/1", "forge-qwen3-4b-sft"),
        ("runs/x/checkpoints/best/", "best"),
        ("__Model__V2__", "model-v2"),
        ("Qwen3-4B", "qwen3-4b"),
    ],
)
def test_model_slug_is_last_segment_slugified(model_ref: str, slug: str) -> None:
    assert make_run_id("sft", model_ref, SHA, on=DAY) == f"2026-11-03-{slug}-sft-a1b2c3d"


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("model_ref", "slug"),
    [
        ("./runs/x@y/ckpt", "ckpt"),
        ("/kaggle/working/runs/a@b/checkpoints/best/", "best"),
        ("../exports/qwen@2", "qwen-2"),
        ("~/models/qwen3@base", "qwen3-base"),
        ("runs/a/b@c", "b-c"),
    ],
)
def test_at_sign_in_a_local_path_is_part_of_its_name(model_ref: str, slug: str) -> None:
    # A path that can't be a hub repo id before its '@' has no revision to strip.
    assert make_run_id("eval", model_ref, SHA, on=DAY) == f"2026-11-03-{slug}-eval-a1b2c3d"


@pytest.mark.phase(1)
def test_dirty_sha_uses_first_seven_hex() -> None:
    assert make_run_id("eval", "m", f"{SHA}-dirty", on=DAY).endswith("-eval-a1b2c3d")


@pytest.mark.phase(1)
def test_datetime_contributes_only_its_date() -> None:
    at = datetime(2026, 11, 3, 23, 59, tzinfo=UTC)
    assert make_run_id("eval", "m", SHA, on=at) == "2026-11-03-m-eval-a1b2c3d"


@pytest.mark.phase(1)
def test_default_date_is_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    class Frozen(datetime):
        @classmethod
        def now(cls, tz: object = None) -> Frozen:
            assert tz is UTC, "run ids must not depend on the machine's time zone"
            return cls(2026, 11, 4, 0, 30, tzinfo=UTC)

    monkeypatch.setattr(ids, "datetime", Frozen)
    assert make_run_id("eval", "m", SHA).startswith("2026-11-04-")


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("kind", "model_ref", "git_sha", "message"),
    [
        ("train", "m", SHA, "kind"),
        ("eval", "m", "unknown", "git_sha"),
        ("eval", "m", SHA[:6], "git_sha"),
        ("eval", "m", SHA.upper(), "git_sha"),
        ("eval", "m", f"{SHA}-modified", "git_sha"),
        ("eval", "m", "", "git_sha"),
        ("eval", "", SHA, "model_ref"),
        ("eval", "@main", SHA, "model_ref"),
        ("eval", "Org/___", SHA, "model_ref"),
    ],
)
def test_rejects_inputs_that_cannot_make_a_valid_id(
    kind: str, model_ref: str, git_sha: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_run_id(kind, model_ref, git_sha, on=DAY)


@pytest.mark.phase(1)
def test_run_kinds_match_the_manifest_contract() -> None:
    assert get_args(RunManifest.model_fields["kind"].annotation) == RUN_KINDS
