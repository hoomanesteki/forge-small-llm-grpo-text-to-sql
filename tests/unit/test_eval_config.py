"""The eval runner's options: what is refused up front, and what the manifest records."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

from forge.core.types import GenParams
from forge.eval.config import EvalConfig, NotAvailable, check_config

BASE = EvalConfig(model="org/model")


@pytest.mark.phase(2)
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"split": "dev"}, ValueError, "--split dev needs --reason"),
        ({"split": "dev", "reason": " \t"}, ValueError, "--split dev needs --reason"),
        ({"reason": "a look"}, ValueError, "--reason goes with --split dev"),
        ({"split": "test"}, ValueError, "split must be one of train, val, dev"),
        ({"backend": "onnx"}, ValueError, "backend must be one of hf, vllm, llamacpp, fake"),
        ({"store": "s3"}, ValueError, "store must be one of local, hub"),
        ({"prompt_version": "p3"}, ValueError, "unknown prompt version 'p3'"),
        ({"limit": 0}, ValueError, "limit must be at least 1"),
        ({"retry": 2}, ValueError, "retry must be 0 or 1"),
        ({"run_id": "Run-1"}, ValueError, "names a folder"),
        ({"run_id": "../up"}, ValueError, "names a folder"),
        ({"run_id": "-x"}, ValueError, "names a folder"),
        ({"backend": "fake"}, ValueError, "--backend fake takes --model gold or select-1"),
        ({"backend": "fake", "model": "gold", "adapter": "a"}, ValueError, "loads no adapter"),
        ({"n": 4}, ValueError, "n=4 needs a temperature above 0"),
        ({"backend": "vllm"}, NotAvailable, "the vllm backend is not available yet"),
        ({"backend": "llamacpp"}, NotAvailable, "comes with FG-702"),
        ({"store": "hub"}, NotAvailable, "--store hub is available after FG-304"),
        ({"retry": 1}, NotAvailable, "--retry 1 is available after FG-206"),
    ],
)
def test_config_refused_before_anything_runs(
    changes: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=re.escape(message)):
        check_config(dataclasses.replace(BASE, **changes))


@pytest.mark.phase(2)
def test_config_accepted() -> None:
    for config in (
        BASE,
        EvalConfig(model="gold", backend="fake", split="train", limit=3, run_id="smoke-abc1234"),
        EvalConfig(model="org/model", adapter="runs/x/best", split="dev", reason="FG-508"),
        EvalConfig(model="org/model", n=8, temperature=0.7, prompt_version="p2"),
    ):
        check_config(config)


@pytest.mark.phase(2)
def test_manifest_config_records_what_changes_a_score() -> None:
    config = BASE.manifest_config()
    assert config["generation"] == GenParams().model_dump()
    assert config["execution"] == {
        "timeout_ms": 2000,
        "gold_timeout_ms": 10_000,
        "max_rows": 100_000,
        "official_item_timeout_s": 10.0,
    }
    for changes in ({"n": 2, "temperature": 0.5}, {"limit": 5}, {"prompt_version": "p2"}):
        assert dataclasses.replace(BASE, **changes).manifest_config() != config
    # Where the files go and why dev was read don't change a score.
    assert dataclasses.replace(BASE, store="hub").manifest_config() == config
    dev = dataclasses.replace(BASE, split="dev", reason="a look").manifest_config()
    assert "reason" not in dev and dev == {**config, "split": "dev"}
