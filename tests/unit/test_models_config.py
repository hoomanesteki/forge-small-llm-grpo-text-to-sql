"""configs/models.yaml, the models runs load by name, and the eval configs that use them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from forge.eval.config import EvalConfig, check_config, read_config_file

ROOT = Path(__file__).resolve().parents[2]
PILOT_CONFIG = ROOT / "configs" / "eval" / "pilot-qwen25coder05b-val.yaml"


@pytest.mark.phase(2)
def test_tiny_model_is_pinned_to_a_commit(tiny_model_ref: str) -> None:
    repo, at, revision = tiny_model_ref.partition("@")

    # A random-weight model from TRL's test org (plan section 5.7), at a full commit sha:
    # the org rewrites these repos in place, so a branch would move under the tests.
    assert repo.startswith("trl-internal-testing/")
    assert at
    assert re.fullmatch(r"[0-9a-f]{40}", revision), revision


@pytest.mark.phase(2)
def test_pilot_model_is_pinned_and_its_config_runs_it_on_val() -> None:
    pilot = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["pilot"]
    # Apache-2.0 and not gated, the one model besides the tiny one a pilot downloads.
    assert pilot["repo"] == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert re.fullmatch(r"[0-9a-f]{40}", pilot["revision"]), pilot["revision"]

    config = EvalConfig(**read_config_file(PILOT_CONFIG))
    check_config(config)
    assert config.model == f"{pilot['repo']}@{pilot['revision']}"
    assert (config.backend, config.device, config.platform) == ("hf", "mps", "local-mps")
    # The full val split, greedy, with the p1 prompt and no retry (ADR-0016).
    assert (config.split, config.limit, config.prompt_version) == ("val", None, "p1")
    assert (config.n, config.temperature, config.retry) == (1, 0.0, 0)


@pytest.mark.phase(2)
def test_every_eval_config_is_runnable() -> None:
    paths = sorted((ROOT / "configs" / "eval").glob("*.yaml"))
    assert PILOT_CONFIG in paths
    for path in paths:
        config = EvalConfig(**read_config_file(path))
        check_config(config)
        # A pilot is never a dev look (ADR-0016), and a config file can't give a reason.
        assert config.split != "dev", path.name
