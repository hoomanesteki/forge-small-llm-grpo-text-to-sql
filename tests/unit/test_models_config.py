"""configs/models.yaml, the models runs load by name, and the eval configs that use them."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from forge.eval.config import EvalConfig, check_config, read_config_file, read_config_sections

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "configs" / "models.yaml"
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
        options, ops = read_config_sections(path)
        config = EvalConfig(**options)
        if config.backend == "vllm":
            # A kernel job's, checked as forge kaggle run checks it, with its estimate.
            config = dataclasses.replace(config, host="kaggle")
            assert ops.est_gpu_hours is not None, path.name
        if config.split == "dev":
            # A config file can't give the reason; the command line does.
            config = dataclasses.replace(config, reason="baseline")
        check_config(config)
        if config.device == "mps":
            # A pilot is never a dev look (ADR-0016).
            assert config.split != "dev", path.name


def _models() -> dict[str, Any]:
    models: dict[str, Any] = yaml.safe_load(MODELS.read_text(encoding="utf-8"))
    return models


@pytest.mark.phase(3)
def test_baseline_models_are_open_pinned_and_sized_for_their_role() -> None:
    models = _models()
    candidates, reference = models["candidates"], models["reference"]
    assert sorted(candidates) == ["qwen2-5-coder-1-5b", "qwen3-1-7b", "qwen3-4b"]
    assert list(reference) == ["qwen3-8b"]
    for entry in (*candidates.values(), *reference.values()):
        # Open and ungated, so a kernel downloads it without a token or anyone's approval.
        assert (entry["license"], entry["gated"]) == ("apache-2.0", False), entry["repo"]
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), entry["repo"]
    # The plan's Qwen2.5-Coder-3B-Instruct is under the non-commercial Qwen Research license.
    assert "Qwen/Qwen2.5-Coder-3B-Instruct" not in {e["repo"] for e in candidates.values()}
    # Candidates are about 1.5B to 4B; the reference is larger, so vllm loads it in 4-bit,
    # and only it.
    assert all(1.5e9 <= e["params"] <= 4.1e9 for e in candidates.values())
    assert not any("quantization" in e for e in candidates.values())
    (big,) = reference.values()
    assert big["params"] > 4.1e9 and big["quantization"] == "bitsandbytes"


@pytest.mark.phase(3)
def test_baseline_configs_run_each_model_as_its_run_needs() -> None:
    models = _models()
    greedy = {"backend": "vllm", "limit": None, "n": 1, "temperature": 0.0, "retry": 0}
    runs = {"val": ("val", "p1"), "val-p2": ("val", "p2"), "dev": ("dev", "p1")}
    expected: dict[str, EvalConfig] = {}
    for group, kinds in (
        ("candidates", runs),
        ("reference", {"val": runs["val"], "dev": runs["dev"]}),
    ):
        for name, entry in models[group].items():
            pinned = {
                "model": f"{entry['repo']}@{entry['revision']}",
                "quantization": entry.get("quantization"),
            }
            for kind, (split, prompt) in kinds.items():
                expected[f"{name}-{kind}.yaml"] = EvalConfig(
                    **greedy, **pinned, split=split, prompt_version=prompt
                )
    # I6's repeat of the first 100 val questions, and the phase 3 smoke's 50.
    coder = expected["qwen2-5-coder-1-5b-val.yaml"]
    expected["qwen2-5-coder-1-5b-val-100.yaml"] = dataclasses.replace(coder, limit=100)
    expected["smoke-kaggle-50-val.yaml"] = dataclasses.replace(coder, limit=50)

    paths = {path.name: path for path in (ROOT / "configs" / "eval").glob("*.yaml")}
    assert sorted(paths) == sorted([*expected, PILOT_CONFIG.name])
    for name, config in expected.items():
        options, ops = read_config_sections(paths[name])
        assert EvalConfig(**options) == config, name
        assert ops.est_gpu_hours is not None and ops.est_gpu_hours <= 0.5, name
