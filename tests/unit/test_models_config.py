"""configs/models.yaml, the models runs load by name, and the eval configs that use them."""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from forge.eval.config import EvalConfig, check_config, read_config_file, read_config_sections
from forge.ops.ledger import read_rows

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "configs" / "models.yaml"
PILOT_CONFIG = ROOT / "configs" / "eval" / "pilot-qwen25coder05b-val.yaml"
SMOKE_CONFIG = ROOT / "configs" / "eval" / "smoke-kaggle-50-val.yaml"
PENDING = ROOT / "docs" / "runs" / "pending.yaml"
# FG-307's GPU hours (plan section 3.8).
BASELINE_BUDGET = 3.0


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


def _chosen_repo() -> str:
    """The model ADR-0012's rule chose from the candidates' val runs."""
    adr = (ROOT / "docs" / "adr" / "0012-model-choice.md").read_text(encoding="utf-8")
    chosen = re.search(r"\n- Chosen model: `([^`]+)`\n", adr)
    assert chosen, "ADR-0012 names no chosen model"
    return chosen.group(1)


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
    # and only it: Qwen's AWQ checkpoint, since vLLM 0.33.0 has no bitsandbytes (FG-301).
    assert all(1.5e9 <= e["params"] <= 4.1e9 for e in candidates.values())
    assert not any("quantization" in e for e in candidates.values())
    (big,) = reference.values()
    assert big["params"] > 4.1e9 and big["quantization"] == "awq"
    assert (big["repo"], big["revision"]) == (
        "Qwen/Qwen3-8B-AWQ",
        "4da05a8edb55c6046cce958586c33b61da07bb79",
    )


@pytest.mark.phase(3)
def test_baseline_configs_run_each_model_as_its_run_needs() -> None:
    models = _models()
    greedy = {"backend": "vllm", "limit": None, "n": 1, "temperature": 0.0, "retry": 0}
    p1, p2 = {"prompt_version": "p1"}, {"prompt_version": "p2"}
    # The runs after the choice use p2, the prompt ADR-0007 fixed from the val runs: the
    # chosen model's dev look, to compare with its trained model's, and the reference's
    # two, to show a larger model with the same prompt. The other candidates' dev configs
    # never run, and stay as they were written.
    chosen = _chosen_repo()
    assert chosen in {entry["repo"] for entry in models["candidates"].values()}
    runs: list[tuple[str, dict[str, Any], dict[str, dict[str, Any]]]] = []
    for name, entry in models["candidates"].items():
        dev = p2 if entry["repo"] == chosen else p1
        kinds = {"val": {"split": "val", **p1}, "val-p2": {"split": "val", **p2}}
        runs.append((name, entry, {**kinds, "dev": {"split": "dev", **dev}}))
    for name, entry in models["reference"].items():
        runs.append((name, entry, {"val": {"split": "val", **p2}, "dev": {"split": "dev", **p2}}))
    expected: dict[str, EvalConfig] = {}
    for name, entry, kinds in runs:
        pinned = {
            "model": f"{entry['repo']}@{entry['revision']}",
            "quantization": entry.get("quantization"),
        }
        for kind, options in kinds.items():
            expected[f"{name}-{kind}.yaml"] = EvalConfig(**greedy, **pinned, **options)
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


def _committed_runs() -> dict[str, dict[str, Any]]:
    """The config each run committed under docs/runs records in its manifest, by run id."""
    runs = {}
    for path in sorted((ROOT / "docs" / "runs").glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        # A push over the week's budget says so in its config, which changes no score.
        manifest["config"].pop("allow_over_budget", None)
        runs[manifest["run_id"]] = manifest["config"]
    return runs


@pytest.mark.phase(3)
def test_baseline_runs_wait_in_pending_within_the_story_budget() -> None:
    pending = yaml.safe_load(PENDING.read_text(encoding="utf-8"))["pending"]
    assert all(isinstance(entry["since"], date) for entry in pending)
    baseline = [entry for entry in pending if entry["story"] == "FG-307"]
    listed: list[str] = []
    for entry in baseline:
        hours = [
            read_config_sections(ROOT / path)[1].est_gpu_hours or 0.0 for path in entry["configs"]
        ]
        # A pick runs one of its configs, so it budgets the dearest; otherwise each runs.
        budget = max(hours) if "pick" in entry else sum(hours)
        assert entry["est_gpu_hours"] == pytest.approx(budget), entry["run"]
        listed += entry["configs"]

    # Each baseline config has made its committed run, waits for its run here, or is the
    # p2 or dev config of a candidate ADR-0012 didn't choose, which never runs: one of the
    # three, once.
    configs = (ROOT / "configs" / "eval").glob("*.yaml")
    baselines = {path for path in configs if path not in (PILOT_CONFIG, SMOKE_CONFIG)}
    committed = _committed_runs()
    ran = {
        run_id: path
        for path in baselines
        for run_id, config in committed.items()
        if EvalConfig(**read_config_file(path)).manifest_config() == config
    }
    chosen = _chosen_repo()
    unchosen = {
        ROOT / "configs" / "eval" / f"{name}-{kind}.yaml"
        for name, entry in _models()["candidates"].items()
        if entry["repo"] != chosen
        for kind in ("val-p2", "dev")
    }
    assert len(set(ran.values())) == len(ran) and not set(ran.values()) & unchosen
    waiting = baselines - set(ran.values()) - unchosen
    assert sorted(listed) == sorted(path.relative_to(ROOT).as_posix() for path in waiting)
    # A run that was made but waits here names itself in its entry, so its ledger row
    # counts instead of its estimate. Together they fit the story: the ledger rows of the
    # runs made, one or more each, and the estimates of the runs waiting.
    held = {entry["run_id"] for entry in baseline if "run_id" in entry}
    assert not held & committed.keys()
    made = ran.keys() | held
    rows = [row for row in read_rows(ROOT / "docs" / "compute-ledger.md") if row.run_id in made]
    assert {row.run_id for row in rows} == made
    spent = sum(row.hours for row in rows)
    waiting_hours = sum(entry["est_gpu_hours"] for entry in baseline if "run_id" not in entry)
    assert spent + waiting_hours <= BASELINE_BUDGET
    # Plan section 3.8's note gives the hours they took, beside its row's estimate.
    plan = (ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8")
    note = " ".join(plan.split("\n### 3.8 ", 1)[1].split("\n### ", 1)[0].split())
    took = re.search(r"FG-307's runs took ([\d.]+) h by their (\d+) ledger rows", note)
    assert took, "plan section 3.8 doesn't say what FG-307's runs took"
    assert float(took.group(1)) == pytest.approx(spent)
    assert int(took.group(2)) == len(rows)
