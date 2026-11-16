"""``forge runs promote``: an SFT run's checkpoints scored by their val runs on the fixture
data, and the best one promoted (FG-404), with the model's replies stood in for."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import GenParams, Prompt
from forge.data.splits import load_train, load_val
from forge.eval.config import EvalConfig, read_config_file
from forge.eval.metrics import summarize
from forge.eval.run_files import read_run
from forge.eval.runner import run_eval
from forge.infer.base import FakeBackend
from forge.ops.hub import BEST, FINAL, HubRunStore, LocalRunStore
from forge.train.sft import PROMOTION_FILE
from tests.fakes import FakeHub, sft_run_folder

RUN = "2026-11-20-qwen3-4b-sft-abc1234"
MODEL = "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
HALF = "checkpoints/step-000242"
OTHER = "2026-11-19-qwen3-4b-sft-abc1234"  # another SFT run
REPO = "owner/forge-runs"


# The base model's full val config, as configs/eval/<model>-val.yaml names it.
BASE = f"model: {MODEL}\nbackend: hf\nsplit: val\nlimit: null\nprompt_version: p1\n"


@pytest.fixture
def runs(spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """runs/ under tmp_path, holding a finished SFT run with its half-way and final
    checkpoints, as forge runs pull leaves one, and the base config beside it."""
    folder = tmp_path / "runs"
    monkeypatch.setenv("FORGE_RUNS_DIR", str(folder))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    sft_run_folder(folder, RUN, MODEL, [HALF, FINAL])
    (tmp_path / "base.yaml").write_text(BASE, encoding="utf-8")
    return folder


def _evaluate(runs: Path, run_id: str, adapter: str, right: int, **options: Any) -> None:
    """A run of ``adapter`` with the base config, on every val question unless ``options``
    say otherwise, whose first ``right`` replies are gold queries where the reply knows
    them, and the rest ``SELECT 1``."""
    gold = {example.id: example.gold_sql for example in [*load_train(), *load_val()]}
    answered: list[str] = []

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        answered.append(prompt.example_id)
        sql = gold.get(prompt.example_id, "SELECT 1") if len(answered) <= right else "SELECT 1"
        return f"```sql\n{sql}\n```"

    base = read_config_file(runs.parent / "base.yaml")
    config = EvalConfig(**{**base, "adapter": adapter, "run_id": run_id, **options})
    run_eval(
        config,
        backend=FakeBackend(reply, model_ref=MODEL),
        allow_dirty=True,
        runs_root=runs,
        workers=1,
    )


def _official_ex(runs: Path, run_id: str) -> float:
    ex = summarize(read_run(runs / run_id).scores).ex_official
    assert ex is not None
    return ex.value


def _promote(runs: Path, *args: str, config: str = "base.yaml") -> Any:
    base = ("--config", str(runs.parent / config))
    return CliRunner().invoke(cli.app, ["runs", "promote", RUN, *base, *args])


@pytest.mark.phase(4)
def test_runs_promote_picks_the_checkpoint_its_val_runs_score_highest(
    runs: Path, tmp_path: Path
) -> None:
    # Each adapter is a folder of the run, named by a path from anywhere, as a kernel's
    # work folder or the laptop's repo root holds it, with a shell's trailing slash or not.
    _evaluate(runs, "half-on-val", f"{runs}/{RUN}/{HALF}/", right=5)
    _evaluate(runs, "final-on-val", f"{runs}/{RUN}/{FINAL}", right=3)
    half, final = _official_ex(runs, "half-on-val"), _official_ex(runs, "final-on-val")
    assert half > final

    done = _promote(runs, "--eval", "final-on-val", "--eval", "half-on-val", "--store", "local")

    assert done.exit_code == 0, done.output
    assert done.stdout.splitlines() == [
        f"{HALF}: official EX {half:.2%} on val, in run half-on-val",
        f"{FINAL}: official EX {final:.2%} on val, in run final-on-val",
        f"best: {HALF}, now checkpoints/best in {runs / RUN / BEST}",
    ]
    record = json.loads((runs / RUN / PROMOTION_FILE).read_text())
    # The scores are the stored runs' own, never typed.
    assert record["scores"] == {HALF: half, FINAL: final}
    assert record["eval_runs"] == {HALF: "half-on-val", FINAL: "final-on-val"}
    weights = (runs / RUN / BEST / "adapter_model.safetensors").read_text()
    assert weights == f"weights of {HALF}"
    assert LocalRunStore(runs).get_manifest(RUN).outputs["best"] == BEST


@pytest.mark.phase(4)
def test_runs_promote_goes_to_the_hub_store(
    runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _evaluate(runs, "final-on-val", f"{runs}/{RUN}/{FINAL}", right=4)
    # The store FORGE_RUN_STORE names, holding the run as it was pulled, and no .env read.
    fake = FakeHub()
    monkeypatch.setattr("huggingface_hub.HfApi", fake.client)
    monkeypatch.setenv("FORGE_RUN_STORE", REPO)
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    HubRunStore(REPO, api=fake).put_folder(RUN, runs / RUN)

    done = _promote(runs, "--eval", "final-on-val")

    assert done.exit_code == 0, done.output
    assert done.stdout.splitlines()[-1] == (
        f"best: {FINAL}, now checkpoints/best in {runs / RUN / BEST} and {REPO}"
    )
    stored = HubRunStore(REPO, api=fake).get_manifest(RUN)
    assert stored.outputs["best"] == BEST and stored == LocalRunStore(runs).get_manifest(RUN)
    assert fake.files[f"runs/{RUN}/{BEST}/adapter_model.safetensors"] == b"weights of " + (
        FINAL.encode()
    )


@pytest.mark.phase(4)
def test_runs_promote_takes_val_runs_of_the_runs_checkpoints_alone(
    runs: Path, tmp_path: Path, with_dev: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_DATA_DIR", str(with_dev))
    _evaluate(runs, "final-on-val", f"{runs}/{RUN}/{FINAL}", right=4)
    _evaluate(runs, "half-on-dev", f"{runs}/{RUN}/{HALF}", right=4, split="dev", reason="test")
    _evaluate(runs, "half-on-train", f"{runs}/{RUN}/{HALF}", right=4, split="train")
    # A part of val, other sampling or a second turn: not the base config with an adapter.
    _evaluate(runs, "half-on-part", f"{runs}/{RUN}/{HALF}", right=4, limit=3)
    _evaluate(runs, "half-sampled", f"{runs}/{RUN}/{HALF}", right=4, temperature=0.7)
    _evaluate(runs, "half-with-retry", f"{runs}/{RUN}/{HALF}", right=4, retry=1)
    sft_run_folder(runs, OTHER, MODEL, [HALF])
    _evaluate(runs, "another-run", f"{runs}/{OTHER}/{HALF}", right=4)
    adapter = Path(shutil.copytree(runs / RUN / FINAL, tmp_path / "adapter"))
    _evaluate(runs, "not-a-checkpoint", str(adapter), right=4)
    _evaluate(runs, "final-again", f"{runs}/{RUN}/{FINAL}", right=2)
    # Other weights in the half-way checkpoint's folder than the run saved there.
    weights = runs / RUN / HALF / "adapter_model.safetensors"
    saved = weights.read_text()
    weights.write_text("weights from elsewhere")
    _evaluate(runs, "other-weights", f"{runs}/{RUN}/{HALF}", right=4)
    weights.write_text(saved)

    differs = "differs from the base config in"
    refused = [
        ("half-on-dev", "run half-on-dev evaluated dev; checkpoints are chosen on val"),
        ("half-on-train", "run half-on-train evaluated train"),
        ("half-on-part", f"run half-on-part {differs} limit, not in its adapter alone"),
        ("half-sampled", f"run half-sampled {differs} generation, not in its adapter alone"),
        ("half-with-retry", f"run half-with-retry {differs} retry, not in its adapter alone"),
        ("another-run", f"run another-run's adapter is {runs}/{OTHER}/{HALF}, no checkpoint"),
        ("not-a-checkpoint", f"run not-a-checkpoint's adapter is {adapter}, no checkpoint of"),
        ("final-again", f"runs final-on-val and final-again both scored {FINAL}; name one"),
        ("other-weights", "run other-weights scored adapter files with sha256 "),
    ]
    for other, message in refused:
        done = _promote(runs, "--eval", "final-on-val", "--eval", other, "--store", "local")
        assert done.exit_code == 1, other
        assert f"error: {message}" in done.stderr, done.stderr
    assert not (runs / RUN / BEST).exists()
    missing = _promote(runs, "--eval", "never-made", "--store", "local")
    assert missing.exit_code == 1 and "error: no run in" in missing.stderr


@pytest.mark.phase(4)
def test_runs_promote_takes_the_base_models_full_val_config_alone(runs: Path) -> None:
    _evaluate(runs, "final-on-val", f"{runs}/{RUN}/{FINAL}", right=4)
    configs = {
        "part.yaml": (BASE.replace("limit: null", "limit: 6"), "scores the first 6 val questions"),
        "dev.yaml": (BASE.replace("split: val", "split: dev"), "scores dev; checkpoints are"),
        "other.yaml": (BASE.replace(MODEL, "Qwen/Qwen3-1.7B"), "scores Qwen/Qwen3-1.7B, and run"),
        "adapted.yaml": (f"{BASE}adapter: org/adapter\n", "names the adapter org/adapter; it"),
    }
    for name, (text, message) in configs.items():
        (runs.parent / name).write_text(text, encoding="utf-8")
        done = _promote(runs, "--eval", "final-on-val", "--store", "local", config=name)
        assert done.exit_code == 1, name
        assert f"error: the base config {message}" in done.stderr, done.stderr
    absent = _promote(runs, "--eval", "final-on-val", "--store", "local", config="absent.yaml")
    assert absent.exit_code == 1 and "error: " in absent.stderr
    assert not (runs / RUN / BEST).exists()
    assert _promote(runs, "--eval", "final-on-val", "--store", "local").exit_code == 0
