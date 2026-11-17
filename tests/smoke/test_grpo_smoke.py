"""GRPO end to end on the tiny random-weight model, on the CPU (FG-504): two steps with
configs/grpo/tiny-smoke.yaml through ``forge train grpo``, the rewards scored on the
fixture databases in the run's sandbox pool, the in-training eval on val, the adapter
saved and evaluated with ``forge eval run --adapter``; a run the time guard stops that
resumes from its checkpoint and goes on counting steps; and a run that starts from an
SFT adapter.

The model is random, so its replies carry no SQL and every reward is 0. One test steers
its rollouts to a query, so a GRPO step's rewards run real SQL and pay for it.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import trl
from safetensors.torch import load_file
from typer.testing import CliRunner

from forge import cli
from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.data.splits import load_train, load_val
from forge.eval.gold_cache import split_gold
from forge.eval.run_files import ADAPTER_DIGEST, RunConflict
from forge.ops.adapters import adapter_sha256, checkpoint_digests, digest_output
from forge.ops.hub import LocalRunStore
from forge.ops.manifest import config_hash
from forge.ops.sessions import read_sessions
from forge.train import backends
from forge.train.backends import TrlBackend
from forge.train.callbacks import MetricsCallback
from forge.train.config import TrainConfig, load_train_config
from forge.train.grpo import grpo_rows, run_grpo
from forge.train.rewards import Rewards, reward_pool
from forge.train.sft import RunExists
from tests.fakes import sft_run_folder
from tests.report_trace import read_stored

ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "configs" / "grpo" / "tiny-smoke.yaml"
REWARDS = ("execution_reward", "validity_reward", "format_reward", "guard_penalty")


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _with_grpo(config: TrainConfig, **changes: Any) -> TrainConfig:
    assert config.grpo is not None
    return config.model_copy(update={"grpo": config.grpo.model_copy(update=changes)})


@pytest.mark.phase(5)
def test_grpo_two_steps_cpu(
    spider_mini: Path, tiny_model_ref: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("FORGE_RUNS_DIR", str(runs))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    cli_runner = CliRunner()
    train = ["train", "grpo", "--config", str(SMOKE_CONFIG), "--run-id", "grpo", "--allow-dirty"]

    trained = cli_runner.invoke(cli.app, train)
    assert trained.exit_code == 0, trained.output
    assert "TRL 1.18.0 with beta 0.0 and the dapo loss" in trained.output

    run_dir = runs / "grpo"
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_bytes())
    assert (manifest.kind, manifest.status, manifest.platform) == ("grpo", "finished", "local-cpu")
    assert (manifest.model_ref, manifest.adapter_ref, manifest.parent_run) == (
        tiny_model_ref,
        None,
        None,
    )
    config = manifest.config
    assert (config["grpo"]["num_generations"], config["grpo"]["max_completion_length"]) == (2, 16)
    assert config["init_adapter_sha256"] is None  # a fresh LoRA
    # What TRL trained with where the config left it to TRL: no reference model, and the
    # dapo loss, in the TRL the lock pins.
    assert config["trl"] == {
        "version": trl.__version__,
        "beta": 0.0,
        "loss_type": "dapo",
        "scale_rewards": "group",
        "epsilon": 0.2,
        "top_p": 1.0,
        "top_k": 0,
    }
    # The fixture's 40 train questions, all of them prompts within 1,536 tokens.
    assert config["grpo_data"]["rows"] == len(load_train()) == 40
    assert manifest.config_hash == config_hash(config)
    assert manifest.data_manifest_hash == data_manifest_hash(spider_mini)

    lines = _lines(run_dir / "metrics.jsonl")
    steps = [line for line in lines if "reward" in line]
    # Two steps, each logging every reward component's mean and spread.
    assert [line["step"] for line in steps] == [1, 2]
    for line in steps:
        for name in REWARDS:
            assert {f"rewards/{name}/mean", f"rewards/{name}/std"} <= line.keys(), name
    # The in-training eval at step 2, on every one of the fixture's val questions.
    (evaluated,) = [line for line in lines if "val/ex_fast" in line]
    assert evaluated["step"] == 2 and evaluated["val/questions"] == len(load_val())
    assert 0.0 <= evaluated["val/ex_fast"] <= evaluated["val/valid"] <= 1.0

    # A checkpoint each step, and the adapter where the manifest says.
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == [
        "final",
        "step-000001",
        "step-000002",
    ]
    adapter = run_dir / manifest.outputs["adapter"]
    assert adapter == run_dir / "checkpoints" / "final"
    weights = load_file(adapter / "adapter_model.safetensors")
    assert weights and all("lora_" in name for name in weights)
    (session,) = read_sessions(run_dir) or []
    assert session.status == "finished" and manifest.outputs["sessions"] == "sessions.json"

    evaluate = [
        *("eval", "run", "--backend", "hf", "--model", tiny_model_ref, "--adapter", str(adapter)),
        *("--split", "val", "--limit", "10", "--run-id", "grpo-eval", "--workers", "2"),
        "--allow-dirty",
    ]
    scored = cli_runner.invoke(cli.app, evaluate)
    assert scored.exit_code == 0, scored.output
    stored = read_stored(runs / "grpo-eval", spider_mini)
    assert stored.manifest.status == "finished" and len(stored.scores) == 10
    # Its scores are those of the weights the GRPO run saved, by their digest.
    saved = manifest.outputs[digest_output("checkpoints/final")]
    assert stored.manifest.outputs[ADAPTER_DIGEST] == saved == adapter_sha256(adapter)
    assert all(score.ex_official is not None for score in stored.scores)


@pytest.mark.phase(5)
def test_grpo_resume_continues_steps(
    spider_mini: Path, tiny_model_ref: str, tmp_path: Path
) -> None:
    config = load_train_config(SMOKE_CONFIG)
    runs, cache, run_id = tmp_path / "runs", tmp_path / "cache", "grpo-resume"
    # The first session's clock moves an hour a reading against a one-minute deadline, so
    # the time guard stops training after the first step and saves it.
    hours = itertools.count()
    guarded = config.model_copy(
        update={"ops": config.ops.model_copy(update={"time_guard_minutes": 1.0})}
    )
    stopped = run_grpo(
        guarded,
        run_id=run_id,
        allow_dirty=True,
        runs_root=runs,
        cache_root=cache,
        env={},
        clock=lambda: next(hours) * 3600.0,
    )
    assert (stopped.stopped_at, stopped.adapter) == (1, None)
    assert stopped.manifest.status == "stopped_time_guard"
    store = LocalRunStore(runs)
    assert store.latest_checkpoint(run_id) == "checkpoints/step-000001"

    # The next session runs the config as written: the ops section changes no weight, so
    # the config hash is the one the run was made with.
    resumed = run_grpo(
        config,
        run_id=run_id,
        resume=True,
        allow_dirty=True,
        runs_root=runs,
        cache_root=cache,
        env={},
    )

    assert resumed.resumed_from == "checkpoints/step-000001"
    assert (resumed.manifest.status, resumed.stopped_at) == ("finished", None)
    assert resumed.manifest.started_at == stopped.manifest.started_at
    assert resumed.manifest.config_hash == stopped.manifest.config_hash
    # The step count went on: the second session trained step 2 alone, not 1 and 2 again.
    run_dir = runs / run_id
    lines = _lines(run_dir / "metrics.jsonl")
    assert [line["step"] for line in lines if "reward" in line] == [1, 2]
    for step in (1, 2):
        state = json.loads(
            (run_dir / f"checkpoints/step-{step:06d}/trainer_state.json").read_text()
        )
        assert state["global_step"] == step
    assert resumed.adapter is not None and (resumed.adapter / "adapter_model.safetensors").is_file()
    # Two sessions, the boundary between them in the run's record.
    assert [s.status for s in read_sessions(run_dir) or []] == ["stopped_time_guard", "finished"]

    # A finished run has nothing to resume, and its id names no new run.
    with pytest.raises(RunConflict, match=f"run {run_id} has finished"):
        run_grpo(config, run_id=run_id, resume=True, allow_dirty=True, runs_root=runs, env={})
    with pytest.raises(RunExists, match="pass --resume or a new --run-id"):
        run_grpo(config, run_id=run_id, allow_dirty=True, runs_root=runs, env={})


@pytest.mark.phase(5)
def test_grpo_starts_from_an_sft_adapter(
    spider_mini: Path, tiny_model_ref: str, tmp_path: Path
) -> None:
    config = load_train_config(SMOKE_CONFIG)
    runs, sft = tmp_path / "runs", "2026-11-20-tiny-sft-abc1234"
    # An SFT run's checkpoints/best, as forge runs promote leaves one: LoRA on the tiny
    # model at the config's settings, its B moved off the zeros a fresh LoRA starts at.
    made = sft_run_folder(runs, sft, tiny_model_ref, [])
    lora, _ = TrlBackend("cpu").load(tiny_model_ref, config.lora, load_in_4bit=False, seed=0)
    with torch.no_grad():
        for name, param in lora.named_parameters():
            if "lora_B" in name:
                param.fill_(0.01)
    best = runs / sft / "checkpoints" / "best"
    lora.save_pretrained(best)
    outputs = {**made.outputs, **checkpoint_digests(runs / sft)}
    LocalRunStore(runs).put_manifest(made.model_copy(update={"outputs": outputs}))

    run = run_grpo(
        _with_grpo(config, init_adapter=str(best), eval_every=None),
        run_id="grpo-from-sft",
        allow_dirty=True,
        runs_root=runs,
        cache_root=tmp_path / "cache",
        env={},
    )

    assert (run.manifest.status, run.manifest.parent_run) == ("finished", sft)
    assert run.manifest.adapter_ref == str(best)
    assert run.manifest.config["init_adapter_sha256"] == adapter_sha256(best)
    # It trained on from the adapter: two steps at 5e-6 move B by far less than the 0.01
    # between the adapter's B and a fresh LoRA's.
    assert run.adapter is not None
    final = load_file(run.adapter / "adapter_model.safetensors")
    lora_b = [weight for name, weight in final.items() if "lora_B" in name]
    assert lora_b and all(
        torch.allclose(weight, torch.full_like(weight, 0.01), atol=1e-3) for weight in lora_b
    )


@pytest.mark.phase(5)
def test_grpo_rewards_score_replies_on_the_fixture_databases(
    spider_mini: Path,
    tiny_model_ref: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The two fixture questions whose gold query is this one, and the tiny model's
    # rollouts steered to write it: each token of it, and then the end of the turn, gets
    # a bias that grows with how much of it is written already.
    query = "SELECT dorm_name FROM dorm WHERE gender  =  'F'"
    # Room for the whole reply, which is longer than the smoke's 16 tokens.
    config = _with_grpo(load_train_config(SMOKE_CONFIG), eval_every=None, max_completion_length=40)
    assert config.grpo is not None
    backend = TrlBackend("cpu")
    model, tokenizer = backend.load(tiny_model_ref, config.lora, load_in_4bit=False, seed=0)
    rows, _ = grpo_rows(config, tokenizer, data_dir=spider_mini)
    asked = [i for i, row in enumerate(rows) if row["gold_sql"] == query]
    assert len(asked) == 2
    reply = tokenizer.encode(f"```sql\n{query}\n```", add_special_tokens=False)
    ids = [*reply, tokenizer.eos_token_id]
    bias = [[ids[: i + 1], 100.0 * (i + 1)] for i in range(len(ids))]
    settings = backends.grpo_settings

    def steered(*args: Any) -> dict[str, Any]:
        return {**settings(*args), "generation_kwargs": {"sequence_bias": bias}}

    monkeypatch.setattr(backends, "grpo_settings", steered)
    gold = split_gold("train", data_dir=spider_mini, cache_root=tmp_path / "cache").results
    metrics = tmp_path / "metrics.jsonl"
    with reward_pool(spider_mini, workers=1) as pool:
        rewards = Rewards(gold, pool, config.rewards)
        backend.grpo(
            model,
            tokenizer,
            rows.select(asked),
            rewards.functions,
            config.grpo,
            [MetricsCallback(metrics)],
            output_dir=tmp_path / "out",
            seed=0,
            reward_weights=rewards.weights,
        )

    steps = [line for line in _lines(metrics) if "reward" in line]
    assert [line["step"] for line in steps] == [1, 2]
    for line in steps:
        # The query ran on dorm_1 and matched its gold rows, in the asked-for format, and
        # set off no guard: 1 + 0.1 + 0.05 by the config's weights.
        means = [line[f"rewards/{name}/mean"] for name in REWARDS]
        assert means == [1.0, 1.0, 1.0, 0.0]
        assert line["reward"] == pytest.approx(1.15)
        assert line["completions/mean_length"] == len(ids)  # it ended its turn there
