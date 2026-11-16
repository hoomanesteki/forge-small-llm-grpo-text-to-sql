"""The spike's GRPO bench through real TRL and PEFT code, on CPU with a random model built
offline (``tests/tiny_hf.py``), so the code a T4 session will run has run in CI first."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.spike import throughput
from tests.tiny_hf import QWEN3, byte_tokenizer, save_random_model


@pytest.mark.phase(3)
def test_grpo_bench_trains_on_cpu(tmp_path: Path) -> None:
    spike = throughput()
    model = save_random_model(tmp_path / "model")
    # The byte tokenizer spends a token a byte, and the spike's prompts run past the
    # 1,024 positions the helper gives the model.
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    config["max_position_embeddings"] = 4096
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    # Qwen3's thinking switch, which the bench renders every prompt with, turned off.
    byte_tokenizer(QWEN3).save_pretrained(model)

    out = tmp_path / "grpo.json"
    argv = [
        "grpo-bench",
        "--out",
        str(out),
        "--model",
        str(model),
        "--stack",
        "trl",
        "--device",
        "cpu",
        "--quant",
        "none",
        "--work-dir",
        str(tmp_path / "work"),
        "--prompts",
        "4",
        "--steps",
        "2",
        "--num-generations",
        "2",
        "--batch-size",
        "2",
        "--grad-accum",
        "1",
        "--max-completion-length",
        "8",
        "--lora-rank",
        "8",
    ]
    code = spike.main(argv)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0, written.get("error")
    result = written["result"]

    assert (result["stack"], result["device"], result["quant"]) == ("trl", "cpu", "none")
    assert len(result["step_seconds"]) == 2
    assert all(seconds > 0 for seconds in result["step_seconds"])
    assert result["median_step_seconds"] == result["step_seconds"][1]
    # What TRL scored, counted by the reward, against what the settings ask for; no GPU
    # here, so the trainer counts none.
    assert result["completions_per_step"] == result["planned_completions_per_step"] == 2
    assert result["n_gpu"] == 0
    assert result["thinking_off"] is True
    history = result["log_history"]
    lengths = [entry["completions/mean_length"] for entry in history if "reward" in entry]
    assert len(lengths) == 2
    assert result["mean_completion_length"] == pytest.approx(sum(lengths) / len(lengths))
    assert 0 < result["mean_completion_length"] <= 8
    settings = result["settings"]
    assert (settings["fp16"], settings["bf16"], settings["use_cpu"]) == (False, False, True)
    # The prompts arrive rendered, so no stack's trainer applies a template of its own.
    assert "chat_template_kwargs" not in settings
    assert settings["generates_with_vllm"] is False
    assert result["memory"] is None  # no CUDA here
    assert result["versions"]["trl"] is not None
    # TRL logged a reward every step: the stand-in reward reached the trainer.
    rewards = [entry["reward"] for entry in result["log_history"] if "reward" in entry]
    assert len(rewards) == 2
    assert all(0.0 <= reward <= 1.0 for reward in rewards)
