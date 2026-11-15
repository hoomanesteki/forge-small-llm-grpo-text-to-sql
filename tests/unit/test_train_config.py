"""Training configs from YAML: what loads, and what is refused before any model loads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge.data.prompts import MAX_PROMPT_TOKENS
from forge.train.config import (
    LORA_TARGETS,
    TrainConfig,
    load_train_config,
    parse_train_config,
)

# Plan section 3.10's GRPO file, without the sections later stories add.
GRPO_FILE = """\
run:
  kind: grpo
  model_ref: Qwen/Qwen3-4B@0123abc
  seed: 42
data:
  prompt_version: p1
  split: train
lora:
  r: 16
  alpha: 32
  dropout: 0.0
  target: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
grpo:
  num_generations: 6
  max_prompt_length: 1536
  max_completion_length: 256
  learning_rate: 5e-6
  per_device_train_batch_size: 6
  gradient_accumulation_steps: 4
  max_steps: 600
  beta: library_default
ops:
  checkpoint_every: 50
  time_guard_minutes: 675
"""

SFT_FILE = """\
run: {kind: sft, model_ref: trl-internal-testing/tiny-Qwen2ForCausalLM-2.5, device: cpu}
sft: {max_steps: 3, per_device_train_batch_size: 2, gradient_accumulation_steps: 1}
ops: {est_gpu_hours: 0.5}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _sft(**sections: Any) -> dict[str, Any]:
    return {"run": {"kind": "sft", "model_ref": "org/model"}, "sft": {}, **sections}


def _grpo(**grpo: Any) -> dict[str, Any]:
    return {"run": {"kind": "grpo", "model_ref": "org/model"}, "grpo": grpo}


@pytest.mark.phase(4)
def test_config_validation(tmp_path: Path) -> None:
    config = load_train_config(_write(tmp_path, GRPO_FILE))
    assert config.run.kind == "grpo" and config.run.seed == 42 and config.sft is None
    assert config.grpo is not None
    # 5e-6 is text to PyYAML; the loader reads the number it means.
    assert config.grpo.learning_rate == 5e-6
    assert config.grpo.beta == "library_default"
    assert config.grpo.generation_batch == 24
    assert config.lora.target == LORA_TARGETS
    assert config.ops.time_guard_minutes == 675.0
    assert config.max_seq_length == 1536 + 256

    sft = load_train_config(_write(tmp_path, SFT_FILE))
    assert sft.sft is not None and sft.sft.max_steps == 3
    assert (sft.sft.learning_rate, sft.sft.lr_scheduler_type) == (2e-4, "cosine")  # ADR-0010
    assert sft.run.device == "cpu" and sft.data.prompt_version == "p1"
    assert sft.ops.est_gpu_hours == 0.5 and sft.ops.checkpoint_every == 50

    refused = [
        (
            GRPO_FILE.replace("max_completion_length: 256", "max_completion_length: 513"),
            r"grpo\.max_completion_length: Input should be less than or equal to 512",
        ),
        (
            GRPO_FILE.replace("num_generations: 6", "num_generations: 5"),
            r"num_generations 5 must divide the generation batch.* = 6 x 4 = 24",
        ),
        (
            GRPO_FILE.replace("checkpoint_every", "checkpoint_evry"),
            r"ops\.checkpoint_evry: Extra inputs are not permitted",
        ),
        (
            GRPO_FILE.replace("split: train", "split: val"),
            r"data\.split: training reads the train split only, never 'val'",
        ),
        (
            GRPO_FILE.replace("max_steps: 600", "max_steps: true"),
            r"grpo\.max_steps: Input should be a valid integer",
        ),
        (GRPO_FILE + "ops: {checkpoint_every: 10}\n", "'ops' is set twice"),
    ]
    for text, message in refused:
        with pytest.raises(ValueError, match=message):
            load_train_config(_write(tmp_path, text))


@pytest.mark.phase(4)
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_sft(lora={"r": 0}), r"lora\.r: Input should be greater than or equal to 1"),
        (_sft(lora={"dropout": 1.0}), r"lora\.dropout: Input should be less than 1"),
        (_sft(lora={"target": []}), r"lora\.target: LoRA needs at least one target module"),
        (_sft(lora={"target": ["q_proj", "q_proj"]}), "target names q_proj more than once"),
        (_sft(data={"prompt_version": "p9"}), "unknown prompt version 'p9'"),
        (_sft(sft={"learning_rate": 0}), r"sft\.learning_rate: Input should be greater than 0"),
        (_sft(sft={"max_steps": 2.5}), r"sft\.max_steps: Input should be a valid integer"),
        (_sft(sft={"warmup_ratio": 1}), r"sft\.warmup_ratio: Input should be less than 1"),
        (_sft(ops={"time_guard_minutes": float("inf")}), "Input should be a finite number"),
        (_sft(ops={"checkpoint_every": 0}), r"ops\.checkpoint_every: Input should be greater"),
        (_grpo(num_generations=1), r"grpo\.num_generations: Input should be greater than or"),
        (_grpo(max_completion_length=0), r"grpo\.max_completion_length: Input should be greater"),
        (_grpo(beta=-0.1), r"grpo\.beta.*greater than or equal to 0"),
        (_grpo(beta="none"), r"grpo\.beta.*Input should be 'library_default'"),
        (_grpo(gpu_memory_utilization=1.5), r"gpu_memory_utilization: Input should be less"),
        (_grpo(max_prompt_length=0), r"grpo\.max_prompt_length: Input should be greater"),
        ({"run": {"kind": "sft", "model_ref": "org/model"}}, "needs the sft section"),
        ({**_sft(), "grpo": {}}, "run.kind is sft, so the file can't have the grpo section"),
        ({"run": {"kind": "eval", "model_ref": "org/model"}, "sft": {}}, r"run\.kind: Input"),
        ({"run": {"kind": "sft", "model_ref": ""}, "sft": {}}, r"run\.model_ref: String should"),
        ({"sft": {}}, r"run: Field required"),
        (
            _sft(
                run={"kind": "sft", "model_ref": "org/model", "backend": "unsloth", "device": "mps"}
            ),
            "backend unsloth trains on CUDA only, not on device mps",
        ),
        (
            _sft(
                run={"kind": "sft", "model_ref": "org/model", "load_in_4bit": True, "device": "cpu"}
            ),
            "load_in_4bit needs bitsandbytes on CUDA, not device cpu",
        ),
        (_sft(run={"kind": "sft", "model_ref": "org/model", "device": "tpu"}), r"run\.device"),
    ],
)
def test_config_refuses(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_train_config(data, where="test.yaml")


@pytest.mark.phase(4)
def test_config_defaults_are_the_adr_starting_values() -> None:
    config = parse_train_config(_grpo())
    assert isinstance(config, TrainConfig) and config.grpo is not None
    grpo = config.grpo
    assert (grpo.num_generations, grpo.max_completion_length, grpo.max_steps) == (6, 256, 600)
    assert grpo.max_prompt_length == MAX_PROMPT_TOKENS
    assert (config.lora.r, config.lora.alpha, config.lora.dropout) == (16, 32, 0.0)
    assert (config.run.backend, config.run.device, config.run.load_in_4bit) == (
        "auto",
        "auto",
        False,
    )
    # The CUDA-only options are fine where the device may be CUDA.
    cuda = {"kind": "sft", "model_ref": "org/model", "backend": "unsloth", "load_in_4bit": True}
    assert parse_train_config(_sft(run=cuda)).run.load_in_4bit


@pytest.mark.phase(4)
def test_config_file_must_be_a_yaml_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not valid YAML"):
        load_train_config(_write(tmp_path, "run: [kind: sft"))
    with pytest.raises(ValueError, match="must be a mapping of sections"):
        load_train_config(_write(tmp_path, "- run\n- sft\n"))
    with pytest.raises(OSError):
        load_train_config(tmp_path / "missing.yaml")
    # Every wrong field is named, after the file it is in.
    text = SFT_FILE.replace("max_steps: 3", "max_steps: 0").replace("0.5}", "0.5, seed: 1}")
    with pytest.raises(ValueError, match=r"config\.yaml: sft\.max_steps: .*; ops\.seed: Extra"):
        load_train_config(_write(tmp_path, text))
