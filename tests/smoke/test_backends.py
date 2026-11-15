"""The TRL backend on the tiny model pinned in configs/models.yaml, on CPU: LoRA where
the config says, and SFT and GRPO trained a few steps with the callbacks every job has,
the checkpoints and metrics going to a local run store."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from forge.core.types import RunManifest
from forge.data.prompts import SYSTEM_PROMPT, count_chat_tokens
from forge.infer.hf_backend import split_ref
from forge.ops.hub import LocalRunStore
from forge.train.backends import ADAPTER_DIR, TrlBackend, select_backend
from forge.train.callbacks import training_callbacks
from forge.train.config import GrpoConfig, LoraConfig, OpsConfig, SftConfig
from forge.train.sft_data import sql_block

RUN = "2026-11-15-tiny-sft-abc1234"


def _store(tmp_path: Path, kind: str) -> LocalRunStore:
    store = LocalRunStore(tmp_path / "store")
    store.put_manifest(
        RunManifest(
            run_id=RUN,
            kind=kind,  # type: ignore[arg-type]
            git_sha="a" * 40,
            config={},
            config_hash="1" * 64,
            data_manifest_hash="2" * 64,
            model_ref="tiny",
            platform="local-cpu",
            seed=0,
            started_at=datetime.now(UTC),
        )
    )
    return store


def _prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Database schema:\nCREATE TABLE t (id INT);\n\n{question}"},
    ]


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.phase(4)
def test_lora_applied_to_tiny_model(tiny_model_ref: str) -> None:
    lora = LoraConfig(r=8, alpha=16)
    model, tokenizer = TrlBackend("cpu").load(tiny_model_ref, lora, load_in_4bit=False, seed=0)
    assert isinstance(model, PeftModel)
    assert tokenizer.chat_template

    params = dict(model.named_parameters())
    trainable = {name: p for name, p in params.items() if p.requires_grad}
    frozen = {name: p for name, p in params.items() if not p.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert not any("lora_" in name for name in frozen)
    # The frozen part is the whole base model.
    repo, revision = split_ref(tiny_model_ref)
    base = AutoModelForCausalLM.from_pretrained(repo, revision=revision)
    assert sum(p.numel() for p in frozen.values()) == sum(p.numel() for p in base.parameters())

    # Every target in every layer, each adding r x (in + out) weights.
    targeted = model.base_model.targeted_module_names
    for target in lora.target:
        hits = [name for name in targeted if name.endswith(f".{target}")]
        assert len(hits) == base.config.num_hidden_layers, target
    expected = 0
    for name in targeted:
        # Names are within the model PEFT wraps.
        layer = model.base_model.model.get_submodule(name)
        expected += lora.r * (layer.in_features + layer.out_features)
    assert sum(p.numel() for p in trainable.values()) == expected > 0

    # The seed draws LoRA's first weights.
    again, _ = TrlBackend("cpu").load(tiny_model_ref, lora, load_in_4bit=False, seed=0)
    other, _ = TrlBackend("cpu").load(tiny_model_ref, lora, load_in_4bit=False, seed=1)
    first_a = next(name for name in trainable if "lora_A" in name)
    assert torch.equal(dict(again.named_parameters())[first_a], trainable[first_a])
    assert not torch.equal(dict(other.named_parameters())[first_a], trainable[first_a])


@pytest.mark.phase(4)
def test_sft_trains_with_the_job_callbacks(tmp_path: Path, tiny_model_ref: str) -> None:
    store = _store(tmp_path, "sft")
    metrics = tmp_path / "run" / "metrics.jsonl"
    ops = OpsConfig(checkpoint_every=2)
    callbacks = training_callbacks(ops, store, RUN, metrics_path=metrics, env={})
    backend = select_backend("trl", device="cpu")
    model, tokenizer = backend.load(tiny_model_ref, LoraConfig(r=4), load_in_4bit=False, seed=0)
    rows = [
        {
            "example_id": f"spider-train-{i:05d}",
            "db_id": "t",
            "prompt": _prompt(f"Question: How many rows have id {i}?"),
            "completion": [
                {"role": "assistant", "content": sql_block(f"SELECT id FROM t WHERE id = {i}")}
            ],
        }
        for i in range(8)
    ]
    cfg = SftConfig(
        max_steps=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        learning_rate=1e-2,
    )
    out = tmp_path / "out"

    adapter = Path(
        backend.sft(
            model, tokenizer, Dataset.from_list(rows), cfg, callbacks, output_dir=out, seed=0
        )
    )

    assert adapter == out / ADAPTER_DIR
    assert {"adapter_model.safetensors", "adapter_config.json"} <= {
        p.name for p in adapter.iterdir()
    }
    weights = load_file(adapter / "adapter_model.safetensors")
    assert weights and all("lora_" in name for name in weights)
    # LoRA's B starts at zero; trained, it no longer is.
    assert any(t.abs().sum() > 0 for name, t in weights.items() if "lora_B" in name)
    # The trainer saved only when the checkpoint callback asked, and each went to the store.
    assert sorted(p.name for p in out.glob("checkpoint-*")) == ["checkpoint-2", "checkpoint-4"]
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000004"
    assert (
        store.root / RUN / "checkpoints" / "step-000002" / "adapter_model.safetensors"
    ).is_file()
    lines = _lines(metrics)
    assert [line["step"] for line in lines if "loss" in line] == [1, 2, 3, 4]
    assert "train_runtime" in lines[-1]
    assert (store.root / RUN / "metrics.jsonl").read_bytes() == metrics.read_bytes()
    assert store.get_manifest(RUN).status == "running"  # the time guard never fired


@pytest.mark.phase(4)
def test_grpo_trains_with_the_job_callbacks(
    tmp_path: Path, tiny_model_ref: str, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path, "grpo")
    metrics = tmp_path / "run" / "metrics.jsonl"
    callbacks = training_callbacks(
        OpsConfig(checkpoint_every=2), store, RUN, metrics_path=metrics, env={}
    )
    backend = TrlBackend("cpu")
    model, tokenizer = backend.load(tiny_model_ref, LoraConfig(r=4), load_in_4bit=False, seed=0)
    rows = [
        {
            "example_id": f"spider-train-{i:05d}",
            "db_id": "t",
            "prompt": _prompt(f"Question: id {i}?"),
        }
        for i in range(4)
    ]
    limit = max(count_chat_tokens(tokenizer, row["prompt"]) for row in rows)
    long_question = "Question: " + " ".join(["which"] * 200) + "?"
    rows.append(
        {"example_id": "spider-train-09999", "db_id": "t", "prompt": _prompt(long_question)}
    )
    seen: list[str] = []

    def length_reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        # The dataset's other columns reach the reward, aligned with the completions.
        assert len(kwargs["example_id"]) == len(kwargs["db_id"]) == len(completions)
        seen.extend(kwargs["example_id"])
        return [float(len(completion[0]["content"])) for completion in completions]

    cfg = GrpoConfig(
        num_generations=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=2,
        max_completion_length=8,
        max_prompt_length=limit,
    )
    with caplog.at_level(logging.WARNING, logger="forge.train.backends"):
        adapter = backend.grpo(
            model,
            tokenizer,
            Dataset.from_list(rows),
            [length_reward],
            cfg,
            callbacks,
            output_dir=tmp_path / "out",
            seed=0,
        )

    assert f"1 of 5 prompts are over max_prompt_length {limit} tokens" in caplog.text
    assert seen and "spider-train-09999" not in seen
    assert (Path(adapter) / "adapter_model.safetensors").is_file()
    assert store.latest_checkpoint(RUN) == "checkpoints/step-000002"
    lines = _lines(metrics)
    assert [line["step"] for line in lines if "reward" in line] == [1, 2]
    assert all("rewards/length_reward/mean" in line for line in lines if "reward" in line)
