"""Choosing a training backend, and what it asks of TRL, without a GPU or a download:
torch's device checks are stood in for."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch
import trl
from datasets import Dataset
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from trl import GRPOConfig

from forge.train.backends import (
    BackendUnavailable,
    Precision,
    TrlBackend,
    check_targets,
    drop_long_prompts,
    peft_lora,
    precision_for,
    select_backend,
)
from forge.train.config import GrpoConfig, LoraConfig, SftConfig
from tests.fakes import WordTokenizer


def _devices(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool = False,
    cuda_bf16: bool = False,
    mps: bool = False,
    macos_14: bool = True,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: cuda_bf16)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda major, minor: macos_14)


@pytest.mark.phase(4)
def test_auto_selects_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    # CI: no CUDA and no MPS.
    _devices(monkeypatch)
    backend = select_backend("auto")
    assert isinstance(backend, TrlBackend)
    assert backend.name == "trl" and backend.device == "cpu"
    assert backend.precision == Precision("float32", fp16=False, bf16=False, use_cpu=True)


@pytest.mark.phase(4)
def test_auto_uses_trl_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _devices(monkeypatch, cuda=True, cuda_bf16=True)
    backend = select_backend()
    assert isinstance(backend, TrlBackend) and backend.device == "cuda"
    assert backend.precision == Precision("bfloat16", bf16=True)


@pytest.mark.phase(4)
def test_backend_says_what_it_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    _devices(monkeypatch)
    with pytest.raises(ValueError, match="a backend is auto or trl, not 'vllm'"):
        select_backend("vllm")  # type: ignore[arg-type]
    with pytest.raises(BackendUnavailable, match="device cuda needs a GPU"):
        TrlBackend("cuda")


@pytest.mark.phase(4)
def test_precision_by_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _devices(monkeypatch, mps=True, macos_14=True)
    assert precision_for("cpu") == Precision("float32", use_cpu=True)
    assert precision_for("mps") == Precision("bfloat16", bf16=True)
    _devices(monkeypatch, mps=True, macos_14=False)
    assert precision_for("mps") == Precision("float32")
    # The pilot's Mac: auto finds MPS when there is no CUDA.
    assert TrlBackend().device == "mps"
    _devices(monkeypatch, cuda=True, cuda_bf16=False)
    assert precision_for("cuda") == Precision("float16", fp16=True)
    _devices(monkeypatch, cuda=True, cuda_bf16=True)
    assert precision_for("cuda") == Precision("bfloat16", bf16=True)
    with pytest.raises(ValueError, match="not 'xpu'"):
        precision_for("xpu")


@pytest.mark.phase(4)
def test_trl_backend_loads_4bit_only_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _devices(monkeypatch)
    # Refused before any download: this repo doesn't exist.
    with pytest.raises(ValueError, match="load_in_4bit needs bitsandbytes on CUDA"):
        TrlBackend("cpu").load("org/no-such-model", LoraConfig(), load_in_4bit=True, seed=0)


def _tiny_qwen() -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    return Qwen2ForCausalLM(config)


@pytest.mark.phase(4)
def test_every_lora_target_must_match_a_module() -> None:
    lora = LoraConfig(r=4, target=("q_proj", "qproj", "o_proj"))
    # PEFT itself is satisfied when any one target matches.
    model = get_peft_model(_tiny_qwen(), peft_lora(lora))
    with pytest.raises(ValueError, match=r"LoRA targets qproj match no module of tiny; it"):
        check_targets(model, lora, "tiny")
    check_targets(get_peft_model(_tiny_qwen(), peft_lora(LoraConfig(r=4))), LoraConfig(), "tiny")


@pytest.mark.phase(4)
def test_long_prompts_are_dropped_and_counted(caplog: pytest.LogCaptureFixture) -> None:
    def row(example_id: str, words: int) -> dict[str, Any]:
        content = " ".join(["word"] * words)
        return {"example_id": example_id, "prompt": [{"role": "user", "content": content}]}

    # WordTokenizer counts the role tag, the words and the opened assistant turn.
    dataset = Dataset.from_list([row("a", 3), row("b", 8), row("c", 5)])
    with caplog.at_level(logging.WARNING, logger="forge.train.backends"):
        kept = drop_long_prompts(dataset, WordTokenizer(), max_prompt_length=7)
    assert kept["example_id"] == ["a", "c"]
    assert "1 of 3 prompts are over max_prompt_length 7 tokens" in caplog.text
    assert drop_long_prompts(dataset, WordTokenizer(), 10) is dataset
    with pytest.raises(ValueError, match="all 3 prompts are over max_prompt_length 4 tokens; "):
        drop_long_prompts(dataset, WordTokenizer(), 4)
    with pytest.raises(ValueError, match="no rows"):
        drop_long_prompts(dataset.select([]), WordTokenizer(), 10)


class Recorder:
    """Stands in for a TRL config or trainer class: keeps what it was made with."""

    made: list[tuple[str, dict[str, Any]]]

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.made.append((type(self).__name__, kwargs))
        self.model = kwargs.get("model")


@pytest.mark.phase(4)
def test_trl_accepts_every_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # TRL's real configs, so a setting TRL renamed or dropped fails here, not on a GPU.
    made: list[tuple[str, dict[str, Any]]] = []
    for name in ("SFTTrainer", "GRPOTrainer"):
        monkeypatch.setattr(trl, name, type(name, (Recorder,), {"made": made}))
    _devices(monkeypatch)
    backend = TrlBackend("cpu")
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": "how many"}]}])
    sft_cfg = SftConfig(max_steps=3, warmup_ratio=0.1)
    backend.sft_trainer("model", WordTokenizer(), dataset, sft_cfg, [], output_dir=tmp_path, seed=5)
    for cfg in (GrpoConfig(max_completion_length=128), GrpoConfig(beta=0.04)):
        backend.grpo_trainer(
            "model", WordTokenizer(), dataset, [], cfg, [], output_dir=tmp_path, seed=5
        )
    sft, grpo, grpo_beta = (kwargs["args"] for _, kwargs in made)

    assert (sft.max_steps, sft.warmup_steps, sft.completion_only_loss) == (3, 0.1, True)
    assert (sft.lr_scheduler_type, sft.max_length, sft.seed) == ("cosine", 2048, 5)
    assert (sft.use_cpu, sft.fp16, sft.bf16, sft.report_to) == (True, False, False, [])
    assert sft.save_strategy == "no" and sft.save_total_limit == 2

    assert (grpo.num_generations, grpo.max_completion_length, grpo.generation_batch_size) == (
        6,
        128,
        24,
    )
    assert grpo.chat_template_kwargs == {"enable_thinking": False}
    assert grpo.beta == GRPOConfig.__dataclass_fields__["beta"].default  # library_default
    assert grpo_beta.beta == 0.04
