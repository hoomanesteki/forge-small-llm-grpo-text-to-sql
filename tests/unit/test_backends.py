"""Choosing a training backend, and what each one asks of TRL, without a GPU or a
download: torch's device checks and Unsloth are stood in for."""

from __future__ import annotations

import importlib.machinery
import logging
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

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
    UnslothBackend,
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


class FastLanguageModel:
    """Unsloth's loader, recording what it is asked for."""

    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> tuple[Any, Any]:
        cls.calls.append(("from_pretrained", kwargs))
        return "model", "tokenizer"

    @classmethod
    def get_peft_model(cls, model: Any, **kwargs: Any) -> Any:
        cls.calls.append(("get_peft_model", {"model": model, **kwargs}))
        names = [f"model.layers.0.mlp.{target}" for target in kwargs["target_modules"]]
        return types.SimpleNamespace(base_model=types.SimpleNamespace(targeted_module_names=names))


@pytest.fixture
def unsloth(monkeypatch: pytest.MonkeyPatch) -> type[FastLanguageModel]:
    """An installed Unsloth, as far as the backend can tell."""
    module = types.ModuleType("unsloth")
    module.__spec__ = importlib.machinery.ModuleSpec("unsloth", None)
    module.FastLanguageModel = FastLanguageModel  # type: ignore[attr-defined]
    FastLanguageModel.calls = []
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return FastLanguageModel


@pytest.fixture
def no_unsloth(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry makes both find_spec and import say there is no such package.
    monkeypatch.setitem(sys.modules, "unsloth", None)


@pytest.mark.phase(4)
def test_auto_selects_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch, unsloth: type[FastLanguageModel]
) -> None:
    # CI: no CUDA and no MPS. Unsloth being installed changes nothing without a GPU.
    _devices(monkeypatch)
    backend = select_backend("auto")
    assert isinstance(backend, TrlBackend)
    assert backend.name == "trl" and backend.device == "cpu"
    assert backend.precision == Precision("float32", fp16=False, bf16=False, use_cpu=True)
    assert unsloth.calls == []


@pytest.mark.phase(4)
def test_auto_selects_unsloth_with_cuda(
    monkeypatch: pytest.MonkeyPatch, unsloth: type[FastLanguageModel]
) -> None:
    _devices(monkeypatch, cuda=True)  # a T4: no bf16
    backend = select_backend("auto", max_seq_length=1792, gpu_memory_utilization=0.6)
    assert isinstance(backend, UnslothBackend)
    assert backend.precision == Precision("float16", fp16=True)
    # A device that can't be CUDA keeps auto on the TRL backend.
    assert isinstance(select_backend("auto", device="cpu"), TrlBackend)

    _, tok = backend.load("unsloth/Qwen3-1.7B@abc123", LoraConfig(r=16), load_in_4bit=True, seed=7)
    (load, loaded), (peft, peft_args) = unsloth.calls
    assert (load, peft) == ("from_pretrained", "get_peft_model")
    assert loaded == {
        "model_name": "unsloth/Qwen3-1.7B",
        "revision": "abc123",
        "max_seq_length": 1792,
        "load_in_4bit": True,
        "dtype": torch.float16,
        "fast_inference": True,
        "gpu_memory_utilization": 0.6,
        "max_lora_rank": 16,
    }
    assert peft_args["target_modules"] == list(LoraConfig().target)
    assert (peft_args["r"], peft_args["lora_alpha"], peft_args["random_state"]) == (16, 32, 7)
    assert tok == "tokenizer"

    # Without a memory share, GRPO generates through TRL and Unsloth starts no vLLM.
    unsloth.calls.clear()
    UnslothBackend().load("org/model", LoraConfig(), load_in_4bit=False, seed=0)
    assert "fast_inference" not in unsloth.calls[0][1]


@pytest.mark.phase(4)
def test_auto_uses_trl_on_cuda_without_unsloth(
    monkeypatch: pytest.MonkeyPatch, no_unsloth: None
) -> None:
    _devices(monkeypatch, cuda=True, cuda_bf16=True)
    backend = select_backend()
    assert isinstance(backend, TrlBackend) and backend.device == "cuda"
    assert backend.precision == Precision("bfloat16", bf16=True)


@pytest.mark.phase(4)
def test_unsloth_backend_says_what_it_lacks(
    monkeypatch: pytest.MonkeyPatch, no_unsloth: None
) -> None:
    _devices(monkeypatch)
    with pytest.raises(BackendUnavailable, match="trains on a CUDA GPU, and torch sees none"):
        select_backend("unsloth")
    _devices(monkeypatch, cuda=True)
    with pytest.raises(BackendUnavailable, match="needs the unsloth package"):
        select_backend("unsloth")
    with pytest.raises(ValueError, match="trains on CUDA only, not on device mps"):
        select_backend("unsloth", device="mps")
    with pytest.raises(ValueError, match="a backend is auto, trl or unsloth, not 'vllm'"):
        select_backend("vllm")  # type: ignore[arg-type]
    _devices(monkeypatch)
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
    # A target that ends a module's name at a dot is one PEFT adapts, so it counts; one that
    # ends it inside a part, or names a path the model lacks, does not.
    suffixes = LoraConfig(r=4, target=("self_attn.q_proj", "mlp.down_proj"))
    model = get_peft_model(_tiny_qwen(), peft_lora(suffixes))
    assert "model.layers.0.self_attn.q_proj" in model.base_model.targeted_module_names
    check_targets(model, suffixes, "tiny")
    partial = LoraConfig(r=4, target=("self_attn.q_proj", "attn.q_proj", "mlp.q_proj"))
    model = get_peft_model(_tiny_qwen(), peft_lora(partial))
    missing = r"targets attn\.q_proj, mlp\.q_proj match no module of tiny; it matched q_proj$"
    with pytest.raises(ValueError, match=missing):
        check_targets(model, partial, "tiny")


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
def test_both_backends_build_trainers_alike(
    monkeypatch: pytest.MonkeyPatch, unsloth: type[FastLanguageModel], tmp_path: Path
) -> None:
    made: list[tuple[str, dict[str, Any]]] = []
    for name in ("SFTConfig", "SFTTrainer", "GRPOConfig", "GRPOTrainer"):
        monkeypatch.setattr(trl, name, type(name, (Recorder,), {"made": made}))
    _devices(monkeypatch, cuda=True)
    callbacks: list[Any] = [object(), object()]
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": "how many"}]}])

    def reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        return [0.0] * len(completions)

    built = {}
    for backend in (TrlBackend("cuda"), UnslothBackend()):
        made.clear()
        backend.sft_trainer(
            "model", WordTokenizer(), dataset, SftConfig(), callbacks, output_dir=tmp_path, seed=3
        )
        backend.grpo_trainer(
            "model",
            WordTokenizer(),
            dataset,
            [reward],
            GrpoConfig(),
            callbacks,
            output_dir=tmp_path,
            seed=3,
        )
        assert [kind for kind, _ in made] == [
            "SFTConfig",
            "SFTTrainer",
            "GRPOConfig",
            "GRPOTrainer",
        ]
        sft_args, sft_trainer, grpo_args, grpo_trainer = (kwargs for _, kwargs in made)
        # The callbacks each trainer gets are the ones given, as they are.
        assert sft_trainer["callbacks"] == callbacks and grpo_trainer["callbacks"] == callbacks
        assert sft_trainer["callbacks"] is not callbacks
        assert sft_trainer["args"].kwargs is sft_args and grpo_trainer["reward_funcs"] == [reward]
        built[backend.name] = (sft_args, grpo_args)

    # One T4, two backends: the same settings, fp16 included.
    assert built["trl"] == built["unsloth"]
    sft_args, grpo_args = built["trl"]
    for args in (sft_args, grpo_args):
        assert args["report_to"] == "none" and args["save_strategy"] == "no"
        assert (args["fp16"], args["bf16"], args["use_cpu"]) == (True, False, False)
        assert (args["seed"], args["output_dir"]) == (3, str(tmp_path))
    assert "max_prompt_length" not in grpo_args


@pytest.mark.phase(4)
def test_trl_accepts_every_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # TRL's real configs, so a setting TRL renamed or dropped fails here, not on a GPU.
    made: list[tuple[str, dict[str, Any]]] = []
    for name in ("SFTTrainer", "GRPOTrainer"):
        monkeypatch.setattr(trl, name, type(name, (Recorder,), {"made": made}))
    _devices(monkeypatch)
    backend = TrlBackend("cpu")
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": "how many"}]}])
    sft_cfg = SftConfig(
        max_steps=3, warmup_ratio=0.1, gradient_checkpointing=False, pad_to_multiple_of=64
    )
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
    # The config's setting, not TRL's default of on, under either of transformers' names.
    assert (sft.gradient_checkpointing, sft.activation_checkpointing) == (False, False)
    assert sft.pad_to_multiple_of == 64
    plain = SftConfig()
    backend.sft_trainer("model", WordTokenizer(), dataset, plain, [], output_dir=tmp_path, seed=5)
    defaults = made[-1][1]["args"]
    assert (defaults.activation_checkpointing, defaults.pad_to_multiple_of) == (True, None)

    assert (grpo.num_generations, grpo.max_completion_length, grpo.generation_batch_size) == (
        6,
        128,
        24,
    )
    assert grpo.chat_template_kwargs == {"enable_thinking": False}
    assert grpo.beta == GRPOConfig.__dataclass_fields__["beta"].default  # library_default
    assert grpo_beta.beta == 0.04
