"""Training backends: one interface for SFT and GRPO on the CPU, Apple's MPS and CUDA
(ADR-0006), so every training path runs in CI before it costs GPU time.

``TrainBackend`` is plan section 3.5's protocol. ``load`` gives the model with LoRA on
the config's target modules, the rest of it frozen, and its tokenizer; ``sft`` and
``grpo`` train it and return the folder they saved the adapter in. Where the trainer
writes and the run's seed are keywords the plan's signatures have no place for, and
``load`` takes the seed too: LoRA draws its first weights there, before any trainer
seeds.

- ``TrlBackend``: transformers, PEFT LoRA and TRL's trainers, generating with
  transformers rather than vLLM, at the precision ``precision_for`` picks for its
  device: float32 on the CPU; on MPS bf16 where torch says macOS supports it, else
  float32; on CUDA bf16 where the GPU has it, else float16 with ``fp16`` (a T4: TRL's
  default bf16 crashes there, TRL issue 7889). 4-bit loading is bitsandbytes NF4, on
  CUDA only.
- ``UnslothBackend``: Unsloth's loading (4-bit if asked) and LoRA, and for GRPO, when
  given ``gpu_memory_utilization``, rollouts from its colocated vLLM. CUDA only.
  Unsloth patches TRL and transformers as it is imported, so the backend imports it
  when it is made: make it before anything imports either.

Both build their trainers the same way (``sft_trainer``, ``grpo_trainer``), with the
callbacks they are given and no tracker of transformers' own (``report_to`` is none;
``forge.train.callbacks.training_callbacks`` adds W&B when there is a key). The trainer
saves only when a callback asks (``save_strategy`` is no): every ``checkpoint_every``
steps and at the time guard. ``max_prompt_length`` is Forge's own filter, since TRL has
no such setting any more: rows whose prompt is longer are dropped, and counted, before
TRL sees them.

``select_backend("auto")`` picks Unsloth when torch sees a CUDA GPU and Unsloth is
installed, and otherwise ``TrlBackend`` on the best device there is: the CPU in CI.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from forge.data.prompts import count_chat_tokens
from forge.infer.base import THINKING_FLAG
from forge.ops.hub import KEEP_CHECKPOINTS
from forge.train.config import (
    LIBRARY_DEFAULT,
    BackendName,
    GrpoConfig,
    LoraConfig,
    SftConfig,
    TrainDevice,
)

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import TrainerCallback
    from trl import GRPOTrainer, SFTTrainer

log = logging.getLogger(__name__)

ADAPTER_DIR = "adapter"  # in the output folder: where sft and grpo save the adapter
# Unsloth's own default length, for a backend made without a config's.
DEFAULT_MAX_SEQ_LENGTH = 2048
# Plan section 3.5's reward, as TRL calls it: keywords carry the dataset's other columns.
RewardFunc = Callable[..., list[float]]


class BackendUnavailable(RuntimeError):
    """The backend asked for can't run here: the package or the GPU it needs is missing."""


class TrainBackend(Protocol):
    name: str
    device: str  # where it trains: cpu, mps or cuda
    precision: Precision  # what it trains at there, which a run's manifest records

    def load(
        self, model_ref: str, lora: LoraConfig, *, load_in_4bit: bool, seed: int
    ) -> tuple[Any, Any]: ...  # model, tokenizer

    def sft(
        self,
        model: Any,
        tok: Any,
        dataset: Dataset,
        cfg: SftConfig,
        callbacks: Sequence[TrainerCallback],
        *,
        output_dir: str | Path,
        seed: int,
    ) -> str: ...  # the adapter's folder

    def grpo(
        self,
        model: Any,
        tok: Any,
        dataset: Dataset,
        rewards: Sequence[RewardFunc],
        cfg: GrpoConfig,
        callbacks: Sequence[TrainerCallback],
        *,
        output_dir: str | Path,
        seed: int,
    ) -> str: ...


@dataclass(frozen=True)
class Precision:
    """The dtype a backend loads weights in, and the trainer's mixed-precision flags."""

    dtype: Literal["float32", "float16", "bfloat16"]
    fp16: bool = False
    bf16: bool = False
    use_cpu: bool = False

    def args(self) -> dict[str, bool]:
        return {"fp16": self.fp16, "bf16": self.bf16, "use_cpu": self.use_cpu}


def precision_for(device: str) -> Precision:
    """The precision a backend trains at on ``device``: cpu, mps or cuda."""
    import torch

    if device == "cpu":
        return Precision("float32", use_cpu=True)
    if device == "mps":
        # Metal runs bf16 from macOS 14 on, as transformers' own check says.
        if torch.backends.mps.is_macos_or_newer(14, 0):
            return Precision("bfloat16", bf16=True)
        return Precision("float32")
    if device == "cuda":
        # Without emulation: a T4 (sm_75) does bf16 only in software, and TRL's fused LM
        # head crashes on it under bf16 (TRL issue 7889), so it trains in fp16.
        if torch.cuda.is_bf16_supported(including_emulation=False):
            return Precision("bfloat16", bf16=True)
        return Precision("float16", fp16=True)
    raise ValueError(f"a backend trains on cpu, mps or cuda, not {device!r}")


def peft_lora(lora: LoraConfig) -> PeftLoraConfig:
    """PEFT's LoRA config for ours."""
    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target),
        bias="none",
        task_type="CAUSAL_LM",
    )


def check_targets(model: Any, lora: LoraConfig, model_ref: str) -> None:
    """Raise ``ValueError`` unless every LoRA target matched a module of the model.

    PEFT fails only when none of them match, so a misspelt one would otherwise leave
    those layers untrained without a word.
    """
    found = {name.rsplit(".", 1)[-1] for name in model.base_model.targeted_module_names}
    missing = [target for target in lora.target if target not in found]
    if missing:
        raise ValueError(
            f"LoRA targets {', '.join(missing)} match no module of {model_ref}; "
            f"it matched {', '.join(sorted(found))}"
        )


def drop_long_prompts(dataset: Dataset, tokenizer: Any, max_prompt_length: int) -> Dataset:
    """The rows whose prompt fits in ``max_prompt_length`` tokens, counted as TRL's GRPO
    trainer encodes it (``count_chat_tokens``).

    Older TRL cut long prompts from the left, which would cut a schema in half; dropping
    the row keeps every prompt the model sees whole.
    """
    lengths = [count_chat_tokens(tokenizer, prompt) for prompt in dataset["prompt"]]
    if not lengths:
        raise ValueError("the GRPO dataset has no rows to train on")
    kept = [index for index, length in enumerate(lengths) if length <= max_prompt_length]
    if not kept:
        raise ValueError(
            f"all {len(lengths)} prompts are over max_prompt_length {max_prompt_length} "
            f"tokens; the shortest has {min(lengths)}"
        )
    if len(kept) == len(lengths):
        return dataset
    log.warning(
        "grpo: %d of %d prompts are over max_prompt_length %d tokens and left out",
        len(lengths) - len(kept),
        len(lengths),
        max_prompt_length,
    )
    # In memory: a dataset loaded from disk would otherwise write an index file beside it.
    return dataset.select(kept, keep_in_memory=True)


def sft_settings(cfg: SftConfig) -> dict[str, Any]:
    """TRL ``SFTConfig`` arguments for an SFT config."""
    return {
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": -1 if cfg.max_steps is None else cfg.max_steps,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        # transformers 5 reads a warmup_steps below 1 as a share of the steps.
        "warmup_steps": cfg.warmup_ratio,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_length": cfg.max_length,
        "logging_steps": cfg.logging_steps,
        # transformers 5.23 names it activation_checkpointing and takes this older name as
        # an alias, so the same call works with a kernel's older transformers.
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "pad_to_multiple_of": cfg.pad_to_multiple_of,
        # The loss is on the reply alone, from forge.train.sft_data's prompt-completion rows.
        "completion_only_loss": True,
    }


def grpo_settings(cfg: GrpoConfig) -> dict[str, Any]:
    """TRL ``GRPOConfig`` arguments for a GRPO config; never ``max_prompt_length``."""
    settings: dict[str, Any] = {
        "num_generations": cfg.num_generations,
        "max_completion_length": cfg.max_completion_length,
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_steps": cfg.max_steps,
        "logging_steps": cfg.logging_steps,
        # Qwen3 answers without thinking (plan section 3.7); other templates ignore it.
        "chat_template_kwargs": {THINKING_FLAG: False},
    }
    if cfg.beta != LIBRARY_DEFAULT:
        settings["beta"] = cfg.beta
    return settings


class _TrlTrainers:
    """The trainers both backends build: TRL's, on a model ``load`` gave."""

    name: str
    device: str
    precision: Precision

    def sft_trainer(
        self,
        model: Any,
        tok: Any,
        dataset: Dataset,
        cfg: SftConfig,
        callbacks: Sequence[TrainerCallback],
        *,
        output_dir: str | Path,
        seed: int,
    ) -> SFTTrainer:
        from trl import SFTConfig, SFTTrainer

        args = SFTConfig(**self._common(output_dir, seed), **sft_settings(cfg))
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tok,
            callbacks=list(callbacks),
        )
        self._adjust(trainer)
        return trainer

    def grpo_trainer(
        self,
        model: Any,
        tok: Any,
        dataset: Dataset,
        rewards: Sequence[RewardFunc],
        cfg: GrpoConfig,
        callbacks: Sequence[TrainerCallback],
        *,
        output_dir: str | Path,
        seed: int,
    ) -> GRPOTrainer:
        from trl import GRPOConfig, GRPOTrainer

        args = GRPOConfig(**self._common(output_dir, seed), **grpo_settings(cfg))
        # TRL types a reward's list as list[float | None], which list[float] is not.
        reward_funcs: list[Any] = list(rewards)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=drop_long_prompts(dataset, tok, cfg.max_prompt_length),
            processing_class=tok,
            callbacks=list(callbacks),
        )
        self._adjust(trainer)
        return trainer

    def sft(
        self,
        model: Any,
        tok: Any,
        dataset: Dataset,
        cfg: SftConfig,
        callbacks: Sequence[TrainerCallback],
        *,
        output_dir: str | Path,
        seed: int,
    ) -> str:
        trainer = self.sft_trainer(
            model, tok, dataset, cfg, callbacks, output_dir=output_dir, seed=seed
        )
        return _train_and_save(trainer, output_dir)

    def grpo(
        self,
        model: Any,
        tok: Any,
        dataset: Dataset,
        rewards: Sequence[RewardFunc],
        cfg: GrpoConfig,
        callbacks: Sequence[TrainerCallback],
        *,
        output_dir: str | Path,
        seed: int,
    ) -> str:
        trainer = self.grpo_trainer(
            model, tok, dataset, rewards, cfg, callbacks, output_dir=output_dir, seed=seed
        )
        return _train_and_save(trainer, output_dir)

    def _common(self, output_dir: str | Path, seed: int) -> dict[str, Any]:
        return {
            "output_dir": str(output_dir),
            "seed": seed,
            # Saves come from the callbacks: the checkpoint cadence and the time guard.
            "save_strategy": "no",
            "save_total_limit": KEEP_CHECKPOINTS,
            # No tracker of transformers' own; training_callbacks adds W&B when asked.
            "report_to": "none",
            **self.precision.args(),
        }

    def _adjust(self, trainer: Any) -> None:
        """What a backend changes on a trainer once it is built."""


def _train_and_save(trainer: Any, output_dir: str | Path) -> str:
    trainer.train()
    adapter = Path(output_dir) / ADAPTER_DIR
    # A PEFT model saves its adapter alone, and the trainer the tokenizer beside it.
    trainer.save_model(str(adapter))
    return str(adapter)


class TrlBackend(_TrlTrainers):
    """transformers, PEFT and TRL on ``device``: auto, cpu, mps or cuda."""

    name = "trl"

    def __init__(self, device: TrainDevice = "auto") -> None:
        import torch

        from forge.infer.hf_backend import resolve_device

        if device == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailable("device cuda needs a GPU torch can see, and there is none")
        self.device = resolve_device(device).type
        self.precision = precision_for(self.device)

    def load(
        self, model_ref: str, lora: LoraConfig, *, load_in_4bit: bool, seed: int
    ) -> tuple[Any, Any]:
        if load_in_4bit and self.device != "cuda":
            raise ValueError(
                f"load_in_4bit needs bitsandbytes on CUDA, and this backend trains on {self.device}"
            )
        import torch
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        from forge.infer.hf_backend import split_ref

        repo, revision = split_ref(model_ref)
        dtype = getattr(torch, self.precision.dtype)
        options: dict[str, Any] = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
            # bitsandbytes places the weights on the GPU as it loads them.
            options["device_map"] = {"": 0}
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        base = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=dtype, **options)
        set_seed(seed)
        model = get_peft_model(base, peft_lora(lora))
        check_targets(model, lora, model_ref)
        return model, tokenizer

    def _adjust(self, trainer: Any) -> None:
        model = trainer.model
        if not (self.precision.fp16 and getattr(model, "is_loaded_in_4bit", False)):
            return
        # TRL casts a 4-bit model's adapters to bf16, which fp16's grad scaler can't
        # unscale (TRL issue 7776); fp32 adapters train fine under fp16 autocast.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()


class UnslothBackend(_TrlTrainers):
    """Unsloth's loading and LoRA, and TRL's trainers as Unsloth patches them; CUDA only.

    ``gpu_memory_utilization``, when given, has GRPO generate its rollouts with Unsloth's
    colocated vLLM, which takes that share of the GPU's memory.
    """

    name = "unsloth"

    def __init__(
        self,
        *,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        gpu_memory_utilization: float | None = None,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise BackendUnavailable(
                "the unsloth backend trains on a CUDA GPU, and torch sees none here; use "
                "the trl backend"
            )
        try:
            unsloth = importlib.import_module("unsloth")
        except ImportError as exc:
            raise BackendUnavailable(
                "the unsloth backend needs the unsloth package, which Forge's lock file "
                "leaves out (its pins clash with the TRL that CI tests); install it on the "
                "GPU image, or use the trl backend"
            ) from exc
        self._fast = unsloth.FastLanguageModel
        self.max_seq_length = max_seq_length
        self.gpu_memory_utilization = gpu_memory_utilization
        self.device = "cuda"
        self.precision = precision_for(self.device)

    def load(
        self, model_ref: str, lora: LoraConfig, *, load_in_4bit: bool, seed: int
    ) -> tuple[Any, Any]:
        import torch

        from forge.infer.hf_backend import split_ref

        repo, revision = split_ref(model_ref)
        options: dict[str, Any] = {}
        if self.gpu_memory_utilization is not None:
            options = {
                "fast_inference": True,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_lora_rank": lora.r,
            }
        model, tokenizer = self._fast.from_pretrained(
            model_name=repo,
            revision=revision,
            max_seq_length=self.max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=getattr(torch, self.precision.dtype),
            **options,
        )
        model = self._fast.get_peft_model(
            model,
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=list(lora.target),
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
        check_targets(model, lora, model_ref)
        return model, tokenizer


def select_backend(
    name: BackendName = "auto",
    *,
    device: TrainDevice = "auto",
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    gpu_memory_utilization: float | None = None,
) -> TrainBackend:
    """The backend ``name`` asks for, on ``device``.

    ``auto`` is Unsloth when the device may be CUDA, torch sees a GPU and Unsloth is
    installed, and ``TrlBackend`` otherwise. ``max_seq_length`` and
    ``gpu_memory_utilization`` are Unsloth's, for ``UnslothBackend``; the TRL backend
    takes its lengths from the trainer's config and generates with transformers.
    """
    import torch

    if name == "auto":
        unsloth_here = (
            device in ("auto", "cuda")
            and torch.cuda.is_available()
            and importlib.util.find_spec("unsloth") is not None
        )
        name = "unsloth" if unsloth_here else "trl"
        log.info("backend auto: %s", name)
    if name == "trl":
        return TrlBackend(device)
    if name == "unsloth":
        if device not in ("auto", "cuda"):
            raise ValueError(f"the unsloth backend trains on CUDA only, not on device {device}")
        return UnslothBackend(
            max_seq_length=max_seq_length, gpu_memory_utilization=gpu_memory_utilization
        )
    raise ValueError(f"a backend is auto, trl or unsloth, not {name!r}")
