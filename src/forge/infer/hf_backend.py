"""Generation with Hugging Face transformers: the backend for CPU runs, tests and smoke runs.

``HfBackend`` loads a model, and a LoRA adapter through PEFT when ``adapter_ref`` is
given. It answers prompts in batches of ``batch_size`` rows, one row per prompt and
sample, each prompt rendered by ``render_chat`` and padded on the left, so every row's
reply starts in the same column.

Decoding follows ``GenParams`` and nothing else. A checkpoint ships generation defaults
(the tiny test model samples at temperature 0.7 with top_k 20 and a 1.05 repetition
penalty) that ``generate`` would apply to every setting a call leaves unset, so the
loaded model keeps only its special token ids. At temperature 0 a reply is greedy: the
most likely token at every step. Above 0, each row samples the tempered, top-p filtered
distribution with its own ``torch.Generator``, seeded by ``sample_seed``, so a sample is
the same whatever else shares its batch.

``tokens_in`` counts the rendered prompt and ``tokens_out`` the reply, up to and including
the token that ended it. ``latency_ms`` is the wall time of the ``generate`` call, which
every row of a batch shares.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from huggingface_hub.utils import HFValidationError, validate_repo_id
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)

from forge.core.types import Generation, GenParams, Prompt
from forge.infer.base import check_params, render_chat, sample_seed
from forge.sql.extract import extract_sql

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# One row of a batch: the prompt, its rendered text, and the sample index.
_Row = tuple[Prompt, str, int]


class CausalLM(Protocol):
    """What the backend calls on a model, with or without a PEFT adapter around it.

    transformers types ``generate``'s ``self`` with a protocol that mypy can't match to
    any model class, so the backend holds its model by this one.
    """

    def generate(self, **kwargs: Any) -> Any: ...


class HfBackend:
    """An ``InferenceBackend`` on transformers, for any causal LM with a chat template."""

    name = "hf"

    def __init__(
        self,
        model_ref: str,
        *,
        adapter_ref: str | None = None,
        device: str = "auto",
        dtype: str | torch.dtype | None = None,
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.model_ref = model_ref
        self.adapter_ref = adapter_ref
        self.batch_size = batch_size
        self.device = resolve_device(device)
        repo, revision = split_ref(model_ref)
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # from_pretrained returns the model in eval mode, as PEFT does an adapter it loads
        # for inference.
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, dtype=resolve_dtype(dtype), device_map=self.device
        )
        shipped = model.generation_config
        # A chat model ends its turn with the tokenizer's end token, which the checkpoint's
        # own list may leave out.
        self.eos_token_ids = list(
            dict.fromkeys([*_ids(shipped.eos_token_id), *_ids(self.tokenizer.eos_token_id)])
        )
        model.generation_config = GenerationConfig(
            bos_token_id=shipped.bos_token_id,
            eos_token_id=self.eos_token_ids or None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.model: CausalLM = model
        if adapter_ref is not None:
            adapter, adapter_revision = split_ref(adapter_ref)
            self.model = PeftModel.from_pretrained(
                model, adapter, revision=adapter_revision, torch_device=str(self.device)
            )

    def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
        check_params(params)
        rows: list[_Row] = []
        for prompt in prompts:
            text = render_chat(self.tokenizer, prompt.messages)
            rows.extend((prompt, text, sample_idx) for sample_idx in range(params.n))
        generations: list[Generation] = []
        for start in range(0, len(rows), self.batch_size):
            generations.extend(self._generate_batch(rows[start : start + self.batch_size], params))
        return generations

    def _generate_batch(self, rows: Sequence[_Row], params: GenParams) -> list[Generation]:
        # The template already holds every special token the model expects.
        batch = self.tokenizer(
            [text for _, text, _ in rows],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                generation_config=GenerationConfig(
                    do_sample=False, max_new_tokens=params.max_new_tokens
                ),
                logits_processor=_sampling(rows, params),
            )
        latency_ms = (time.perf_counter() - start) * 1000
        width = batch["input_ids"].shape[1]
        tokens_in = batch["attention_mask"].sum(dim=1).tolist()
        replies = [reply_ids(ids[width:], self.eos_token_ids) for ids in output.tolist()]
        texts = self.tokenizer.batch_decode(replies, skip_special_tokens=True)
        return [
            Generation(
                example_id=prompt.example_id,
                model_ref=self.model_ref,
                adapter_ref=self.adapter_ref,
                sample_idx=sample_idx,
                text=text,
                sql=extract_sql(text),
                tokens_in=count,
                tokens_out=len(reply),
                latency_ms=latency_ms,
            )
            for (prompt, _, sample_idx), text, reply, count in zip(
                rows, texts, replies, tokens_in, strict=True
            )
        ]


class GumbelSampler(LogitsProcessor):
    """Turns greedy search into sampling, each row with its own generator.

    The argmax of scores plus Gumbel noise is a draw from the scores' softmax (the
    Gumbel-max trick), so the token greedy search picks is a sample, and a row's draws
    come only from its own generator.
    """

    def __init__(self, generators: Sequence[torch.Generator]) -> None:
        self.generators = list(generators)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[0] != len(self.generators):
            raise ValueError(
                f"{scores.shape[0]} rows of scores but {len(self.generators)} generators"
            )
        # Drawn on the CPU, so a seed gives the same noise on every device.
        uniform = torch.stack(
            [torch.rand(scores.shape[-1], generator=generator) for generator in self.generators]
        )
        return scores - torch.log(-torch.log(uniform)).to(scores.device)


def resolve_device(device: str) -> torch.device:
    """``auto`` is CUDA when there is a GPU, then Apple's MPS, then the CPU."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    """``None`` is float32, which every device runs and which rounds the least.

    transformers would otherwise load a checkpoint in the dtype it was saved in.
    """
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(DTYPES)}, got {dtype!r}")
    return DTYPES[dtype]


def split_ref(ref: str) -> tuple[str, str | None]:
    """A model or adapter ref as ``from_pretrained`` takes it: a local directory with no
    revision, or a Hub repo id with the revision after an optional ``@``."""
    if Path(ref).is_dir():
        return ref, None
    repo, at, revision = ref.partition("@")
    try:
        validate_repo_id(repo)
    except HFValidationError:
        # Otherwise a mistyped local path fails as a malformed repo id.
        raise FileNotFoundError(
            f"{ref!r} is neither a directory nor a Hub repo id with an optional @revision"
        ) from None
    if at and not revision:
        raise ValueError(f"{ref!r} has nothing after its '@'; give a revision or drop the '@'")
    return repo, revision or None


def reply_ids(generated: Sequence[int], eos_token_ids: Collection[int]) -> list[int]:
    """A row's reply: its tokens up to and including the first end token, after which
    ``generate`` pads the row until the whole batch is done."""
    for position, token in enumerate(generated):
        if token in eos_token_ids:
            return list(generated[: position + 1])
    return list(generated)


def _sampling(rows: Sequence[_Row], params: GenParams) -> LogitsProcessorList:
    """What greedy search runs on the scores to decode as ``params`` asks."""
    processors = LogitsProcessorList()
    if params.temperature == 0:
        return processors
    processors.append(TemperatureLogitsWarper(params.temperature))
    if params.top_p < 1:
        processors.append(TopPLogitsWarper(params.top_p))
    processors.append(
        GumbelSampler(
            [
                torch.Generator().manual_seed(sample_seed(params.seed, prompt.example_id, idx))
                for prompt, _, idx in rows
            ]
        )
    )
    return processors


def _ids(value: int | list[int] | None) -> list[int]:
    if value is None:
        return []
    return [value] if isinstance(value, int) else list(value)
