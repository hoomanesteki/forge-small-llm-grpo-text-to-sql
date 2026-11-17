"""Generation with Hugging Face transformers: the backend for CPU runs, tests and smoke runs.

``HfBackend`` loads a model, and a LoRA adapter through PEFT when ``adapter_ref`` is
given. ``HfBackend.around`` generates with a model already loaded instead, such as one a
trainer is training, and leaves it as it found it. It answers prompts in batches of
``batch_size`` rows, one row per prompt and sample, each prompt rendered by
``render_chat`` and padded on the left, so every row's reply starts in the same column.

Decoding follows ``GenParams`` and nothing else. A checkpoint ships generation defaults
(the tiny test model samples at temperature 0.7 with top_k 20 and a 1.05 repetition
penalty) that ``generate`` would apply to every setting a call leaves unset, so a model
the backend loads keeps only its special token ids, and one ``around`` is given decodes
with only those, in eval mode, for the length of each ``generate`` call. At temperature 0
a reply is greedy: the most likely token at every step. Above 0, each row samples the
tempered, top-p filtered distribution with its own ``torch.Generator``, seeded by
``sample_seed``, so a sample is the same whatever else shares its batch.

``tokens_in`` counts the rendered prompt and ``tokens_out`` the reply, up to and including
the token that ended it. ``latency_ms`` is the wall time of the ``generate`` call, which
every row of a batch shares.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, Protocol

import torch
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
from forge.infer.base import check_params, render_chat, sample_seed, split_ref
from forge.sql.extract import extract_sql

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# One row of a batch: the prompt, its rendered text, and the sample index.
_Row = tuple[Prompt, str, int]


class CausalLM(Protocol):
    """What the backend calls on a model, with or without a PEFT adapter around it.

    transformers types ``generate``'s ``self`` with a protocol that mypy can't match to
    any model class, so the backend holds its model by this one.
    """

    training: bool

    def generate(self, **kwargs: Any) -> Any: ...

    def train(self, mode: bool = True) -> Any: ...


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
        # Every argument is checked before the tokenizer and model load, which can take
        # minutes, so a mistyped adapter or dtype fails at once.
        self.device = resolve_device(device)
        torch_dtype = resolve_dtype(dtype)
        repo, revision = split_ref(model_ref)
        adapter = None if adapter_ref is None else split_ref(adapter_ref)
        self.model_ref = model_ref
        self.adapter_ref = adapter_ref
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # from_pretrained returns the model in eval mode, as PEFT does an adapter it loads
        # for inference.
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, dtype=torch_dtype, device_map=self.device
        )
        self.eos_token_ids, self.generation_config = _decoding_config(
            model.generation_config, self.tokenizer
        )
        model.generation_config = self.generation_config
        self.model: CausalLM = model
        self.borrowed = False  # the model is this backend's own, as it decodes
        if adapter is not None:
            adapter_repo, adapter_revision = adapter
            self.model = PeftModel.from_pretrained(
                model, adapter_repo, revision=adapter_revision, torch_device=str(self.device)
            )

    @classmethod
    def around(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        model_ref: str,
        adapter_ref: str | None = None,
        batch_size: int = 8,
    ) -> HfBackend:
        """A backend that generates with ``model`` and ``tokenizer``, already loaded, on the
        device the model is on, and changes neither: each ``generate`` call puts back the
        model's mode and generation config once it is done. ``model_ref`` and
        ``adapter_ref`` are what its generations name."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if tokenizer.pad_token_id is None:
            raise ValueError("the tokenizer has no pad token, which a batch of prompts needs")
        backend = cls.__new__(cls)
        backend.device = next(model.parameters()).device
        backend.model_ref = model_ref
        backend.adapter_ref = adapter_ref
        backend.batch_size = batch_size
        backend.tokenizer = tokenizer
        backend.model = model
        backend.borrowed = True
        backend.eos_token_ids, backend.generation_config = _decoding_config(
            _generating_model(model).generation_config, tokenizer
        )
        return backend

    def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
        check_params(params)
        rows: list[_Row] = []
        for prompt in prompts:
            text = render_chat(self.tokenizer, prompt.messages)
            rows.extend((prompt, text, sample_idx) for sample_idx in range(params.n))
        generations: list[Generation] = []
        with self._decoding() if self.borrowed else nullcontext():
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start : start + self.batch_size]
                generations.extend(self._generate_batch(batch, params))
        return generations

    @contextmanager
    def _decoding(self) -> Iterator[None]:
        """The borrowed model in eval mode with ``generation_config`` as its defaults, then
        as it was: it may be training, with the checkpoint's sampling defaults."""
        target = _generating_model(self.model)
        shipped, training = target.generation_config, self.model.training
        target.generation_config = self.generation_config
        self.model.train(False)
        try:
            yield
        finally:
            target.generation_config = shipped
            self.model.train(training)

    def _generate_batch(self, rows: Sequence[_Row], params: GenParams) -> list[Generation]:
        # The template already holds every special token the model expects.
        batch = self.tokenizer(
            [text for _, text, _ in rows],
            return_tensors="pt",
            padding=True,
            padding_side="left",
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


def _decoding_config(
    shipped: GenerationConfig, tokenizer: Any
) -> tuple[list[int], GenerationConfig]:
    """The ids that end a reply, and a generation config of the model's special token ids
    alone, which ``generate`` then fills unset settings from rather than the checkpoint's."""
    # A chat model ends its turn with the tokenizer's end token, which the checkpoint's own
    # list may leave out.
    eos_token_ids = list(
        dict.fromkeys([*_ids(shipped.eos_token_id), *_ids(tokenizer.eos_token_id)])
    )
    config = GenerationConfig(
        bos_token_id=shipped.bos_token_id,
        eos_token_id=eos_token_ids or None,
        pad_token_id=tokenizer.pad_token_id,
    )
    return eos_token_ids, config


def _generating_model(model: Any) -> Any:
    """The model whose generation config ``generate`` reads: under a PEFT adapter, the
    base model PEFT wraps."""
    base = getattr(model, "get_base_model", None)
    return model if base is None else base()


def resolve_device(device: str) -> torch.device:
    """``auto`` is CUDA when there is a GPU, then Apple's MPS, then the CPU.

    ``mps`` asked for by name must be there: an eval run's manifest names the device it
    asked for (ADR-0016), so the run must not quietly fall back to another one.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "device mps needs Apple silicon and a torch build with MPS, and torch can't "
            "use it here; run on the cpu device instead"
        )
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
