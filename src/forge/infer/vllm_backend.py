"""Generation with vLLM: the backend for evaluation on a GPU, a Kaggle T4 in fp16.

``VllmBackend`` loads the model once, in fp16 (``dtype="half"``: a T4 has no bf16). With
``adapter_ref`` it enables LoRA and names the adapter in every request through one
``LoRARequest``. vLLM schedules and batches requests itself, so ``generate`` hands it every
prompt at once.

Every prompt is rendered by ``render_chat``, the chat in the model's template with the
assistant turn opened and Qwen3's thinking off, and sent to ``LLM.generate`` as the token
ids the hf backend feeds its model: encoded with no special tokens added, since the
template holds every one the model expects. So both backends give a model the same prompt.
``LLM.chat`` would render with vLLM's own code, and a text prompt would be encoded by
vLLM's rules, where a tokenizer that adds a BOS puts a second one before the template's.

Decoding follows ``GenParams`` and nothing else (``sampling_kwargs``), and ``check_params``
refuses a temperature between 0 and ``MIN_TEMPERATURE``, which vLLM would raise. vLLM
applies a checkpoint's ``generation_config.json`` sampling defaults only to a call that
passes no ``SamplingParams``; from that file it takes the end token ids alone. Each prompt
is seeded by its example id (``sample_seed``), and vLLM seeds the prompt's n samples with
that seed, the seed plus 1 and so on, so a sample draws the same in any batch and two
questions don't share their draws.

``tokens_in`` and ``tokens_out`` are vLLM's counts of the prompt and of the reply, and
``latency_ms`` is the wall time of the whole call, which every generation shares.

vLLM is a GPU package kept out of ``uv.lock`` (ADR-0002) and pinned in
``kaggle/requirements-gpu.txt``. It is imported when a backend is made, so this module
imports anywhere, CI included.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from forge.core.types import Generation, GenParams, Prompt
from forge.infer.base import check_params, render_chat, sample_seed, split_ref
from forge.sql.extract import extract_sql

REQUIREMENTS = "kaggle/requirements-gpu.txt"
# The plan loads 4-bit bitsandbytes weights, and only for a reference model past 4B.
QUANTIZATIONS = ("bitsandbytes",)
ADAPTER_NAME = "adapter"
# The share of the GPU's memory the engine takes unless an eval config sets another.
GPU_MEMORY_UTILIZATION = 0.85


class VllmBackend:
    """An ``InferenceBackend`` on vLLM, for any causal LM with a chat template, on CUDA."""

    name = "vllm"

    def __init__(
        self,
        model_ref: str,
        *,
        adapter_ref: str | None = None,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
        max_model_len: int = 2048,
        max_lora_rank: int = 64,
        quantization: str | None = None,
    ) -> None:
        if quantization is not None and quantization not in QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {', '.join(QUANTIZATIONS)}, got {quantization!r}"
            )
        llm_class, self._sampling_params, lora_class = _import_vllm()
        self.model_ref = model_ref
        self.adapter_ref = adapter_ref
        repo, revision = split_ref(model_ref)
        lora: dict[str, Any] = {}
        self.lora_request: Any = None  # what vLLM takes for no adapter
        if adapter_ref is not None:
            # Fetched before the model loads, so a missing adapter fails in seconds.
            self.lora_request = lora_class(ADAPTER_NAME, 1, adapter_path(adapter_ref))
            lora = {"enable_lora": True, "max_lora_rank": max_lora_rank}
        # vLLM reads the tokenizer at the model's revision unless told otherwise.
        self.llm: Any = llm_class(
            model=repo,
            revision=revision,
            dtype="half",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            quantization=quantization,
            **lora,
        )
        self.tokenizer: Any = self.llm.get_tokenizer()

    def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
        check_params(params)
        if not prompts:
            # Nothing to send, so no engine call for an empty batch.
            return []
        sampling = [
            self._sampling_params(**sampling_kwargs(params, prompt.example_id))
            for prompt in prompts
        ]
        start = time.perf_counter()
        outputs = self._request(prompts, sampling)
        latency_ms = (time.perf_counter() - start) * 1000
        if len(outputs) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts")
        generations: list[Generation] = []
        for prompt, output in zip(prompts, outputs, strict=True):
            replies = sorted(output.outputs, key=lambda reply: reply.index)
            if [reply.index for reply in replies] != list(range(params.n)):
                raise RuntimeError(
                    f"vLLM returned samples {[reply.index for reply in replies]} for "
                    f"{prompt.example_id}, not 0 to {params.n - 1}"
                )
            generations.extend(
                Generation(
                    example_id=prompt.example_id,
                    model_ref=self.model_ref,
                    adapter_ref=self.adapter_ref,
                    sample_idx=reply.index,
                    text=reply.text,
                    sql=extract_sql(reply.text),
                    tokens_in=len(output.prompt_token_ids),
                    tokens_out=len(reply.token_ids),
                    latency_ms=latency_ms,
                )
                for reply in replies
            )
        return generations

    def _request(self, prompts: Sequence[Prompt], sampling: list[Any]) -> list[Any]:
        token_prompts = [
            {
                "prompt_token_ids": self.tokenizer.encode(
                    render_chat(self.tokenizer, prompt.messages), add_special_tokens=False
                )
            }
            for prompt in prompts
        ]
        return self.llm.generate(
            token_prompts, sampling_params=sampling, use_tqdm=False, lora_request=self.lora_request
        )


def sampling_kwargs(params: GenParams, example_id: str) -> dict[str, Any]:
    """The ``SamplingParams`` arguments that decode the prompt of ``example_id`` as
    ``params`` asks."""
    return {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_new_tokens,
        "n": params.n,
        # 63 bits, so the seed plus a sample's index still fits vLLM's 64-bit seed.
        "seed": sample_seed(params.seed, example_id, 0) >> 1,
    }


def adapter_path(ref: str) -> str:
    """The folder of the adapter ``ref`` names, as ``LoRARequest`` needs one: ``ref`` itself,
    or the Hub repo's snapshot at the revision after its ``@``."""
    if Path(ref).is_dir():
        return ref
    repo, revision = split_ref(ref)
    return snapshot_download(repo, revision=revision)


def _import_vllm() -> tuple[Any, Any, Any]:
    """vLLM's ``LLM``, ``SamplingParams`` and ``LoRARequest``."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise ImportError(
            "the vllm backend needs vllm, a GPU package that neither CI nor the cpu extra "
            f"installs: on a machine with a CUDA GPU, pip install -r {REQUIREMENTS}, as a "
            "Kaggle job does"
        ) from exc
    return LLM, SamplingParams, LoRARequest
