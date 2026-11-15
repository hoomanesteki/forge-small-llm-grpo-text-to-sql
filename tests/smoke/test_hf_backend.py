"""The transformers backend on the tiny random-weight model pinned in configs/models.yaml.

The first run downloads it, about 20 MB, into the Hugging Face cache. Its replies are
noise, so these check the wiring: Spider prompts in, one complete generation per prompt
out, and an adapter loaded through PEFT.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from forge.core.types import Generation, GenParams, Prompt
from forge.data.prompts import build_prompt, chat_token_counter
from forge.data.splits import load_train
from forge.infer.hf_backend import HfBackend
from forge.sql.extract import extract_sql


def _prompts(count: int) -> list[Prompt]:
    # Every eighth train question of the fixture spans its four schemas, so the prompts
    # differ in length and each batch pads its shorter ones.
    return [build_prompt(example, "p1") for example in load_train()[::8][:count]]


def _texts(generations: list[Generation]) -> list[str]:
    return [generation.text for generation in generations]


@pytest.mark.phase(2)
def test_generates_with_tiny_model(spider_mini: Path, tiny_model_ref: str) -> None:
    prompts = _prompts(5)
    backend = HfBackend(tiny_model_ref, device="cpu", batch_size=2)
    params = GenParams(max_new_tokens=16)

    generations = backend.generate(prompts, params)

    assert [g.example_id for g in generations] == [p.example_id for p in prompts]
    count = chat_token_counter(backend.tokenizer)
    for prompt, generation in zip(prompts, generations, strict=True):
        assert generation.model_ref == tiny_model_ref
        assert generation.adapter_ref is None
        assert generation.sample_idx == 0
        assert generation.text
        assert generation.sql == extract_sql(generation.text)
        # As many prompt tokens as the overflow policy counts when it fits the prompt.
        assert generation.tokens_in == count(prompt)
        assert 1 <= generation.tokens_out <= params.max_new_tokens
        assert generation.latency_ms > 0
    # The same batches give the same greedy replies.
    assert _texts(backend.generate(prompts, params)) == _texts(generations)


@pytest.mark.phase(2)
def test_loads_adapter(spider_mini: Path, tiny_model_ref: str, tmp_path: Path) -> None:
    repo, _, revision = tiny_model_ref.partition("@")
    base = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.float32)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        # PEFT starts B at zero, which makes a new adapter a no-op; this one is random, so
        # it changes what the model writes.
        lora = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                init_lora_weights=False,
            ),
        )
    adapter = tmp_path / "adapter"
    lora.save_pretrained(adapter)
    prompts = _prompts(2)
    params = GenParams(max_new_tokens=8)

    backend = HfBackend(tiny_model_ref, adapter_ref=str(adapter), device="cpu")
    adapted = backend.generate(prompts, params)
    plain = HfBackend(tiny_model_ref, device="cpu").generate(prompts, params)

    assert isinstance(backend.model, PeftModel)
    saved = load_file(adapter / "adapter_model.safetensors")
    loaded = get_peft_model_state_dict(backend.model)
    assert saved.keys() == loaded.keys()
    assert all(torch.equal(saved[name], loaded[name]) for name in saved)
    assert [g.adapter_ref for g in adapted] == [str(adapter)] * len(prompts)
    assert _texts(adapted) != _texts(plain)
