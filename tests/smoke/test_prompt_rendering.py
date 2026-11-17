"""The prompt each backend gives a model: hf and vllm send the same token ids, for the tiny
model's own Qwen2.5 template and for a Qwen3-style one that reads the thinking flag.

The tokenizer is the real one of the tiny model pinned in configs/models.yaml. vLLM needs a
GPU, so it is faked (tests.fakes.FakeVllm) around that tokenizer: the fake only records
the ids the backend sends, and has no rendering of its own to agree with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer

from forge.core.types import GenParams
from forge.data.prompts import build_prompt
from forge.data.splits import load_train
from forge.infer.base import split_ref
from forge.infer.hf_backend import CausalLM, HfBackend
from forge.infer.vllm_backend import VllmBackend
from tests.fakes import FakeVllm
from tests.tiny_hf import QWEN3

THINK_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Spy:
    """The hf backend's model, keeping the ids of each prompt it is asked to continue."""

    def __init__(self, model: CausalLM) -> None:
        self.model = model
        self.prompts: list[list[int]] = []

    def generate(self, **kwargs: Any) -> Any:
        # Rows are padded on the left; the attention mask marks the prompt's own tokens.
        rows = zip(kwargs["input_ids"].tolist(), kwargs["attention_mask"].tolist(), strict=True)
        for ids, mask in rows:
            self.prompts.append([token for token, kept in zip(ids, mask, strict=True) if kept])
        return self.model.generate(**kwargs)


@pytest.mark.phase(3)
@pytest.mark.parametrize("template", ["tiny", "qwen3"])
def test_hf_and_vllm_give_the_model_the_same_prompt(
    template: str, spider_mini: Path, tiny_model_ref: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Three schemas of different sizes, so a batch of two pads its shorter prompt.
    prompts = [build_prompt(example, "p1") for example in load_train()[::8][:3]]
    params = GenParams(max_new_tokens=1)
    hf = HfBackend(tiny_model_ref, device="cpu", batch_size=2)
    repo, revision = split_ref(tiny_model_ref)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    if template == "qwen3":
        hf.tokenizer.chat_template = tokenizer.chat_template = QWEN3
    spy = Spy(hf.model)
    hf.model = spy
    vllm = FakeVllm(lambda text, sample_idx: "SELECT 1", tokenizer=tokenizer).install(monkeypatch)

    on_hf = hf.generate(prompts, params)
    on_vllm = VllmBackend(tiny_model_ref).generate(prompts, params)

    assert len(spy.prompts) == len(prompts)
    assert vllm.prompt_ids == spy.prompts
    assert [g.tokens_in for g in on_vllm] == [g.tokens_in for g in on_hf]
    texts = [tokenizer.decode(ids) for ids in spy.prompts]
    if template == "qwen3":
        # Thinking off: the template opens the reply with an empty think block.
        assert all(text.endswith(THINK_OFF) for text in texts)
    else:
        assert all(text.endswith("<|im_start|>assistant\n") for text in texts)
        assert not any("<think>" in text for text in texts)
    # The template's own special tokens start the prompt, with none added before them.
    assert all(text.startswith("<|im_start|>system\n") for text in texts)
