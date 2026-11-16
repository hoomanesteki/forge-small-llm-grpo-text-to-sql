"""The inference interface: chat rendering, the checks every backend shares, sample seeds,
and the fake backend."""

from __future__ import annotations

import re
from typing import Any

import pytest

from forge.core.types import GenParams, Prompt
from forge.infer.base import (
    MIN_TEMPERATURE,
    FakeBackend,
    InferenceBackend,
    chat_template_kwargs,
    check_params,
    render_chat,
    sample_seed,
)
from tests.fakes import WordTokenizer
from tests.tiny_hf import CHATML, QWEN3, byte_tokenizer


def _prompt(example_id: str, question: str) -> Prompt:
    return Prompt(
        example_id=example_id,
        messages=(
            {"role": "system", "content": "You write SQLite queries."},
            {"role": "user", "content": question},
        ),
        prompt_version="p1",
    )


def _unreachable(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
    raise AssertionError("the fake called its fn for parameters it should refuse")


@pytest.mark.phase(2)
def test_chat_template_kwargs() -> None:
    prompt = _prompt("q-0", "How many singers are there?")
    qwen3 = WordTokenizer(chat_template=QWEN3)
    qwen25 = WordTokenizer(chat_template=CHATML)

    assert chat_template_kwargs(qwen3) == {"enable_thinking": False}
    assert chat_template_kwargs(qwen25) == {}

    text = render_chat(qwen3, prompt.messages)
    render_chat(qwen25, prompt.messages)

    assert text == (
        "<|system|> You write SQLite queries. <|user|> How many singers are there? <|assistant|>"
    )
    [(conversation, kwargs)] = qwen3.calls
    assert conversation == list(prompt.messages)
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    [(_, kwargs)] = qwen25.calls
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}


@pytest.mark.phase(2)
def test_render_chat_turns_thinking_off_where_the_template_can() -> None:
    messages = _prompt("q-0", "How many singers are there?").messages
    opened = (
        "<|im_start|>system\nYou write SQLite queries.<|im_end|>\n"
        "<|im_start|>user\nHow many singers are there?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Real transformers rendering: Qwen3's template pre-fills an empty think block.
    assert render_chat(byte_tokenizer(QWEN3), messages) == opened + "<think>\n\n</think>\n\n"
    assert render_chat(byte_tokenizer(CHATML), messages) == opened


@pytest.mark.phase(2)
def test_render_chat_needs_the_text_back() -> None:
    class Encodes(WordTokenizer):
        """A tokenizer that encodes even when told not to."""

        def apply_chat_template(
            self, conversation: list[dict[str, str]], **kwargs: Any
        ) -> str | dict[str, list[int]]:
            return super().apply_chat_template(conversation, **{**kwargs, "tokenize": True})

    with pytest.raises(TypeError, match="returned dict, not the prompt text"):
        render_chat(Encodes(), _prompt("q-0", "How many singers are there?").messages)


@pytest.mark.phase(2)
def test_fake_backend_fills_every_field() -> None:
    calls: list[tuple[str, GenParams, int]] = []

    def reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
        calls.append((prompt.example_id, params, sample_idx))
        if prompt.example_id == "q-1":
            return "I can't answer that."
        return f"Here it is:\n```sql\nSELECT {sample_idx} FROM singer;\n```"

    prompts = [_prompt("q-0", "How many singers are there?"), _prompt("q-1", "Why?")]
    params = GenParams(temperature=0.7, n=2)
    backend = FakeBackend(reply, model_ref="org/model@abc1234")

    generations = backend.generate(prompts, params)

    assert isinstance(backend, InferenceBackend)
    assert backend.name == "fake"
    assert calls == [("q-0", params, 0), ("q-0", params, 1), ("q-1", params, 0), ("q-1", params, 1)]
    assert [(g.example_id, g.sample_idx, g.sql) for g in generations] == [
        ("q-0", 0, "SELECT 0 FROM singer"),
        ("q-0", 1, "SELECT 1 FROM singer"),
        ("q-1", 0, None),
        ("q-1", 1, None),
    ]
    # Words of the system and user messages, then of the reply.
    assert [g.tokens_in for g in generations] == [9, 9, 5, 5]
    assert [g.tokens_out for g in generations] == [9, 9, 4, 4]
    for generation in generations:
        assert generation.model_ref == "org/model@abc1234"
        assert generation.adapter_ref is None
        assert generation.latency_ms >= 0


@pytest.mark.phase(2)
def test_fake_backend_wants_text_from_its_fn() -> None:
    def no_text(prompt: Prompt, params: GenParams, sample_idx: int) -> Any:
        return None

    with pytest.raises(TypeError, match="the fake's fn returned NoneType, not str"):
        FakeBackend(no_text).generate([_prompt("q-0", "How many singers?")], GenParams())


@pytest.mark.phase(2)
@pytest.mark.parametrize(
    ("params", "message"),
    [
        (GenParams(temperature=-0.1), "temperature must be 0 or more"),
        (GenParams(temperature=float("nan")), "temperature must be 0 or more"),
        (GenParams(top_p=0.0), "top_p must be above 0 and at most 1"),
        (GenParams(top_p=1.5), "top_p must be above 0 and at most 1"),
        (GenParams(max_new_tokens=0), "max_new_tokens must be at least 1"),
        (GenParams(n=0), "n must be at least 1"),
        (GenParams(n=3), "n=3 needs a temperature above 0"),
    ],
)
def test_check_params_refuses_what_no_backend_can_honor(params: GenParams, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_params(params)
    with pytest.raises(ValueError, match=message):
        FakeBackend(_unreachable).generate([_prompt("q-0", "How many singers?")], params)


@pytest.mark.phase(3)
def test_check_params_refuses_a_temperature_vllm_would_raise() -> None:
    # vLLM decodes 0 < t < 0.01 at 0.01, while the manifest would record t.
    for temperature in (1e-9, 0.005, 0.0099):
        message = f"temperature must be 0 or at least 0.01, got {temperature}: vLLM would"
        with pytest.raises(ValueError, match=re.escape(message)):
            check_params(GenParams(temperature=temperature))
        with pytest.raises(ValueError, match=re.escape(message)):
            FakeBackend(_unreachable).generate(
                [_prompt("q-0", "How many singers?")], GenParams(temperature=temperature)
            )
    assert MIN_TEMPERATURE == 0.01
    for temperature in (0.0, MIN_TEMPERATURE, 0.7):
        check_params(GenParams(temperature=temperature))


@pytest.mark.phase(2)
def test_sample_seed_depends_only_on_the_sample() -> None:
    seed = sample_seed(0, "spider-train-00042", 1)

    # Fixed for good: a resumed run has to draw its samples exactly as before.
    assert seed == 16001323741844453125
    others = {
        sample_seed(1, "spider-train-00042", 1),
        sample_seed(0, "spider-train-00043", 1),
        sample_seed(0, "spider-train-00042", 0),
    }
    assert len(others | {seed}) == 4
    assert all(0 <= value < 2**64 for value in others)
