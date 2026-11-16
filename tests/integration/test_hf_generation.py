"""HfBackend generating through real transformers and PEFT code, on a random model built
offline (``tests/tiny_hf.py``), so this runs in CI with no download."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
)

from forge.core.types import ExecResult, Generation, GenParams, Prompt
from forge.eval.retry import ask_again, retry_prompt
from forge.infer import hf_backend
from forge.infer.base import MIN_TEMPERATURE, render_chat
from forge.infer.hf_backend import HfBackend
from forge.sql.extract import extract_sql
from tests.tiny_hf import byte_tokenizer, save_random_model

# Defaults a checkpoint can ship that would change a greedy reply if they reached generate.
SHIPPED = {
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 3,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 2,
}


def _prompt(example_id: str, question: str) -> Prompt:
    return Prompt(
        example_id=example_id,
        messages=(
            {"role": "system", "content": "You write SQLite queries."},
            {"role": "user", "content": question},
        ),
        prompt_version="p1",
    )


# Of different lengths, so a batch pads all but its longest.
PROMPTS = [
    _prompt("q-0", "How many singers are there?"),
    _prompt("q-1", "List the name of every stadium that holds more than 5000 people, by name."),
    _prompt("q-2", "Hi"),
]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_random_model(tmp_path_factory.mktemp("model"), generation=SHIPPED)


def _texts(generations: list[Generation]) -> list[str]:
    return [generation.text for generation in generations]


def _argmax_reply(
    model: PreTrainedModel, prompt_ids: list[int], max_new_tokens: int, eos: int
) -> list[int]:
    """Greedy decoding written out: the likeliest token, one step at a time, to the end."""
    ids = torch.tensor([prompt_ids])
    reply: list[int] = []
    with torch.inference_mode():
        while len(reply) < max_new_tokens:
            token = int(model(input_ids=ids).logits[0, -1].argmax())
            reply.append(token)
            if token == eos:
                break
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
    return reply


@pytest.mark.phase(2)
def test_greedy_reply_is_the_argmax_at_every_step(model_dir: Path) -> None:
    # q-0 and q-1 share a batch, so q-0 is padded on the left.
    backend = HfBackend(str(model_dir), device="cpu", batch_size=2)
    generations = backend.generate(PROMPTS, GenParams(max_new_tokens=24))

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    greedy, with_shipped_defaults = [], []
    for prompt, generation in zip(PROMPTS, generations, strict=True):
        text = render_chat(tokenizer, prompt.messages)
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        reply = _argmax_reply(model, prompt_ids, 24, tokenizer.eos_token_id)
        greedy.append(reply)

        assert (generation.example_id, generation.sample_idx) == (prompt.example_id, 0)
        assert generation.text == tokenizer.decode(reply, skip_special_tokens=True)
        assert generation.sql == extract_sql(generation.text)
        assert generation.tokens_in == len(prompt_ids)
        assert generation.tokens_out == len(reply)
        assert generation.latency_ms > 0
        assert generation.model_ref == str(model_dir)
        assert generation.adapter_ref is None

        with torch.inference_mode():
            output = model.generate(
                input_ids=torch.tensor([prompt_ids]), do_sample=False, max_new_tokens=24
            )
        with_shipped_defaults.append(output[0, len(prompt_ids) :].tolist())

    # Left to the checkpoint's defaults, generate decodes otherwise, so the backend has to
    # drop them to be greedy.
    assert with_shipped_defaults != greedy


class _EndRows(LogitsProcessor):
    """Ends rows of a batch early: at step ``n`` (from 1), row ``r`` must pick ``token``."""

    def __init__(self, ends: dict[int, tuple[int, int]]) -> None:
        self.ends = ends  # {step: (row, token)}
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        self.step += 1
        if self.step not in self.ends:
            return scores
        row, token = self.ends[self.step]
        forced = scores.clone()
        forced[row] = float("-inf")
        forced[row, token] = 0.0
        return forced


@pytest.mark.phase(2)
def test_rows_that_end_early_are_cut_at_their_end_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = byte_tokenizer()
    end_of_text = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # The checkpoint lists only <|endoftext|>, so a chat turn's end stops a row only if
    # the backend adds the tokenizer's own end token.
    model_dir = save_random_model(tmp_path / "model", generation={"eos_token_id": end_of_text})
    backend = HfBackend(str(model_dir), device="cpu", batch_size=3)
    assert backend.eos_token_ids == [end_of_text, end_of_turn]
    params = GenParams(max_new_tokens=24)
    free = backend.generate(PROMPTS, params)
    assert [g.tokens_out for g in free] == [24, 24, 24]

    # In one batch: q-1 ends its turn at step 5, q-2 at step 9 with the checkpoint's
    # token, and q-0 runs on, so generate pads the two after they end.
    sampling = hf_backend._sampling

    def ending(rows: Sequence[tuple[Prompt, str, int]], params: GenParams) -> LogitsProcessorList:
        processors = sampling(rows, params)
        processors.append(_EndRows({5: (1, end_of_turn), 9: (2, end_of_text)}))
        return processors

    monkeypatch.setattr(hf_backend, "_sampling", ending)
    cut = backend.generate(PROMPTS, params)

    assert [g.tokens_out for g in cut] == [24, 5, 9]
    assert cut[0].text == free[0].text
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    for generation, steps, end in ((cut[1], 4, end_of_turn), (cut[2], 8, end_of_text)):
        prompt = PROMPTS[int(generation.example_id[-1])]
        text = render_chat(tokenizer, prompt.messages)
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # Greedy up to the end token, and nothing generate padded the row with after it.
        before = _argmax_reply(model, prompt_ids, steps, end_of_turn)
        assert generation.text == tokenizer.decode([*before, end], skip_special_tokens=True)


@pytest.mark.phase(2)
def test_each_sample_draws_from_its_own_seed(model_dir: Path) -> None:
    # Six rows in batches of three: q-2's two samples share a batch with q-1's second.
    backend = HfBackend(str(model_dir), device="cpu", batch_size=3)
    params = GenParams(temperature=1.0, n=2, seed=7, max_new_tokens=16)

    together = backend.generate(PROMPTS, params)
    backend.batch_size = 1
    alone = backend.generate(PROMPTS[2:], params)
    reseeded = backend.generate(PROMPTS[2:], params.model_copy(update={"seed": 8}))

    assert [(g.example_id, g.sample_idx) for g in together] == [
        ("q-0", 0),
        ("q-0", 1),
        ("q-1", 0),
        ("q-1", 1),
        ("q-2", 0),
        ("q-2", 1),
    ]
    assert _texts(alone) == _texts(together[4:])
    assert together[4].text != together[5].text
    assert _texts(reseeded) != _texts(alone)


@pytest.mark.phase(2)
def test_temperature_and_top_p_shape_the_draws(model_dir: Path) -> None:
    backend = HfBackend(str(model_dir), device="cpu")

    def replies(**settings: float) -> list[str]:
        return _texts(backend.generate(PROMPTS, GenParams(max_new_tokens=16, **settings)))

    greedy = replies()
    # A top_p this small leaves only the likeliest token, and at the lowest temperature a
    # run takes it wins every draw: both sample the greedy reply.
    assert replies(temperature=1.0, top_p=1e-6) == greedy
    assert replies(temperature=MIN_TEMPERATURE) == greedy
    assert replies(temperature=1.0) != greedy


@pytest.mark.phase(2)
def test_adapter_changes_the_replies(model_dir: Path, tmp_path: Path) -> None:
    base = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        # PEFT starts B at zero, which makes a new adapter a no-op; this one is random.
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
    lora.save_pretrained(tmp_path / "adapter")
    params = GenParams(max_new_tokens=16)

    plain = HfBackend(str(model_dir), device="cpu").generate(PROMPTS, params)
    adapted = HfBackend(
        str(model_dir), adapter_ref=str(tmp_path / "adapter"), device="cpu"
    ).generate(PROMPTS, params)

    assert [g.adapter_ref for g in adapted] == [str(tmp_path / "adapter")] * len(PROMPTS)
    assert _texts(adapted) != _texts(plain)


@pytest.mark.phase(2)
def test_a_retry_is_one_more_turn_of_the_chat(model_dir: Path) -> None:
    backend = HfBackend(str(model_dir), device="cpu")
    params = GenParams(temperature=0.7, n=2, max_new_tokens=8, seed=3)
    firsts = backend.generate(PROMPTS[:2], params)
    failed = ExecResult(ok=False, error="the reply has no SQL query", error_kind="empty_sql")
    asks = [(PROMPTS[int(g.example_id[-1])], g, failed) for g in firsts]
    seconds = ask_again(backend, asks, params)
    assert [(g.example_id, g.sample_idx) for g in seconds] == [
        (g.example_id, g.sample_idx) for g in firsts
    ]
    # The model reads the whole conversation: its first reply as its own turn, then the
    # error, with the assistant's next turn opened.
    rendered = render_chat(backend.tokenizer, retry_prompt(*asks[0]).messages)
    assert rendered.endswith(
        f"<|im_start|>assistant\n{firsts[0].text}<|im_end|>\n<|im_start|>user\n"
        "That query failed with: the reply has no SQL query. Reply with a corrected query."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    tokens = backend.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    assert seconds[0].tokens_in == len(tokens) > firsts[0].tokens_in
    # Drawn the same way again, so a resumed run gets the same second replies.
    assert _texts(ask_again(backend, asks, params)) == _texts(seconds)
