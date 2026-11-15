"""HfBackend generating through real transformers and PEFT code, on a random model built
offline (``tests/tiny_hf.py``), so this runs in CI with no download."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from forge.core.types import Generation, GenParams, Prompt
from forge.infer.base import render_chat
from forge.infer.hf_backend import HfBackend
from forge.sql.extract import extract_sql
from tests.tiny_hf import save_random_model

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
    # A top_p this small leaves only the likeliest token, and at a temperature this low
    # it wins every draw: both sample the greedy reply.
    assert replies(temperature=1.0, top_p=1e-6) == greedy
    assert replies(temperature=1e-4) == greedy
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
