"""The vLLM backend with vLLM faked (tests.fakes.FakeVllm): how it maps the decoding
settings, builds the engine, renders prompts and reads the engine's outputs. The real
engine on a T4 is FG-307's to prove."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

import forge.infer
from forge.core.types import GenParams, Prompt
from forge.eval.config import EvalConfig
from forge.eval.runner import model_backend
from forge.infer import vllm_backend
from forge.infer.base import THINKING_FLAG, InferenceBackend, render_chat, sample_seed
from forge.infer.vllm_backend import REQUIREMENTS, VllmBackend, sampling_kwargs
from tests.fakes import FakeLoRARequest, FakeVllm, VocabTokenizer

ROOT = Path(__file__).resolve().parents[2]
QWEN3_TEMPLATE = "{%- if enable_thinking is defined and enable_thinking is false %}<think>"


def _prompt(example_id: str, question: str) -> Prompt:
    return Prompt(
        example_id=example_id,
        messages=(
            {"role": "system", "content": "You write SQLite queries."},
            {"role": "user", "content": question},
        ),
        prompt_version="p1",
    )


PROMPTS = [_prompt("spider-val-00001", "How many singers?"), _prompt("spider-val-00002", "Ages?")]


def _sql_reply(text: str, sample_idx: int) -> str:
    return f"```sql\nSELECT {sample_idx} -- {text.split()[-2]}\n```"


@pytest.mark.phase(3)
def test_sampling_params_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    params = GenParams(temperature=0.7, top_p=0.9, max_new_tokens=128, n=4, seed=3)
    kwargs = sampling_kwargs(params, "spider-val-00001")
    assert {k: v for k, v in kwargs.items() if k != "seed"} == {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
        "n": 4,
    }
    # Each prompt has a seed of its own, the same in any batch, from the run's seed and its
    # example id; 63 bits leave room for vLLM's seed + sample index.
    assert kwargs["seed"] == sample_seed(3, "spider-val-00001", 0) >> 1
    assert 0 <= kwargs["seed"] < 2**63
    assert sampling_kwargs(params, "spider-val-00002")["seed"] != kwargs["seed"]
    assert sampling_kwargs(params.model_copy(update={"seed": 4}), "spider-val-00001") != kwargs
    # The defaults decode greedily, one reply of up to 256 tokens: plan section 3.7.
    greedy = sampling_kwargs(GenParams(), "spider-val-00001")
    assert {k: v for k, v in greedy.items() if k != "seed"} == {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "n": 1,
    }

    # The backend sends those arguments and nothing else, one SamplingParams a prompt, so
    # no sampling default from the checkpoint's generation_config.json applies.
    vllm = FakeVllm(_sql_reply).install(monkeypatch)
    VllmBackend("org/model").generate(PROMPTS, params)
    [(_, call)] = vllm.calls
    assert [sp.kwargs for sp in call["sampling_params"]] == [
        sampling_kwargs(params, prompt.example_id) for prompt in PROMPTS
    ]


@pytest.mark.phase(3)
def test_import_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes `import vllm` fail as it does where vLLM isn't installed.
    for name in ("vllm", "vllm.lora", "vllm.lora.request"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "forge.infer.vllm_backend")
    monkeypatch.delattr(forge.infer, "vllm_backend")
    module = importlib.import_module("forge.infer.vllm_backend")
    assert module.VllmBackend.name == "vllm"
    with pytest.raises(ImportError, match=r"pip install -r kaggle/requirements-gpu\.txt") as err:
        module.VllmBackend("org/model")
    assert isinstance(err.value.__cause__, ImportError)
    assert (ROOT / REQUIREMENTS).is_file()
    # A setting no run takes is refused before vLLM is even looked for.
    with pytest.raises(ValueError, match="quantization must be one of bitsandbytes, got 'awq'"):
        module.VllmBackend("org/model", quantization="awq")


@pytest.mark.phase(3)
def test_engine_loads_the_model_in_fp16_with_lora_only_for_an_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = FakeVllm(_sql_reply).install(monkeypatch)
    plain = VllmBackend("org/model@0123abc")
    assert isinstance(plain, InferenceBackend)
    # A T4 has no bf16; the revision is the model's, and vLLM reads the tokenizer at it too.
    assert vllm.engines == [
        {
            "model": "org/model",
            "revision": "0123abc",
            "dtype": "half",
            "gpu_memory_utilization": 0.85,
            "max_model_len": 2048,
            "quantization": None,
        }
    ]
    assert plain.tokenizer is vllm.tokenizer and plain.lora_request is None

    base, adapter = tmp_path / "base", tmp_path / "runs" / "sft" / "checkpoints" / "best"
    base.mkdir()
    adapter.mkdir(parents=True)
    tuned = VllmBackend(
        str(base),
        adapter_ref=str(adapter),
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        max_lora_rank=16,
        quantization="bitsandbytes",
    )
    assert vllm.engines[1] == {
        "model": str(base),
        "revision": None,
        "dtype": "half",
        "gpu_memory_utilization": 0.6,
        "max_model_len": 4096,
        "quantization": "bitsandbytes",
        "enable_lora": True,
        "max_lora_rank": 16,
    }
    assert tuned.lora_request == FakeLoRARequest("adapter", 1, str(adapter))
    generations = tuned.generate(PROMPTS[:1], GenParams())
    assert vllm.calls[-1][1]["lora_request"] == tuned.lora_request
    assert [(g.model_ref, g.adapter_ref) for g in generations] == [(str(base), str(adapter))]


@pytest.mark.phase(3)
def test_an_eval_config_in_4bit_loads_its_engine_in_4bit(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm = FakeVllm(_sql_reply).install(monkeypatch)
    config = EvalConfig(model="org/big@0123abc", backend="vllm", host="kaggle")
    for quantization in (None, "bitsandbytes"):
        backend, _ = model_backend(dataclasses.replace(config, quantization=quantization))
        assert isinstance(backend, VllmBackend)
    assert [engine["quantization"] for engine in vllm.engines] == [None, "bitsandbytes"]


@pytest.mark.phase(3)
def test_hub_adapter_is_fetched_at_its_revision_before_the_model_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = FakeVllm(_sql_reply).install(monkeypatch)
    fetched: list[tuple[str, str | None]] = []

    def snapshot_download(repo_id: str, *, revision: str | None = None) -> str:
        fetched.append((repo_id, revision))
        if repo_id == "org/missing":
            raise OSError(f"{repo_id} is not on the Hub")
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(vllm_backend, "snapshot_download", snapshot_download)
    backend = VllmBackend("org/model", adapter_ref="org/adapter@abc123")
    assert fetched == [("org/adapter", "abc123")]
    assert backend.lora_request.lora_path == str(tmp_path / "snapshot")

    with pytest.raises(OSError, match="not on the Hub"):
        VllmBackend("org/model", adapter_ref="org/missing")
    assert len(vllm.engines) == 1  # no model was loaded for the missing adapter


@pytest.mark.phase(3)
def test_a_thinking_template_is_rendered_by_render_chat_with_thinking_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm = FakeVllm(_sql_reply, tokenizer=VocabTokenizer(QWEN3_TEMPLATE)).install(monkeypatch)
    VllmBackend("Qwen/Qwen3-1.7B").generate(PROMPTS, GenParams())
    # As any other template is, so a Qwen3 prompt is the hf backend's too, token for token
    # (tests/smoke/test_prompt_rendering.py); vLLM's own chat rendering is never used.
    [(method, call)] = vllm.calls
    assert method == "generate" and "chat_template_kwargs" not in call
    assert call["use_tqdm"] is False and call["lora_request"] is None
    # Each chat rendered with the flag off and the assistant turn opened.
    assert [kwargs for _, kwargs in vllm.tokenizer.calls] == [
        {"tokenize": False, "add_generation_prompt": True, THINKING_FLAG: False}
    ] * 2
    assert [chat for chat, _ in vllm.tokenizer.calls] == [list(p.messages) for p in PROMPTS]
    assert vllm.prompts == [render_chat(VocabTokenizer(), p.messages) for p in PROMPTS]


@pytest.mark.phase(3)
def test_other_templates_are_rendered_by_render_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm = FakeVllm(_sql_reply).install(monkeypatch)
    VllmBackend("Qwen/Qwen2.5-Coder-3B-Instruct").generate(PROMPTS, GenParams())
    [(method, call)] = vllm.calls
    assert method == "generate" and "chat_template_kwargs" not in call
    # The same text the hf backend continues, as ids with no <s> added before the template.
    tokenizer = VocabTokenizer()
    assert vllm.prompts == [render_chat(tokenizer, p.messages) for p in PROMPTS]
    assert not any(text.startswith("<s>") for text in vllm.prompts)


@pytest.mark.phase(3)
def test_generations_come_in_prompt_then_sample_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # The engine lists a prompt's samples last first here; the backend puts them in order.
    vllm = FakeVllm(_sql_reply, reverse=True).install(monkeypatch)
    backend = VllmBackend("org/model")
    params = GenParams(temperature=0.8, n=2)
    generations = backend.generate(PROMPTS, params)
    assert [(g.example_id, g.sample_idx) for g in generations] == [
        ("spider-val-00001", 0),
        ("spider-val-00001", 1),
        ("spider-val-00002", 0),
        ("spider-val-00002", 1),
    ]
    first = generations[0]
    assert first.text == "```sql\nSELECT 0 -- singers?\n```"
    assert first.sql == "SELECT 0 -- singers?"
    assert (first.model_ref, first.adapter_ref) == ("org/model", None)
    # Tokens as the engine counted them: the prompt it encoded, and the reply.
    assert first.tokens_in == len(vllm.prompts[0].split())
    assert first.tokens_out == len(first.text.split())
    # One call answers every prompt, and its wall time is every generation's latency.
    assert len(vllm.calls) == 1 and first.latency_ms >= 0
    assert len({g.latency_ms for g in generations}) == 1


@pytest.mark.phase(3)
def test_wrong_requests_and_outputs_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm = FakeVllm(_sql_reply).install(monkeypatch)
    backend = VllmBackend("org/model")
    with pytest.raises(ValueError, match="n=2 needs a temperature above 0"):
        backend.generate(PROMPTS, GenParams(n=2))
    # No prompts, no request: vLLM would take an empty list for one empty conversation.
    assert backend.generate([], GenParams()) == []
    assert vllm.calls == []

    engine = backend.llm

    class Short:
        """An engine that drops the last sample of every prompt."""

        def generate(self, prompts: list[Any], **kwargs: Any) -> list[Any]:
            outputs = engine.generate(prompts, **kwargs)
            for output in outputs:
                output.outputs.pop()
            return outputs

    backend.llm = Short()
    with pytest.raises(RuntimeError, match=r"samples \[0\] for spider-val-00001, not 0 to 1"):
        backend.generate(PROMPTS, GenParams(temperature=0.5, n=2))
