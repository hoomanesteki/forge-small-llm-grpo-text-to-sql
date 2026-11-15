"""The parts of the transformers backend that need no model: device and dtype choice,
model refs, where a reply ends, and the seeded sampler."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest
import torch

from forge.infer import hf_backend
from forge.infer.hf_backend import (
    GumbelSampler,
    HfBackend,
    reply_ids,
    resolve_device,
    resolve_dtype,
    split_ref,
)


@pytest.mark.phase(2)
def test_auto_device_prefers_cuda_then_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")

    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="device string: gpu"):
        resolve_device("gpu")


@pytest.mark.phase(2)
def test_mps_by_name_is_refused_where_torch_cant_use_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run whose manifest says local-mps must not have run on the CPU instead (ADR-0016).
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="device mps needs Apple silicon"):
        resolve_device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps") == torch.device("mps")


@pytest.mark.phase(2)
def test_dtype_is_float32_unless_named() -> None:
    assert resolve_dtype(None) is torch.float32
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype(torch.float16) is torch.float16
    with pytest.raises(ValueError, match="dtype must be one of float32, float16, bfloat16"):
        resolve_dtype("fp16")


@pytest.mark.phase(2)
def test_split_ref(tmp_path: Path) -> None:
    # A directory is a local model, '@' and all.
    local = tmp_path / "ckpt@v2"
    local.mkdir()
    assert split_ref(str(local)) == (str(local), None)

    assert split_ref("org/model") == ("org/model", None)
    assert split_ref("org/model@0123abc") == ("org/model", "0123abc")
    assert split_ref("org/model@refs/pr/1") == ("org/model", "refs/pr/1")
    with pytest.raises(ValueError, match="nothing after its '@'"):
        split_ref("org/model@")
    with pytest.raises(FileNotFoundError, match="neither a directory nor a Hub repo id"):
        split_ref(str(tmp_path / "runs" / "missing" / "adapter"))


@pytest.mark.phase(2)
def test_reply_ends_at_the_first_end_token() -> None:
    # generate pads a finished row until the batch is done; here 0 pads and 2 ends.
    assert reply_ids([5, 6, 2, 0, 0], {2}) == [5, 6, 2]
    assert reply_ids([5, 2, 6, 2], {2}) == [5, 2]
    assert reply_ids([3, 5, 6], {2, 3}) == [3]
    # A reply cut off at max_new_tokens keeps every token.
    assert reply_ids([5, 6, 7], {2}) == [5, 6, 7]
    # Qwen pads with one of its end tokens.
    assert reply_ids([5, 0, 0], {0, 2}) == [5, 0]


@pytest.mark.phase(2)
def test_gumbel_sampler_draws_from_the_softmax() -> None:
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, float("-inf")])
    sampler = GumbelSampler([torch.Generator().manual_seed(0)])

    draws = torch.cat(
        [
            sampler(torch.zeros(1, 1, dtype=torch.long), logits[None]).argmax(dim=-1)
            for _ in range(20_000)
        ]
    )
    frequencies = torch.bincount(draws, minlength=5) / len(draws)

    assert torch.allclose(frequencies, torch.softmax(logits, dim=0), atol=0.01)
    # A token top-p filtered out (-inf) is never drawn.
    assert frequencies[4] == 0


@pytest.mark.phase(2)
def test_gumbel_sampler_rows_draw_only_from_their_own_generator() -> None:
    scores = torch.randn(2, 50, generator=torch.Generator().manual_seed(3))
    ids = torch.zeros(2, 1, dtype=torch.long)
    pair = GumbelSampler([torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)])
    alone = GumbelSampler([torch.Generator().manual_seed(2)])

    assert torch.equal(pair(ids, scores)[1], alone(ids[1:], scores[1:])[0])
    with pytest.raises(ValueError, match="2 rows of scores but 1 generators"):
        alone(ids, scores)


@pytest.mark.phase(2)
def test_batch_size_must_be_positive() -> None:
    # Refused before anything loads.
    with pytest.raises(ValueError, match="batch_size must be at least 1, got 0"):
        HfBackend("org/model", batch_size=0)


class _NoLoad:
    @staticmethod
    def from_pretrained(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("loaded before its arguments were checked")


@pytest.mark.phase(2)
def test_bad_adapter_ref_or_dtype_is_refused_before_anything_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for loader in ("AutoTokenizer", "AutoModelForCausalLM", "PeftModel"):
        monkeypatch.setattr(hf_backend, loader, _NoLoad)
    # A run folder's path, mistyped: not a directory, and too many segments for a Hub id.
    with pytest.raises(FileNotFoundError, match="neither a directory nor a Hub repo id"):
        HfBackend("org/model", adapter_ref="runs/2026-11-15-sft-a1b2c3d/adapter", device="cpu")
    with pytest.raises(ValueError, match="has nothing after its '@'"):
        HfBackend("org/model", adapter_ref="org/adapter@", device="cpu")
    with pytest.raises(ValueError, match="dtype must be one of float32, float16, bfloat16"):
        HfBackend("org/model", dtype="fp16", device="cpu")
