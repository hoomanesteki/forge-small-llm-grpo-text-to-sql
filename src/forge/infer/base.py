"""The interface every inference backend implements (plan section 3.2, stage 4).

``InferenceBackend.generate`` answers each prompt ``params.n`` times and returns the
generations in prompt order, then sample order. Each one carries the SQL that
``extract_sql`` finds in its text, its token counts and its latency, so the eval runner
can treat every backend alike.

``FakeBackend`` stands in for a model in tests, ``render_chat`` turns a prompt into the
text a model continues, and ``split_ref`` reads a model or adapter ref.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from huggingface_hub.utils import HFValidationError, validate_repo_id

from forge.core.hashing import canonical_json
from forge.core.types import Generation, GenParams, Prompt
from forge.sql.extract import extract_sql

# The chat template variable Qwen3 reads: false pre-fills an empty think block, so the
# model answers without reasoning first (plan section 3.7).
THINKING_FLAG = "enable_thinking"

# A fake model: (prompt, params, sample_idx) -> reply text.
ReplyFn = Callable[[Prompt, GenParams, int], str]


@runtime_checkable
class InferenceBackend(Protocol):
    """Generates ``params.n`` replies per prompt, in prompt order, then sample order."""

    name: str

    def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]: ...


class ChatTemplateTokenizer(Protocol):
    """The part of a Hugging Face tokenizer that rendering a chat needs."""

    def get_chat_template(self) -> str: ...

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> Any: ...


def check_params(params: GenParams) -> None:
    """Raise ``ValueError`` for settings no backend can honor, before any work starts."""
    # Negated so that a NaN temperature fails too.
    if not params.temperature >= 0:
        raise ValueError(f"temperature must be 0 or more, got {params.temperature}")
    if not 0 < params.top_p <= 1:
        raise ValueError(f"top_p must be above 0 and at most 1, got {params.top_p}")
    if params.max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1, got {params.max_new_tokens}")
    if params.n < 1:
        raise ValueError(f"n must be at least 1, got {params.n}")
    if params.n > 1 and params.temperature == 0:
        raise ValueError(
            f"n={params.n} needs a temperature above 0: greedy decoding gives the same reply "
            "every time"
        )


def sample_seed(seed: int, example_id: str, sample_idx: int) -> int:
    """The seed for one sample's random draws: the same in any batch, distinct across samples.

    A backend that samples seeds each sample with this, so a resumed run, or one with
    another batch size, draws every sample again exactly as before.
    """
    digest = hashlib.sha256(canonical_json([seed, example_id, sample_idx])).digest()
    return int.from_bytes(digest[:8], "big")


def chat_template_kwargs(tokenizer: ChatTemplateTokenizer) -> dict[str, Any]:
    """The template options Forge renders prompts with: ``enable_thinking=False`` for a
    template that reads the flag, such as Qwen3's, and none for any other."""
    return {THINKING_FLAG: False} if THINKING_FLAG in tokenizer.get_chat_template() else {}


def render_chat(tokenizer: ChatTemplateTokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    """The text a model continues to answer ``messages``: the chat in the tokenizer's
    template, with the assistant's turn opened."""
    text = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs(tokenizer),
    )
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}, not the prompt text")
    return text


def split_ref(ref: str) -> tuple[str, str | None]:
    """A model or adapter ref as ``from_pretrained`` and vLLM take it: a local directory
    with no revision, or a Hub repo id with the revision after an optional ``@``."""
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


class FakeBackend:
    """A backend whose replies come from ``fn(prompt, params, sample_idx)``, for tests.

    It refuses what every backend refuses (``check_params``), applies ``extract_sql``,
    counts whitespace-separated words as tokens, of all the prompt's messages for
    ``tokens_in`` and of the reply for ``tokens_out``, and times each call of ``fn``.
    """

    name = "fake"

    def __init__(self, fn: ReplyFn, *, model_ref: str = "fake") -> None:
        self.fn = fn
        self.model_ref = model_ref

    def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
        check_params(params)
        generations: list[Generation] = []
        for prompt in prompts:
            tokens_in = sum(len(message["content"].split()) for message in prompt.messages)
            for sample_idx in range(params.n):
                start = time.perf_counter()
                text = self.fn(prompt, params, sample_idx)
                latency_ms = (time.perf_counter() - start) * 1000
                if not isinstance(text, str):
                    raise TypeError(f"the fake's fn returned {type(text).__name__}, not str")
                generations.append(
                    Generation(
                        example_id=prompt.example_id,
                        model_ref=self.model_ref,
                        sample_idx=sample_idx,
                        text=text,
                        sql=extract_sql(text),
                        tokens_in=tokens_in,
                        tokens_out=len(text.split()),
                        latency_ms=latency_ms,
                    )
                )
        return generations
