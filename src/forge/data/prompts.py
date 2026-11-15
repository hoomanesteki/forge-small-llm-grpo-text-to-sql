"""Prompts in the plan's format (section 3.7), fitted under Forge's prompt token limit.

A prompt is two chat messages: a fixed system instruction, then the schema and the
question. ``build_prompt`` gives the whole prompt. ``fit_prompt`` measures it with a
tokenizer and, past ``MAX_PROMPT_TOKENS``, applies the overflow policy:

1. Drop the sample rows (p2).
2. Drop the columns of every table the question doesn't name. The match is a
   case-insensitive substring test, so "flights" names ``flight`` and "border info"
   names ``border_info``.
3. If it is still too long, exclude the example from training. It stays in evaluation,
   where every example counts.

Each step is logged at INFO on ``forge.data.prompts``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from forge.core.types import Example, Prompt
from forge.data.schema import Table, check_version, render_schema, schema_tables, schema_text

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You write SQLite queries. Reply with exactly one SQL query in a ```sql code block. "
    "No explanation."
)
USER_TEMPLATE = "Database schema:\n{schema}\n\nQuestion: {question}"

# Forge's own limit (plan section 3.10). TRL 1.18 removed max_prompt_length from
# GRPOConfig, so long prompts are dealt with here, when the data is built.
MAX_PROMPT_TOKENS = 1536

TokenCounter = Callable[[Prompt], int]


class ChatTokenizer(Protocol):
    """The part of a Hugging Face tokenizer that counting needs."""

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class FittedPrompt:
    """A prompt after the overflow policy, with what the policy dropped."""

    prompt: Prompt
    tokens: int  # length of ``prompt``
    full_tokens: int  # length before the policy
    dropped_samples: bool = False
    dropped_columns: tuple[str, ...] = ()  # tables shown without their columns
    exclude_from_training: bool = False  # still over the limit; evaluation keeps it


@dataclass(frozen=True)
class PromptStats:
    prompts: int
    p50: int
    p95: int
    max: int
    max_tokens: int
    over_limit: int  # prompts longer than max_tokens before the policy
    dropped_samples: int
    dropped_columns: int
    excluded: int  # left out of training


def build_prompt(example: Example, version: str, *, data_dir: Path | None = None) -> Prompt:
    """The full prompt for this example: its database's whole schema, then the question."""
    return _prompt(example, version, schema_text(example.db_id, version, data_dir=data_dir))


def fit_prompt(
    example: Example,
    version: str,
    count_tokens: TokenCounter,
    *,
    max_tokens: int = MAX_PROMPT_TOKENS,
    data_dir: Path | None = None,
) -> FittedPrompt:
    """The prompt cut down by the overflow policy until it fits in ``max_tokens``.

    Training leaves out ``exclude_from_training`` examples; evaluation scores them too.
    """
    tables = schema_tables(example.db_id, version, data_dir=data_dir)
    prompt = _prompt(example, version, render_schema(tables))
    full_tokens = tokens = count_tokens(prompt)
    if tokens <= max_tokens:
        return FittedPrompt(prompt=prompt, tokens=tokens, full_tokens=full_tokens)

    dropped_samples = any(table.samples for table in tables)
    if dropped_samples:
        prompt = _prompt(example, version, render_schema(tables, samples=False))
        tokens = count_tokens(prompt)
        log.info(
            "%s: %d tokens is over %d; dropped the sample rows: %d tokens",
            example.id,
            full_tokens,
            max_tokens,
            tokens,
        )

    dropped_columns: tuple[str, ...] = ()
    if tokens > max_tokens:
        dropped_columns = _unnamed_tables(example.question, tables)
        if dropped_columns:
            before = tokens
            schema = render_schema(tables, samples=False, stubs=dropped_columns)
            prompt = _prompt(example, version, schema)
            tokens = count_tokens(prompt)
            log.info(
                "%s: %d tokens is over %d; dropped the columns of %d table(s) the question "
                "doesn't name (%s): %d tokens",
                example.id,
                before,
                max_tokens,
                len(dropped_columns),
                ", ".join(dropped_columns),
                tokens,
            )

    excluded = tokens > max_tokens
    if excluded:
        log.info(
            "%s: still %d tokens, over %d; excluded from training, kept in evaluation",
            example.id,
            tokens,
            max_tokens,
        )
    return FittedPrompt(
        prompt=prompt,
        tokens=tokens,
        full_tokens=full_tokens,
        dropped_samples=dropped_samples,
        dropped_columns=dropped_columns,
        exclude_from_training=excluded,
    )


def names_table(question: str, table: str) -> bool:
    """Whether the question names the table: case-insensitive, ``_`` matching a space too."""
    # The question as the prompt shows it, with each run of whitespace one space.
    text, name = " ".join(question.lower().split()), table.lower()
    return name in text or name.replace("_", " ") in text


def count_chat_tokens(tokenizer: ChatTokenizer, messages: Sequence[Mapping[str, str]]) -> int:
    """A chat's tokens as TRL's GRPO trainer encodes it, assistant turn opened."""
    encoded = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        # Qwen3 models run without thinking (plan section 3.7), which puts an empty
        # think block in every prompt. Templates without the flag ignore it.
        enable_thinking=False,
    )
    return len(encoded["input_ids"])


def chat_token_counter(tokenizer: ChatTokenizer) -> TokenCounter:
    """Count a prompt's tokens as ``count_chat_tokens`` does."""

    def count(prompt: Prompt) -> int:
        return count_chat_tokens(tokenizer, prompt.messages)

    return count


def load_tokenizer(name: str) -> ChatTokenizer:
    """A model's tokenizer, from a local directory or the Hub, where ``<repo>@<revision>``
    pins a commit as model refs do."""
    try:
        from transformers import AutoTokenizer

        from forge.infer.hf_backend import split_ref
    except ImportError as exc:
        raise ImportError(
            "counting tokens needs transformers, from the cpu extra: run `make setup`"
        ) from exc
    repo, revision = split_ref(name)
    tokenizer: ChatTokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    return tokenizer


def prompt_stats(
    fitted: Sequence[FittedPrompt], *, max_tokens: int = MAX_PROMPT_TOKENS
) -> PromptStats:
    """Full prompt lengths at p50, p95 and max, and what the overflow policy did."""
    if not fitted:
        raise ValueError("no prompts to summarize")
    lengths = sorted(item.full_tokens for item in fitted)
    return PromptStats(
        prompts=len(lengths),
        p50=_percentile(lengths, 50),
        p95=_percentile(lengths, 95),
        max=lengths[-1],
        max_tokens=max_tokens,
        over_limit=sum(length > max_tokens for length in lengths),
        dropped_samples=sum(item.dropped_samples for item in fitted),
        dropped_columns=sum(bool(item.dropped_columns) for item in fitted),
        excluded=sum(item.exclude_from_training for item in fitted),
    )


def render_prompt_stats(stats: PromptStats) -> str:
    rows = [
        ("prompts", stats.prompts),
        ("tokens p50", stats.p50),
        ("tokens p95", stats.p95),
        ("tokens max", stats.max),
        (f"over {stats.max_tokens} tokens", stats.over_limit),
        ("  sample rows dropped", stats.dropped_samples),
        ("  columns dropped", stats.dropped_columns),
        ("  excluded from training", stats.excluded),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value:>6}" for label, value in rows)


def _prompt(example: Example, version: str, schema: str) -> Prompt:
    # One line, like the schema's statements: a few Spider questions have double spaces.
    question = " ".join(example.question.split())
    return Prompt(
        example_id=example.id,
        messages=(
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(schema=schema, question=question)},
        ),
        prompt_version=check_version(version),
    )


def _unnamed_tables(question: str, tables: Sequence[Table]) -> tuple[str, ...]:
    return tuple(table.name for table in tables if not names_table(question, table.name))


def _percentile(ordered: Sequence[int], q: int) -> int:
    """Nearest rank: the smallest value with at least q% of the values at or below it."""
    # Integer ceiling of q% of the count, so no float rounding picks the wrong rank.
    return ordered[(q * len(ordered) + 99) // 100 - 1]
