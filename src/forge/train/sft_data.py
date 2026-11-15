"""The SFT dataset: the train split as TRL conversational prompt-completion rows.

A row holds ``prompt``, the system and user messages of ``forge.data.prompts`` after the
overflow policy, and ``completion``, one assistant message with the gold query in a sql
code block, plus ``example_id`` and ``db_id``. In this form TRL's ``SFTTrainer`` puts
the loss on the completion only (``completion_only_loss``, its default for
prompt-completion rows): it masks what the tokenized prompt and the tokenized prompt
plus completion share, which works with any chat template, unlike ``assistant_only_loss``
and its generation tags. GRPO reads the same ``prompt`` column.

The rows come from ``load_train()`` alone, and ``assert_train_only`` checks them again
before they are returned. Left out, and counted in the meta's ``counts``:

- a question asked again on the same database, compared casefolded with each run of
  whitespace as one space; the first one asked stays;
- a gold query SQLite can't run (``FAILING_GOLD_IDS``);
- a prompt still over the token limit after the overflow policy, which keeps it for
  evaluation only;
- a prompt the policy cut down to show a table the gold query reads without its
  columns: training on it would teach the model to name columns it can't see.

The hash is the sha256 of the canonical JSON of the rows, the prompt version and the
tokenizer id; a Hub tokenizer given as ``<repo>@<revision>`` makes it the same on every
machine.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import sqlglot
from pydantic import BaseModel
from sqlglot import exp
from sqlglot.errors import SqlglotError

from forge.core.hashing import canonical_json, sha256_hex
from forge.core.types import Example, Prompt
from forge.data.prompts import (
    MAX_PROMPT_TOKENS,
    FittedPrompt,
    PromptStats,
    TokenCounter,
    fit_prompt,
    prompt_stats,
)
from forge.data.spider import FAILING_GOLD_IDS
from forge.data.splits import assert_train_only, load_train

log = logging.getLogger(__name__)


class SftRow(TypedDict):
    example_id: str
    db_id: str
    prompt: list[dict[str, str]]  # system, then user
    completion: list[dict[str, str]]  # one assistant message


class SftCounts(BaseModel, frozen=True):
    """Questions in, what was left out and why, and rows out."""

    train: int  # questions load_train() gave
    duplicates: int
    failing_gold: int
    over_limit: int  # still over the token limit after the overflow policy
    hidden_gold_tables: int  # a table the gold query reads was shown without columns
    rows: int


class SftMeta(BaseModel, frozen=True):
    hash: str
    prompt_version: str
    tokenizer: str
    counts: SftCounts
    # Lengths and what the overflow policy did, over every prompt it fitted.
    prompts: PromptStats


@dataclass(frozen=True)
class SftDataset:
    rows: list[SftRow]
    meta: SftMeta


def build_sft_dataset(
    prompt_version: str,
    tokenizer: str,
    count_tokens: TokenCounter,
    *,
    max_tokens: int = MAX_PROMPT_TOKENS,
    data_dir: Path | None = None,
) -> SftDataset:
    """The SFT rows of the train split, in its order, and their ``meta.json``.

    ``count_tokens`` measures prompts with the chat template of the model to be trained,
    which ``tokenizer`` names; ``chat_token_counter`` makes one from a tokenizer.
    """
    train = load_train(data_dir)
    unique = _drop_duplicates(train)
    runnable = [example for example in unique if example.id not in FAILING_GOLD_IDS]
    fitted: list[FittedPrompt] = []
    kept: list[Example] = []
    rows: list[SftRow] = []
    over_limit = hidden = 0
    for example in runnable:
        fit = fit_prompt(
            example, prompt_version, count_tokens, max_tokens=max_tokens, data_dir=data_dir
        )
        fitted.append(fit)
        if fit.exclude_from_training:
            over_limit += 1
            continue
        if tables := _hidden_tables(example, fit.dropped_columns):
            log.info(
                "%s: left out, its gold query reads %s, shown without columns",
                example.id,
                ", ".join(tables),
            )
            hidden += 1
            continue
        kept.append(example)
        rows.append(_row(example, fit.prompt))
    assert_train_only(kept)
    if not rows:
        raise ValueError(f"no SFT rows: all {len(train)} train questions were left out")

    counts = SftCounts(
        train=len(train),
        duplicates=len(train) - len(unique),
        failing_gold=len(unique) - len(runnable),
        over_limit=over_limit,
        hidden_gold_tables=hidden,
        rows=len(rows),
    )
    meta = SftMeta(
        hash=sft_hash(rows, prompt_version, tokenizer),
        prompt_version=prompt_version,
        tokenizer=tokenizer,
        counts=counts,
        prompts=prompt_stats(fitted, max_tokens=max_tokens),
    )
    return SftDataset(rows=rows, meta=meta)


def sft_hash(rows: Sequence[SftRow], prompt_version: str, tokenizer: str) -> str:
    """sha256 of the canonical JSON of the rows, the prompt version and the tokenizer id."""
    payload = {"rows": list(rows), "prompt_version": prompt_version, "tokenizer": tokenizer}
    return sha256_hex(canonical_json(payload))


def sql_block(sql: str) -> str:
    """The reply the model learns: the query alone in a sql code block (plan section 3.7)."""
    return f"```sql\n{sql.strip()}\n```"


def _drop_duplicates(examples: Sequence[Example]) -> list[Example]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for example in examples:
        key = (example.db_id, " ".join(example.question.split()).casefold())
        if key in seen:
            log.info("%s: left out, the same question as an earlier one", example.id)
            continue
        seen.add(key)
        unique.append(example)
    return unique


def _hidden_tables(example: Example, stubs: Sequence[str]) -> list[str]:
    """The tables shown without columns that the gold query reads, case-insensitively."""
    if not stubs:
        return []
    try:
        statements = [s for s in sqlglot.parse(example.gold_sql, read="sqlite") if s is not None]
    # sqlglot recurses per nesting level, and deep enough nesting runs out of stack.
    except (SqlglotError, RecursionError) as exc:
        raise ValueError(
            f"{example.id}: sqlglot can't read the gold query, so which of the tables shown "
            f"without columns it reads is unknown: {exc}"
        ) from exc
    ctes = {cte.alias.casefold() for s in statements for cte in s.find_all(exp.CTE)}
    read = {t.name.casefold() for s in statements for t in s.find_all(exp.Table)} - ctes
    return [table for table in stubs if table.casefold() in read]


def _row(example: Example, prompt: Prompt) -> SftRow:
    return {
        "example_id": example.id,
        "db_id": example.db_id,
        # Plain dicts, so the rows hash and save the same however the messages were built.
        "prompt": [dict(message) for message in prompt.messages],
        "completion": [{"role": "assistant", "content": sql_block(example.gold_sql)}],
    }
