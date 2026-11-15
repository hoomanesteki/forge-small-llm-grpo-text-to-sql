"""What an eval run evaluates: the options of ``forge eval run``, checked before it starts.

``EvalConfig`` holds one field per option. ``check_config`` refuses options that no run
can take, and those that a later story builds, before any data is loaded or any file
written. ``manifest_config`` is what a run's manifest records and hashes: every setting
that can change a score, the sandbox and official-EX bounds among them (ADR-0008), so a
resumed run has to match it exactly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from forge.core.types import GenParams, Split
from forge.data.prompts import MAX_PROMPT_TOKENS
from forge.data.schema import check_version
from forge.data.splits import SPLITS
from forge.eval.agreement import GOLD_TIMEOUT_MS
from forge.eval.gold_cache import MAX_ROWS
from forge.infer.base import check_params
from forge.sql.executor import DEFAULT_TIMEOUT_MS

Backend = Literal["hf", "vllm", "llamacpp", "fake"]
Store = Literal["local", "hub"]
BACKENDS: tuple[Backend, ...] = get_args(Backend)
STORES: tuple[Store, ...] = get_args(Store)
# The fake backend's models: one answers with each question's gold query, the other
# always with SELECT 1. Tests build other fakes with FakeBackend(fn).
FAKE_MODELS = ("gold", "select-1")
# Backends that later stories build, with the story that brings each.
LATER_BACKENDS: Mapping[str, str] = {"vllm": "FG-306", "llamacpp": "FG-702"}

PRED_TIMEOUT_MS = DEFAULT_TIMEOUT_MS
# Spider's gold queries run in milliseconds, so 10 s is ample for an item, where the
# evaluator's own 60 s would let one adversarial prediction hold it for a minute while
# its memory grows (ADR-0008).
OFFICIAL_ITEM_TIMEOUT_S = 10.0

# Run ids name folders here and in the Hub run store, so they keep to make_run_id's
# alphabet, and nothing like "../x" can leave the runs folder.
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]*")


class NotAvailable(RuntimeError):
    """An option that a later story builds."""


@dataclass(frozen=True)
class EvalConfig:
    """The options of ``forge eval run``, one field each."""

    model: str
    adapter: str | None = None
    backend: Backend = "hf"
    split: Split = "val"
    reason: str | None = None
    limit: int | None = None
    prompt_version: str = "p1"
    n: int = 1
    temperature: float = 0.0
    retry: int = 0
    run_id: str | None = None
    store: Store = "local"

    @property
    def params(self) -> GenParams:
        return GenParams(temperature=self.temperature, n=self.n)

    def manifest_config(self) -> dict[str, Any]:
        """What the manifest records and hashes: everything that can change a score.

        The reason is in the dev usage log, and the store says only where files go.
        """
        return {
            "backend": self.backend,
            "model": self.model,
            "adapter": self.adapter,
            "split": self.split,
            "limit": self.limit,
            "prompt_version": self.prompt_version,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "generation": self.params.model_dump(),
            "retry": self.retry,
            "execution": {
                "timeout_ms": PRED_TIMEOUT_MS,
                "gold_timeout_ms": GOLD_TIMEOUT_MS,
                "max_rows": MAX_ROWS,
                "official_item_timeout_s": OFFICIAL_ITEM_TIMEOUT_S,
            },
        }


def check_config(config: EvalConfig) -> None:
    """Refuse options no run can take, before anything is loaded or written.

    Raises ``ValueError`` for a wrong option and ``NotAvailable`` for one that a later
    story builds.
    """
    if config.split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}, got {config.split!r}")
    if config.split == "dev" and not (config.reason or "").strip():
        raise ValueError(
            "--split dev needs --reason: every dev evaluation is logged in "
            "docs/dev-usage.md, and there are six at most (ADR-0004)"
        )
    if config.split != "dev" and config.reason is not None:
        raise ValueError("--reason goes with --split dev, the only split whose uses are logged")
    if config.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {config.backend!r}")
    if config.store not in STORES:
        raise ValueError(f"store must be one of {', '.join(STORES)}, got {config.store!r}")
    check_version(config.prompt_version)
    if config.limit is not None and config.limit < 1:
        raise ValueError(f"limit must be at least 1, got {config.limit}")
    if config.retry not in (0, 1):
        raise ValueError(f"retry must be 0 or 1, got {config.retry}")
    if config.run_id is not None:
        check_run_id(config.run_id)
    if config.backend == "fake":
        if config.model not in FAKE_MODELS:
            raise ValueError(
                f"--backend fake takes --model {' or '.join(FAKE_MODELS)}, got {config.model!r}"
            )
        if config.adapter is not None:
            raise ValueError("--backend fake loads no adapter")
    check_params(config.params)
    if config.backend in LATER_BACKENDS:
        raise NotAvailable(
            f"the {config.backend} backend is not available yet: it comes with "
            f"{LATER_BACKENDS[config.backend]}"
        )
    if config.store == "hub":
        raise NotAvailable("--store hub is available after FG-304 (Hub run store)")


def check_run_id(run_id: str) -> None:
    """Raise ``ValueError`` unless ``run_id`` is a plain folder name in the run id alphabet."""
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(
            f"run id {run_id!r} names a folder, so it takes lowercase letters, digits "
            "and '-', and starts with a letter or digit"
        )
