"""Shared contracts (plan section 3.5).

The Protocols in that section arrive with the stories that implement them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

Hardness = Literal["easy", "medium", "hard", "extra"]
Split = Literal["train", "val", "dev"]


class Example(BaseModel, frozen=True):
    id: str  # "spider-train-00042", stable across runs
    db_id: str
    question: str
    gold_sql: str
    hardness: Hardness  # from the official evaluator's hardness function
    split: Split


class Prompt(BaseModel, frozen=True):
    example_id: str
    messages: tuple[dict[str, str], ...]  # chat format: {"role", "content"}
    prompt_version: str  # "p1", "p2" (ADR-0007)


class GenParams(BaseModel, frozen=True):
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    n: int = 1
    seed: int = 0


class Generation(BaseModel):
    example_id: str
    model_ref: str  # "Qwen/Qwen3-4B" or "<hub repo>@<revision>"
    adapter_ref: str | None = None
    sample_idx: int = 0
    text: str
    sql: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: float


ErrorKind = Literal[
    "syntax", "no_such_table", "no_such_column", "timeout", "denied", "empty_sql", "other"
]


class ExecResult(BaseModel):
    ok: bool
    columns: list[str] | None = None
    rows: list[tuple[Any, ...]] | None = None
    error: str | None = None
    error_kind: ErrorKind | None = None
    truncated: bool = False
    ms: float = 0.0


class Score(BaseModel):
    example_id: str
    sample_idx: int = 0
    hardness: Hardness
    valid: bool  # executed without error
    ex_fast: bool  # forge.sql.compare
    ex_official: bool | None = None  # test-suite evaluator, when run
    error_class: str | None = None  # taxonomy label (FG-204)


class RunManifest(BaseModel):
    run_id: str  # "2026-11-03-qwen3-4b-grpo-a1b2c3d"
    kind: Literal["eval", "sft", "grpo", "export", "audit"]
    git_sha: str
    config: dict[str, Any]
    config_hash: str
    data_manifest_hash: str
    model_ref: str
    adapter_ref: str | None = None
    parent_run: str | None = None
    platform: Literal["local-cpu", "kaggle", "colab", "modal"]
    gpu: str | None = None
    seed: int
    started_at: datetime
    ended_at: datetime | None = None
    gpu_hours: float = 0.0
    cost_usd: float = 0.0
    outputs: dict[str, str] = {}  # logical name -> path in run store
    status: Literal["running", "stopped_time_guard", "finished", "failed"] = "running"
