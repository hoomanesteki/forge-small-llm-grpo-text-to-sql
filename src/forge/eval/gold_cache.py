"""Gold results for a split, run once and kept on disk as JSON lines.

Fast EX compares each prediction's result with its gold query's. Gold SQL is trusted
(ADR-0014), so it runs in process through ``SqliteExecutor.run``, with a long timeout
because a slow gold query is not the prediction's fault, and with a row cap above
Spider's largest gold result (20,662 rows, wta_1 on dev), so no gold result is cut
short. Predictions scored against them run with the same cap.

A split's results go to ``<cache>/gold/<split>-<data manifest hash[:12]>.jsonl``, where
``<cache>`` is ``forge.ops.paths.cache_dir()``: a header line, then one line per question
with its ``ExecResult``. The data manifest hash pins every question and database, so the
file holds as long as the data does; one whose header, questions or gold queries don't
fit is rebuilt. It is JSON rather than pickle, so reading it runs no code. Two kinds of
value are tagged, because pydantic's JSON would turn bytes into text and an infinite
float into ``null``: bytes as ``{"bytes": "<hex>"}`` and infinite floats as
``{"float": "inf"}``.

A gold query that fails raises ``GoldFailed``: that is a data problem, and no prediction
could match it. Callers leave out the known failures (``FAILING_GOLD_IDS``).

``split_gold`` serves a whole split the way the eval runner and the GRPO rewards read it,
and ``forge data cache-gold`` calls it to build the file ahead of a run; any of them builds
it on first use. Train comes from ``load_train`` alone, checked by ``assert_train_only``.
Only train and val are built ahead: a dev file is built by the dev evaluation that reads
it.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from forge.core.types import Example, ExecResult, Split
from forge.data.manifest import data_manifest_hash
from forge.data.spider import FAILING_GOLD_IDS, resolve_data_dir
from forge.data.splits import assert_train_only, load_train, load_val
from forge.eval.agreement import GOLD_TIMEOUT_MS
from forge.ops.paths import cache_dir
from forge.sql.executor import SqliteExecutor

log = logging.getLogger(__name__)

# Above Spider's largest gold result, and the official evaluator's own ceiling
# (exec_eval.MAX_ROWS), so a gold result is never truncated.
MAX_ROWS = 100_000
# Raised whenever the line format changes, so older files are rebuilt, not misread.
FORMAT = 1

GoldSplit = Literal["train", "val"]
GOLD_SPLITS: tuple[GoldSplit, ...] = ("train", "val")


class GoldFailed(ValueError):
    """A gold query failed on its database."""

    def __init__(self, example: Example, result: ExecResult) -> None:
        super().__init__(
            f"the gold query of {example.id} fails on {example.db_id} "
            f"({result.error_kind}: {result.error}): {example.gold_sql}"
        )


class CacheHeader(BaseModel, frozen=True):
    """What a cache file was built for; a file whose header differs is rebuilt."""

    format: int = FORMAT
    split: Split
    data_manifest_hash: str
    max_rows: int = MAX_ROWS
    examples: int


@dataclass(frozen=True)
class SplitGold:
    """A split's gold results, by example id, and the file that holds them."""

    path: Path
    results: dict[str, ExecResult]
    built: bool  # run just now, rather than read from the file


class _Line(BaseModel, frozen=True):
    example_id: str
    db_id: str
    gold_sql: str
    result: ExecResult


def cache_path(cache_root: Path, split: Split, data_manifest_hash: str) -> Path:
    return cache_root / "gold" / f"{split}-{data_manifest_hash[:12]}.jsonl"


def gold_results(
    examples: Sequence[Example],
    split: Split,
    data_manifest_hash: str,
    *,
    cache_root: Path,
    data_dir: Path | None = None,
) -> dict[str, ExecResult]:
    """Each example's gold result, by example id: read from the split's cache file, or
    run and written there when the file is missing or doesn't fit.

    ``examples`` should be the whole split, so that one file serves every run on it.
    """
    return _read_or_build(examples, split, data_manifest_hash, cache_root, data_dir).results


def split_gold(
    split: GoldSplit, *, data_dir: Path | None = None, cache_root: Path | None = None
) -> SplitGold:
    """The gold results of every question of ``split`` but ``FAILING_GOLD_IDS``, the ones
    the eval runner scores, as ``gold_results`` reads or builds them.

    ``cache_root`` defaults to ``forge.ops.paths.cache_dir()``. Raises ``ValueError`` for
    any other split, as dev's file is built only by a dev evaluation, and ``SplitLeak``
    should train hold anything else.
    """
    root = resolve_data_dir(data_dir)
    if split == "train":
        loaded = load_train(root)
        assert_train_only(loaded)
    elif split == "val":
        loaded = load_val(root)
    else:
        raise ValueError(f"gold is cached ahead of a run for {' and '.join(GOLD_SPLITS)} only")
    examples = [example for example in loaded if example.id not in FAILING_GOLD_IDS]
    if not examples:
        raise ValueError(f"the {split} split has no questions in {root}")
    digest = data_manifest_hash(root)
    return _read_or_build(examples, split, digest, cache_root or cache_dir(), root)


def _read_or_build(
    examples: Sequence[Example],
    split: Split,
    data_manifest_hash: str,
    cache_root: Path,
    data_dir: Path | None,
) -> SplitGold:
    path = cache_path(cache_root, split, data_manifest_hash)
    header = CacheHeader(split=split, data_manifest_hash=data_manifest_hash, examples=len(examples))
    cached = read_cache(path, header, examples)
    if cached is not None:
        return SplitGold(path, cached, built=False)
    results = run_gold(examples, data_dir=data_dir)
    write_cache(path, header, examples, results)
    log.info("built the %s gold cache: %d results in %s", split, len(results), path)
    return SplitGold(path, results, built=True)


def run_gold(examples: Sequence[Example], *, data_dir: Path | None = None) -> dict[str, ExecResult]:
    """Run each gold query in process, as trusted SQL; raise ``GoldFailed`` if one fails."""
    results: dict[str, ExecResult] = {}
    with SqliteExecutor(data_dir) as executor:
        for example in examples:
            result = executor.run(
                example.db_id, example.gold_sql, timeout_ms=GOLD_TIMEOUT_MS, max_rows=MAX_ROWS
            )
            if not result.ok:
                raise GoldFailed(example, result)
            results[example.id] = result
    return results


def write_cache(
    path: Path,
    header: CacheHeader,
    examples: Sequence[Example],
    results: dict[str, ExecResult],
) -> None:
    """Write the file whole, then move it into place, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header.model_dump_json()]
    for example in examples:
        result = results[example.id]
        rows = None if result.rows is None else [_tagged(row) for row in result.rows]
        line = _Line(
            example_id=example.id,
            db_id=example.db_id,
            gold_sql=example.gold_sql,
            result=result.model_copy(update={"rows": rows}),
        )
        lines.append(line.model_dump_json())
    # Named for the process, so two runs building the same file can't mix their writes.
    partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with partial.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def read_cache(
    path: Path, header: CacheHeader, examples: Sequence[Example]
) -> dict[str, ExecResult] | None:
    """The cached results, or ``None`` when the file is missing or doesn't fit, which
    means it must be built again."""
    try:
        first, *rest = path.read_bytes().splitlines() or [b""]
    except FileNotFoundError:
        return None
    try:
        found = CacheHeader.model_validate_json(first)
        lines = [_Line.model_validate_json(line) for line in rest]
        results = {
            line.example_id: line.result.model_copy(
                update={"rows": None if line.result.rows is None else _untagged(line.result.rows)}
            )
            for line in lines
        }
    # A ValidationError is a ValueError, and so is a tag that isn't one of ours.
    except ValueError as exc:
        log.warning("rebuilding %s: it can't be read (%s)", path, exc)
        return None
    if found != header:
        log.info("rebuilding %s: it was built for %s, not %s", path, found, header)
        return None
    cached = {line.example_id: (line.db_id, line.gold_sql) for line in lines}
    wanted = {example.id: (example.db_id, example.gold_sql) for example in examples}
    if len(cached) != len(lines) or cached != wanted:
        log.info("rebuilding %s: its questions or gold queries differ from the split's", path)
        return None
    return results


def _tagged(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_tag(value) for value in row)


def _tag(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": repr(value)}
    return value


def _untagged(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [tuple(_untag(value) for value in row) for row in rows]


def _untag(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if value.keys() == {"bytes"}:
        return bytes.fromhex(value["bytes"])
    if value.keys() == {"float"} and value["float"] in ("inf", "-inf", "nan"):
        return float(value["float"])
    raise ValueError(f"unknown tagged value {value!r}")
