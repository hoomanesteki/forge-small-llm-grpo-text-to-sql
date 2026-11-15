"""``forge eval run``: one resumable command from a model to stored, traceable scores.

A run evaluates one model on one split:

1. Load the split. Dev loads only through ``load_dev``, which needs a reason and spends
   one of the project's six looks (ADR-0004). The known failing golds are left out
   (``FAILING_GOLD_IDS``), and ``limit`` keeps the first questions in id order.
2. Build each prompt as training does, with ``fit_prompt`` measured by the backend's
   tokenizer (the fake backend counts words), so a long schema is cut the same way. A
   prompt still over the limit is sent as cut, and its question still counts.
3. Generate in batches of ``batch_size`` questions.
4. Run each predicted query in the run's ``SandboxPool``: model SQL is untrusted
   (ADR-0014), so each query has a hard deadline of ``PRED_TIMEOUT_MS`` plus the kill
   grace.
5. Compare with the split's gold results for fast EX. Gold runs in process, once, and is
   cached (``forge.eval.gold_cache``).
6. Official EX (``forge.eval.testsuite``) in one batch at the end, on the same pool, at
   most ``OFFICIAL_ITEM_TIMEOUT_S`` an item. A prediction that timed out in step 4 is no
   match without the evaluator (ADR-0008).
7. The error class: a placeholder until FG-204 (``forge.eval.scoring``).
8. Each batch's generations and scores are stored before the next batch starts.

The options and their checks are ``forge.eval.config``; the run's folder, its manifest
and how a killed run resumes are ``forge.eval.run_files``.
Resuming a dev run calls ``load_dev`` again, so it logs another look.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from forge.core.types import Example, ExecResult, Generation, GenParams, Prompt, RunManifest, Score
from forge.data.prompts import TokenCounter, chat_token_counter, fit_prompt
from forge.data.spider import FAILING_GOLD_IDS, resolve_data_dir
from forge.data.splits import load_dev, load_train, load_val
from forge.eval import testsuite
from forge.eval.config import (
    FAKE_MODELS,
    OFFICIAL_ITEM_TIMEOUT_S,
    PRED_TIMEOUT_MS,
    EvalConfig,
    check_config,
)
from forge.eval.gold_cache import MAX_ROWS, gold_results
from forge.eval.run_files import Key, RunFiles, RunPaths, finish, hold, key, start
from forge.eval.scoring import no_sql, score_generation
from forge.eval.testsuite import official_exec_match
from forge.infer.base import FakeBackend, InferenceBackend
from forge.ops.manifest import build_manifest
from forge.ops.paths import cache_dir, runs_dir
from forge.sql.sandbox import SandboxPool

log = logging.getLogger(__name__)

BATCH_SIZE = 32
WORKERS = min(4, os.cpu_count() or 1)


@dataclass(frozen=True)
class EvalRun:
    """A run as it was stored: its final manifest and every score, in file order."""

    manifest: RunManifest
    paths: RunPaths
    questions: int
    scores: tuple[Score, ...]
    resumed: int  # scores kept from an earlier attempt
    # Wall time of this attempt's generate calls. Rows of one batch share its time as
    # latency_ms, so summing the generations would count a batch once per row.
    generate_s: float


def word_tokens(prompt: Prompt) -> int:
    """A prompt's length the way ``FakeBackend`` counts it: words in all its messages."""
    return sum(len(message["content"].split()) for message in prompt.messages)


def fake_backend(model: str, examples: Sequence[Example]) -> FakeBackend:
    """The fake backend's ``gold`` or ``select-1`` model, answering in a sql block."""
    if model == "gold":
        gold = {example.id: example.gold_sql for example in examples}

        def gold_reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
            return f"```sql\n{gold[prompt.example_id]}\n```"

        return FakeBackend(gold_reply, model_ref=model)
    if model == "select-1":

        def constant_reply(prompt: Prompt, params: GenParams, sample_idx: int) -> str:
            return "```sql\nSELECT 1\n```"

        return FakeBackend(constant_reply, model_ref=model)
    raise ValueError(f"the fake backend's models are {', '.join(FAKE_MODELS)}, not {model!r}")


def run_eval(
    config: EvalConfig,
    *,
    backend: InferenceBackend | None = None,
    count_tokens: TokenCounter | None = None,
    allow_dirty: bool = False,
    data_dir: Path | None = None,
    runs_root: Path | None = None,
    cache_root: Path | None = None,
    workers: int = WORKERS,
    batch_size: int = BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> EvalRun:
    """Evaluate as ``config`` says, storing everything as the module docstring describes.

    ``backend`` replaces the one ``config`` names, such as a ``FakeBackend(fn)`` in a
    test; ``count_tokens`` measures its prompts, and counts words by default.
    ``progress`` is called after each batch with the scores stored and the total.
    Raises ``DirtyTreeError`` on uncommitted changes unless ``allow_dirty``.
    """
    check_config(config)
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    root = resolve_data_dir(data_dir).resolve()
    fresh = build_manifest(
        "eval",
        config.manifest_config(),
        model_ref=config.model,
        seed=config.params.seed,
        run_id=config.run_id,
        adapter_ref=config.adapter,
        allow_dirty=allow_dirty,
        data_dir=root,
    )
    paths = RunPaths((runs_root or runs_dir()) / fresh.run_id, config.split)
    with hold(paths):
        manifest = start(paths, fresh)
        try:
            questions, files, generate_s = _evaluate(
                config,
                manifest,
                paths,
                backend=backend,
                count_tokens=count_tokens,
                root=root,
                cache_root=cache_root or cache_dir(),
                workers=workers,
                batch_size=batch_size,
                progress=progress,
            )
        except BaseException as exc:
            # KeyboardInterrupt too: a run stopped by hand is failed, not still running.
            finish(paths, manifest, exc)
            raise
        final = finish(paths, manifest, None)
    return EvalRun(final, paths, questions, tuple(files.scores), files.resumed, generate_s)


def _evaluate(
    config: EvalConfig,
    manifest: RunManifest,
    paths: RunPaths,
    *,
    backend: InferenceBackend | None,
    count_tokens: TokenCounter | None,
    root: Path,
    cache_root: Path,
    workers: int,
    batch_size: int,
    progress: Callable[[int, int], None] | None,
) -> tuple[int, RunFiles, float]:
    if backend is None and config.backend == "hf":
        # Loaded before dev, so a model that fails to load spends no look at dev.
        backend, count_tokens = _hf_backend(config)
    loaded = _load_split(config, manifest.run_id, root)
    examples = sorted((e for e in loaded if e.id not in FAILING_GOLD_IDS), key=lambda e: e.id)
    if not examples:
        raise ValueError(f"the {config.split} split has no questions in {root}")
    gold = gold_results(
        examples, config.split, manifest.data_manifest_hash, cache_root=cache_root, data_dir=root
    )
    chosen = examples[: config.limit]
    if backend is None:
        backend, count_tokens = fake_backend(config.model, chosen), word_tokens
    measure = count_tokens or word_tokens
    params = config.params
    by_id = {example.id: example for example in chosen}
    files = RunFiles(paths, {(e.id, i) for e in chosen for i in range(params.n)})
    pending = [e for e in chosen if any((e.id, i) not in files.done for i in range(params.n))]
    if files.resumed:
        log.info(
            "resuming %s: %d scores stored, %d questions to go",
            manifest.run_id,
            files.resumed,
            len(pending),
        )
    generate_s = 0.0
    with SandboxPool(workers, data_dir=root, imports=(testsuite.__name__,)) as pool:
        for begin in range(0, len(pending), batch_size):
            batch = pending[begin : begin + batch_size]
            prompts = [
                fit_prompt(example, config.prompt_version, measure, data_dir=root).prompt
                for example in batch
            ]
            started = time.perf_counter()
            replies = backend.generate(prompts, params)
            generate_s += time.perf_counter() - started
            generations = _new_generations(replies, batch, params.n, files.done)
            preds = _execute(pool, generations, by_id)
            scores = [
                score_generation(by_id[g.example_id], g, pred, gold[g.example_id])
                for g, pred in zip(generations, preds, strict=True)
            ]
            files.append(generations, scores)
            if progress is not None:
                progress(len(files.scores), len(files.expected))
        _fill_official(files, by_id, pool)
    return len(chosen), files, generate_s


def _hf_backend(config: EvalConfig) -> tuple[InferenceBackend, TokenCounter]:
    # Imported here: torch and transformers take seconds to load, and only this needs them.
    from forge.infer.hf_backend import HfBackend

    hf = HfBackend(config.model, adapter_ref=config.adapter)
    return hf, chat_token_counter(hf.tokenizer)


def _load_split(config: EvalConfig, run_id: str, root: Path) -> list[Example]:
    if config.split == "dev":
        # check_config made sure there is a reason; load_dev logs the look once dev loads.
        return load_dev(reason=config.reason or "", run_id=run_id, data_dir=root)
    if config.split == "val":
        return load_val(root)
    return load_train(root)


def _new_generations(
    generations: Sequence[Generation], batch: Sequence[Example], n: int, done: Collection[Key]
) -> list[Generation]:
    """The backend's generations, checked against what it was asked for, without the
    samples an earlier attempt already scored."""
    asked = [(example.id, i) for example in batch for i in range(n)]
    if [key(generation) for generation in generations] != asked:
        raise RuntimeError(
            f"the backend returned {len(generations)} generations for {len(batch)} prompts "
            f"and {n} samples each, or not in prompt and sample order"
        )
    return [generation for generation in generations if key(generation) not in done]


def _execute(
    pool: SandboxPool, generations: Sequence[Generation], examples: Mapping[str, Example]
) -> list[ExecResult]:
    """Each generation's query run in the pool; a reply with no SQL gets ``no_sql()``."""
    results = [no_sql() for _ in generations]
    runnable = [(i, g.sql) for i, g in enumerate(generations) if g.sql is not None]
    ran = pool.run_many(
        [(examples[generations[i].example_id].db_id, sql) for i, sql in runnable],
        timeout_ms=PRED_TIMEOUT_MS,
        max_rows=MAX_ROWS,
    )
    for (i, _), result in zip(runnable, ran, strict=True):
        results[i] = result
    return results


def _fill_official(files: RunFiles, examples: Mapping[str, Example], pool: SandboxPool) -> None:
    """Official EX for every stored score that lacks it, in one batch on the run's pool."""
    todo = [i for i, score in enumerate(files.scores) if score.ex_official is None]
    if not todo:
        return
    items: list[tuple[str, str | None, str]] = []
    for i in todo:
        example = examples[files.scores[i].example_id]
        items.append((example.db_id, files.generations[key(files.scores[i])].sql, example.gold_sql))
    verdicts = official_exec_match(
        items, data_dir=pool.data_dir, item_timeout_s=OFFICIAL_ITEM_TIMEOUT_S, pool=pool
    )
    scores = list(files.scores)
    for i, verdict in zip(todo, verdicts, strict=True):
        scores[i] = scores[i].model_copy(update={"ex_official": verdict})
    files.replace_scores(scores)
