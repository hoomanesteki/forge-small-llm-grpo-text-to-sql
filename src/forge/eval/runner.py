"""``forge eval run``: one resumable command from a model to stored, traceable scores.

A run evaluates one model on one split:

1. Load the split. Dev loads only through ``load_dev``, which needs a reason and spends
   one of the project's six looks (ADR-0004). The known failing golds are left out
   (``FAILING_GOLD_IDS``), and ``limit`` keeps the first questions: in id order on val
   and dev, in the loader's file order on train, so train_spider comes before the rowless
   train_others databases.
2. Build each prompt as training does, with ``fit_prompt`` measured by the backend's
   tokenizer (the fake backend counts words), so a long schema is cut the same way. A
   prompt still over the limit is sent as cut, and its question still counts.
3. Generate in batches of ``batch_size`` questions.
4. Run each predicted query in the run's ``SandboxPool``: model SQL is untrusted
   (ADR-0014), so each query has a hard deadline of ``PRED_TIMEOUT_MS`` plus the kill
   grace.
5. With ``--retry 1``, ask once more for each sample whose query failed to run, with
   the error (``forge.eval.retry``), and run the second reply's query in the same pool.
   The sample is scored on the second reply, and its first is kept as its first try.
6. Compare with the split's gold results for fast EX. Gold runs in process, once, and is
   cached (``forge.eval.gold_cache``).
7. Official EX (``forge.eval.testsuite``) in one batch at the end, on the same pool, at
   most ``OFFICIAL_ITEM_TIMEOUT_S`` an item, for each scored reply and each first try
   (``first_try_ex``). A prediction that timed out in step 4 or 5 is no match without
   the evaluator (ADR-0008).
8. Name the error class of each wrong prediction (``forge.eval.taxonomy``), against its
   database's columns, which are read in process like gold.
9. Each batch's first tries, generations and scores are stored before the next batch
   starts.
10. With a run store (``--store hub``), an earlier attempt's files come from it when
    this machine has none, and the run's folder goes to it when the run ends, finished
    or failed (``forge.ops.hub``).

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
from typing import Literal

from forge.core.types import Example, ExecResult, Generation, GenParams, Prompt, RunManifest, Score
from forge.data.prompts import TokenCounter, chat_token_counter, fit_prompt
from forge.data.spider import FAILING_GOLD_IDS, resolve_data_dir
from forge.data.splits import load_dev, load_train, load_val
from forge.eval import testsuite
from forge.eval.agreement import table_columns
from forge.eval.config import (
    FAKE_MODELS,
    OFFICIAL_ITEM_TIMEOUT_S,
    PRED_TIMEOUT_MS,
    EvalConfig,
    check_config,
)
from forge.eval.gold_cache import MAX_ROWS, gold_results
from forge.eval.mutations import Columns
from forge.eval.retry import ask_again, retried
from forge.eval.run_files import Key, RunFiles, RunPaths, finish, hold, key, start
from forge.eval.scoring import no_sql, score_generation
from forge.eval.testsuite import official_exec_match
from forge.infer.base import FakeBackend, InferenceBackend
from forge.ops.hub import RunNotFound, RunStore, hub_store
from forge.ops.manifest import CODE_SHA_ENV, build_manifest
from forge.ops.paths import RepoRootNotFound, cache_dir, repo_root, runs_dir
from forge.sql.executor import SqliteExecutor
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
    # Wall time of this attempt's generate calls, retries included. Rows of one batch
    # share its time as latency_ms, so summing the generations would count a batch once
    # per row.
    generate_s: float
    stored: str | None = None  # the run store that holds the run's folder, by name


class _TimedBackend:
    """A backend that adds up the wall time of its ``generate`` calls in ``seconds``."""

    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend
        self.name = backend.name
        self.seconds = 0.0

    def generate(self, prompts: Sequence[Prompt], params: GenParams) -> list[Generation]:
        started = time.perf_counter()
        generations = self.backend.generate(prompts, params)
        self.seconds += time.perf_counter() - started
        return generations


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
    store: RunStore | None = None,
) -> EvalRun:
    """Evaluate as ``config`` says, storing everything as the module docstring describes.

    ``backend`` replaces the one ``config`` names, such as a ``FakeBackend(fn)`` in a
    test; ``count_tokens`` measures its prompts, and counts words by default.
    ``store`` likewise replaces the run store ``config.store`` names, ``hub_store()`` for
    ``hub``. ``progress`` is called after each batch with the scores stored and the total.
    Raises ``DirtyTreeError`` on uncommitted changes unless ``allow_dirty``, and
    ``OSError`` when a finished run can't be stored.
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
        platform=config.platform,
        adapter_ref=config.adapter,
        allow_dirty=allow_dirty,
        repo=_code_root(),
        data_dir=root,
    )
    if store is None and config.store == "hub":
        store = hub_store()
    paths = RunPaths((runs_root or runs_dir()) / fresh.run_id, config.split)
    with hold(paths):
        if store is not None and not paths.manifest.is_file():
            _fetch_earlier(store, paths)
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
            if store is not None:
                _store_failed(store, paths)
            raise
        final = finish(paths, manifest, None)
        if store is not None:
            _store(store, paths)
    stored = None if store is None else str(store)
    return EvalRun(final, paths, questions, tuple(files.scores), files.resumed, generate_s, stored)


def _fetch_earlier(store: RunStore, paths: RunPaths) -> None:
    """Bring an earlier attempt's files from ``store``, so this attempt resumes it."""
    try:
        store.pull(paths.root.name, paths.root)
    except RunNotFound:
        return  # no earlier attempt: the run starts here
    log.info("fetched run %s from %s to resume it", paths.root.name, store)


def _store(store: RunStore, paths: RunPaths) -> None:
    """Make ``store``'s copy of the run its folder, saying how to try again when that fails.

    The folder is the whole run: an earlier attempt's files were fetched into it first.
    """
    run = paths.root.name
    try:
        store.put_folder(run, paths.root, replace=True)
    except OSError as exc:
        raise OSError(
            f"run {run} is in {paths.root}, but storing it in {store} failed: {exc}; "
            f"upload it with forge runs push {run}"
        ) from exc


def _store_failed(store: RunStore, paths: RunPaths) -> None:
    """Store a failed run's folder, so another machine can resume it."""
    try:
        _store(store, paths)
    # Whatever went wrong here is logged and not raised: the run's own error is the one
    # its caller needs.
    except Exception as exc:
        log.warning("%s", exc)


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
    kept = [e for e in loaded if e.id not in FAILING_GOLD_IDS]
    # By id, train_others would come first, and its gold results are all empty, 0 or NULL.
    examples = kept if config.split == "train" else sorted(kept, key=lambda e: e.id)
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
    schemas = _schemas(pending, root)
    timed = _TimedBackend(backend)
    with SandboxPool(workers, data_dir=root, imports=(testsuite.__name__,)) as pool:
        for begin in range(0, len(pending), batch_size):
            batch = pending[begin : begin + batch_size]
            prompts = [
                fit_prompt(example, config.prompt_version, measure, data_dir=root).prompt
                for example in batch
            ]
            replies = timed.generate(prompts, params)
            generations = _new_generations(replies, batch, params.n, files.done)
            preds = _execute(pool, generations, by_id)
            firsts = (
                _retry_failed(timed, pool, prompts, generations, preds, by_id, params)
                if config.retry
                else {}
            )
            scores: list[Score] = []
            for i, (generation, pred) in enumerate(zip(generations, preds, strict=True)):
                example = by_id[generation.example_id]
                score = score_generation(
                    example, generation, pred, gold[example.id], columns=schemas[example.db_id]
                )
                scores.append(retried(score, firsts[i][1]) if i in firsts else score)
            files.append(generations, scores, [first for first, _ in firsts.values()])
            if progress is not None:
                progress(len(files.scores), len(files.expected))
        _fill_official(files, by_id, pool)
    return len(chosen), files, timed.seconds


def _hf_backend(config: EvalConfig) -> tuple[InferenceBackend, TokenCounter]:
    # Imported here: torch and transformers take seconds to load, and only this needs them.
    from forge.infer.hf_backend import HfBackend

    hf = HfBackend(config.model, adapter_ref=config.adapter, device=config.device)
    return hf, chat_token_counter(hf.tokenizer)


def _schemas(examples: Sequence[Example], root: Path) -> dict[str, Columns]:
    """The tables and columns of each database the examples ask about."""
    # Schema reads are trusted SQL, so they run in process (ADR-0014).
    with SqliteExecutor(root) as executor:
        return {db: table_columns(executor, db) for db in sorted({e.db_id for e in examples})}


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


def _retry_failed(
    backend: InferenceBackend,
    pool: SandboxPool,
    prompts: Sequence[Prompt],
    generations: list[Generation],
    preds: list[ExecResult],
    examples: Mapping[str, Example],
    params: GenParams,
) -> dict[int, tuple[Generation, ExecResult]]:
    """Ask once more for each generation whose query failed to run, and run the second
    replies' queries in the pool.

    Each second reply and its result take the first's place in ``generations`` and
    ``preds``; the first replies and their results are returned, by position.
    """
    # ok is the whole rule: a query that ran is never retried, whatever its rows.
    failed = [i for i, pred in enumerate(preds) if not pred.ok]
    if not failed:
        return {}
    asked = {prompt.example_id: prompt for prompt in prompts}
    firsts = {i: (generations[i], preds[i]) for i in failed}
    seconds = ask_again(
        backend, [(asked[generations[i].example_id], *firsts[i]) for i in failed], params
    )
    for i, second, result in zip(failed, seconds, _execute(pool, seconds, examples), strict=True):
        generations[i], preds[i] = second, result
    return firsts


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
    """Official EX for every stored score that lacks it, and for each retried score's
    first try that lacks it, in one batch on the run's pool."""
    # (score index, the field the verdict fills, the reply it judges)
    todo: list[tuple[int, Literal["ex_official", "first_try_ex"], Generation]] = []
    for i, score in enumerate(files.scores):
        if score.ex_official is None:
            todo.append((i, "ex_official", files.generations[key(score)]))
        if score.attempts > 1 and score.first_try_ex is None:
            todo.append((i, "first_try_ex", files.first_tries[key(score)]))
    if not todo:
        return
    items: list[tuple[str, str | None, str]] = []
    for i, _, reply in todo:
        example = examples[files.scores[i].example_id]
        items.append((example.db_id, reply.sql, example.gold_sql))
    verdicts = official_exec_match(
        items, data_dir=pool.data_dir, item_timeout_s=OFFICIAL_ITEM_TIMEOUT_S, pool=pool
    )
    scores = list(files.scores)
    for (i, field, _), verdict in zip(todo, verdicts, strict=True):
        scores[i] = scores[i].model_copy(update={field: verdict})
    files.replace_scores(scores)


def _code_root() -> Path | None:
    """Where the run's commit is read: the Forge root, or None on a Kaggle kernel.

    A kernel runs the installed wheel with no Forge checkout, and its entry script names
    the code dataset's commit in ``FORGE_CODE_SHA``, which ``code_sha`` then takes. Anywhere
    else a missing Forge root is an error, so a run never records a commit it can't name.
    """
    try:
        return repo_root()
    except RepoRootNotFound:
        if os.environ.get(CODE_SHA_ENV):
            return None
        raise
