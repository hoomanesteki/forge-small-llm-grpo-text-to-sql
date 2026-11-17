"""In-training evaluation: fast EX on a fixed val subset while a model trains (FG-504).

It lives in ``forge.eval``, not ``forge.train``: training code never reads val (ADR-0004,
and ``tests/unit/test_splits.py`` scans the train package for the val and dev loaders),
and scoring a model on val is evaluation. ``forge.train.grpo`` gets the callback from
``val_eval_callback`` and never holds a val question itself.

``ValEvalCallback`` runs at the end of every ``every``-th optimizer step, before the
trainer saves that step's checkpoint, so a line's EX is that checkpoint's:

1. Greedy replies from the training model itself (``HfBackend.around``), at most
   ``max_new_tokens`` tokens each, to the questions of ``val_subset``: the same ones in
   every run, those whose ids hash lowest, so runs and their checkpoints compare. Prompts
   are fitted as the eval runner fits them, with the model's tokenizer.
2. Each reply's SQL run in the run's ``SandboxPool`` (ADR-0014), with the eval runner's
   timeout and row cap.
3. Fast EX against val's gold results (``forge.eval.gold_cache``), the comparator the
   rewards use (ADR-0008). Official EX waits for the full val evaluation of the
   checkpoints chosen from these lines (FG-508).
4. ``record(step, line)``, ``line`` holding ``val/ex_fast`` and ``val/valid`` (shares of
   the questions), ``val/questions`` and ``val/seconds``; ``forge.train.grpo`` writes it
   to the run's ``metrics.jsonl``.

A step after which training stops before ``max_steps``, as the time guard stops it, is
skipped: the deadline kept no time for an evaluation. The callback must come after the
guard in the trainer's list, so the guard has said so by then.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from forge.core.types import Example, ExecResult, GenParams, Prompt
from forge.data.prompts import ChatTokenizer, chat_token_counter, fit_prompt
from forge.data.spider import FAILING_GOLD_IDS, resolve_data_dir
from forge.data.splits import load_val
from forge.eval.config import PRED_TIMEOUT_MS
from forge.eval.gold_cache import MAX_ROWS, split_gold
from forge.infer.base import InferenceBackend
from forge.sql.compare import order_matters, results_match
from forge.sql.sandbox import SandboxPool

log = logging.getLogger(__name__)

# The plan's subset (FG-504): enough questions to see EX move, few enough to run often.
VAL_EVAL_EXAMPLES = 200
EX_KEY = "val/ex_fast"
# Hands a step's line on, such as forge.train.callbacks.MetricsCallback.write.
Record = Callable[[int, Mapping[str, Any]], None]
# The backend that generates with the trainer's model and tokenizer, named by model_ref.
BackendFor = Callable[[Any, Any, str], InferenceBackend]


def val_subset(size: int, *, data_dir: Path | None = None) -> list[Example]:
    """The ``size`` val questions in-training eval asks, in id order: those whose ids hash
    lowest, so every run asks the same ones, or all of val when it has fewer. The known
    failing golds are left out, as the eval runner leaves them out."""
    if size < 1:
        raise ValueError(f"an in-training eval asks 1 question or more, not {size}")
    val = [example for example in load_val(data_dir) if example.id not in FAILING_GOLD_IDS]
    if not val:
        raise ValueError(f"the val split has no questions in {resolve_data_dir(data_dir)}")
    ranked = sorted(val, key=lambda example: hashlib.sha256(example.id.encode()).hexdigest())
    return sorted(ranked[:size], key=lambda example: example.id)


def in_memory_backend(model: Any, tokenizer: Any, model_ref: str) -> InferenceBackend:
    """``HfBackend.around`` the trainer's model, as ``ValEvalCallback`` generates by default."""
    from forge.infer.hf_backend import HfBackend

    return HfBackend.around(model, tokenizer, model_ref=model_ref)


class ValEvalCallback(TrainerCallback):
    """Score the training model on ``examples`` every ``every`` steps, as the module
    docstring says. ``prompts`` are the examples' prompts, in their order, ``gold`` their
    gold results by id, and ``pool`` runs the replies' SQL; the caller owns the pool."""

    def __init__(
        self,
        examples: Sequence[Example],
        prompts: Sequence[Prompt],
        gold: Mapping[str, ExecResult],
        pool: SandboxPool,
        *,
        every: int,
        max_new_tokens: int,
        model_ref: str,
        record: Record,
        backend_for: BackendFor = in_memory_backend,
    ) -> None:
        if every < 1:
            raise ValueError(f"an in-training eval comes every 1 step or more, not every {every}")
        if [prompt.example_id for prompt in prompts] != [example.id for example in examples]:
            raise ValueError("the prompts must be the examples', one each, in their order")
        missing = [example.id for example in examples if example.id not in gold]
        if missing:
            raise ValueError(f"no gold result for {', '.join(missing[:3])}")
        self.examples = list(examples)
        self.prompts = list(prompts)
        self.gold = gold
        self.pool = pool
        self.every = every
        self.params = GenParams(temperature=0.0, max_new_tokens=max_new_tokens)
        self.model_ref = model_ref
        self.record = record
        self.backend_for = backend_for

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        processing_class: Any = None,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        if step % self.every:
            return
        if control.should_training_stop and step < state.max_steps:
            log.info("val eval: none at step %d, after which training stops early", step)
            return
        if model is None or processing_class is None:
            raise ValueError("the trainer gave the val eval no model or no tokenizer")
        line = self.evaluate(self.backend_for(model, processing_class, self.model_ref))
        log.info(
            "val eval at step %d: fast EX %.3f on %d questions",
            step,
            line[EX_KEY],
            len(self.examples),
        )
        self.record(step, line)

    def evaluate(self, backend: InferenceBackend) -> dict[str, float]:
        """Greedy replies from ``backend`` to the questions, scored by fast EX."""
        started = time.perf_counter()
        replies = backend.generate(self.prompts, self.params)
        if [reply.example_id for reply in replies] != [example.id for example in self.examples]:
            raise RuntimeError(
                f"the backend returned {len(replies)} replies for {len(self.examples)} prompts, "
                "or not in their order"
            )
        runnable = [(i, reply.sql) for i, reply in enumerate(replies) if reply.sql is not None]
        ran = self.pool.run_many(
            [(self.examples[i].db_id, sql) for i, sql in runnable],
            timeout_ms=PRED_TIMEOUT_MS,
            max_rows=MAX_ROWS,
        )
        valid = matched = 0
        for (i, _), pred in zip(runnable, ran, strict=True):
            example = self.examples[i]
            valid += pred.ok
            matched += results_match(
                self.gold[example.id], pred, order_matters=order_matters(example.gold_sql)
            )
        questions = len(self.examples)
        return {
            EX_KEY: matched / questions,
            "val/valid": valid / questions,
            "val/questions": float(questions),
            "val/seconds": round(time.perf_counter() - started, 3),
        }


def val_eval_callback(
    pool: SandboxPool,
    tokenizer: ChatTokenizer,
    *,
    every: int,
    examples: int,
    prompt_version: str,
    max_new_tokens: int,
    model_ref: str,
    record: Record,
    data_dir: Path | None = None,
    cache_root: Path | None = None,
    backend_for: BackendFor = in_memory_backend,
) -> ValEvalCallback:
    """The in-training eval of a run: ``examples`` questions of ``val_subset``, prompted
    at ``prompt_version`` and fitted with ``tokenizer``, the model's, and scored against
    val's gold results, read from or built in ``cache_root`` (``forge.ops.paths.cache_dir()``
    by default)."""
    root = resolve_data_dir(data_dir)
    chosen = val_subset(examples, data_dir=root)
    count = chat_token_counter(tokenizer)
    prompts = [
        fit_prompt(example, prompt_version, count, data_dir=root).prompt for example in chosen
    ]
    gold = split_gold("val", data_dir=root, cache_root=cache_root).results
    return ValEvalCallback(
        chosen,
        prompts,
        gold,
        pool,
        every=every,
        max_new_tokens=max_new_tokens,
        model_ref=model_ref,
        record=record,
        backend_for=backend_for,
    )
