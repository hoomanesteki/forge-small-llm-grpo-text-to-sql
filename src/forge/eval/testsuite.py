"""Official execution match: the vendored Spider test-suite evaluator, called safely.

Reported accuracy comes from ``official_exec_match`` (ADR-0008). For each item it calls
``test_suite_eval.exec_eval.eval_exec_match`` directly, with ``plug_value``,
``keep_distinct`` and the progress bar off, the evaluator's recommended execution
setting. It skips ``evaluation.evaluate()``, which first replaces every ``value`` in the
prediction with ``1`` and so breaks columns named ``value`` (upstream issue #21). The
evaluator compares on every ``.sqlite`` file in the database's folder: classical EX
with Spider's own databases, test-suite EX once the distilled databases sit beside them.

Around that call:

- A prediction that is ``None`` (the reply had no SQL), or that ``forge.sql.executor``
  reports as ``empty_sql`` or ``denied``, is not a match and never reaches the
  evaluator. Upstream an empty prediction crashes ``remove_distinct``, and a
  comment-only one, or a no-op such as ``DROP TABLE IF EXISTS`` on a missing table, runs
  to an empty result that matches any empty gold result (upstream issue #33). Both
  refusals happen before SQLite computes a row, so the executor's look is cut short
  after ``VET_TIMEOUT_MS`` and only its verdict is kept.
- The evaluator rewrites a prediction before running it: ``postprocess`` turns ``> =``
  into ``>=``, ``remove_distinct`` drops every ``DISTINCT`` and keeps the first statement
  sqlparse finds, and ``replace_cur_year`` turns ``YEAR(CURDATE())`` into ``2020``. So
  the string it will run is built here and vetted too: dropping ``DISTINCT`` turns
  ``DISTINCT ATTACH ...``, a syntax error, into an ``ATTACH``.
- What SQLite skips before the first statement (whitespace, comments, semicolons) is
  stripped first. Otherwise the first statement sqlparse finds in ``; SELECT ...`` is the
  bare ``;``, which runs to no rows and matches any empty gold result.
- A prediction too big for sqlparse's grouping limits (``SQLParseError`` from
  ``remove_distinct``) is not a match: the evaluator can't read it.
- A gold query that fails raises ``GoldQueryFailed``. The evaluator asserts that gold
  runs, and a broken gold is a data problem, not a wrong prediction; callers skip the
  known ones (``forge.data.spider.FAILING_GOLD_IDS``).
- Each item, the checks above and the evaluator's call alike, runs in a
  ``forge.sql.sandbox.SandboxPool`` worker that is killed when the item takes longer
  than ``item_timeout_s``: the evaluator's own timeout can't stop work inside one row,
  or its column-permutation search in Python. A killed item is not a match. The
  evaluator calls ``asyncio.run`` for every query, which raises inside a running event
  loop such as a notebook cell's; a worker has none.

The evaluator opens its own connections, read-only and with the executor's authorizer
and limits (``third_party/test_suite_eval/CHANGES.md``), so SQL that got past these
checks still can't write.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple

from sqlparse.exceptions import SQLParseError
from test_suite_eval import exec_eval
from test_suite_eval.parse import remove_distinct

from forge.core.types import ErrorKind
from forge.data.spider import db_path, resolve_data_dir
from forge.sql.executor import SqliteExecutor, strip_leading
from forge.sql.sandbox import SandboxPool, SandboxTimeout

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 3600.0
# The evaluator's own timeout for one query (exec_eval.TIMEOUT); a prediction that runs
# that long is no match whichever deadline stops it.
DEFAULT_ITEM_TIMEOUT_S = 60.0
VET_TIMEOUT_MS = 100
REFUSED: frozenset[ErrorKind] = frozenset({"empty_sql", "denied"})


class MatchItem(NamedTuple):
    """One prediction to score. ``pred_sql`` is ``None`` when the reply had no SQL."""

    db_id: str
    pred_sql: str | None
    gold_sql: str


class GoldQueryFailed(ValueError):
    """A gold query raised on its database, so no prediction can be scored against it."""

    def __init__(self, db_id: str, gold_sql: str) -> None:
        super().__init__(f"gold query fails on {db_id}: {gold_sql}")
        self.db_id = db_id
        self.gold_sql = gold_sql

    def __reduce__(self) -> tuple[type[GoldQueryFailed], tuple[str, str], dict[str, object]]:
        # Rebuilt from its fields, notes included, when it comes back from a worker.
        return (GoldQueryFailed, (self.db_id, self.gold_sql), self.__dict__)


class OfficialEvalTimeout(TimeoutError):
    """The items took longer than the overall deadline."""


def official_exec_match(
    items: Sequence[tuple[str, str | None, str]],
    *,
    data_dir: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    item_timeout_s: float = DEFAULT_ITEM_TIMEOUT_S,
    pool: SandboxPool | None = None,
) -> list[bool]:
    """The official verdict for each ``(db_id, pred_sql, gold_sql)`` item, in order.

    Items run one at a time in a worker of ``pool``, or of a one-worker pool made for
    this call; a pool passed in must be on the same data directory. An item still
    running after ``item_timeout_s`` is killed and counts as no match. Raises
    ``OfficialEvalTimeout`` once all of them take longer than ``timeout_s`` (worker
    start-ups aside), killing the item then running. A missing database raises
    ``DataNotFound``, and a failing gold query ``GoldQueryFailed``.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")
    if item_timeout_s <= 0:
        raise ValueError(f"item_timeout_s must be positive, got {item_timeout_s}")
    root = resolve_data_dir(data_dir).resolve()
    if pool is not None and pool.data_dir != root:
        raise ValueError(f"the pool runs on {pool.data_dir}, not on {root}")
    jobs = [MatchItem(*item) for item in items]
    verdicts: list[bool] = []
    with ExitStack() as stack:
        if pool is None:
            # This module loads in the worker before the first item's deadline starts.
            pool = stack.enter_context(SandboxPool(1, data_dir=root, imports=(__name__,)))
        deadline = time.monotonic() + timeout_s
        for number, job in enumerate(jobs):
            left = deadline - time.monotonic()
            if left <= 0:
                raise _overall_timeout(len(jobs), timeout_s)
            try:
                verdicts.append(pool.call(_match, root, job, timeout_s=min(item_timeout_s, left)))
            except SandboxTimeout:
                if left <= item_timeout_s:
                    raise _overall_timeout(len(jobs), timeout_s) from None
                log.debug(
                    "official EX item %d on %s killed after %g s", number, job.db_id, item_timeout_s
                )
                verdicts.append(False)
    return verdicts


def _overall_timeout(items: int, timeout_s: float) -> OfficialEvalTimeout:
    return OfficialEvalTimeout(
        f"official evaluation of {items} items took longer than {timeout_s:g} s"
    )


def _match(executor: SqliteExecutor, root: Path, item: MatchItem) -> bool:
    """One item's verdict; ``executor`` vets the prediction, as the worker's own."""
    if item.pred_sql is None or _refused(executor, item.db_id, item.pred_sql):
        return False
    pred = strip_leading(item.pred_sql)
    try:
        # What eval_exec_match will run, built the way it builds it.
        as_run: str = exec_eval.replace_cur_year(remove_distinct(exec_eval.postprocess(pred)))
    except SQLParseError:
        return False
    if as_run != pred and _refused(executor, item.db_id, as_run):
        return False
    try:
        matched: int = exec_eval.eval_exec_match(
            db=str(db_path(item.db_id, root)),
            p_str=pred,
            g_str=item.gold_sql,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
    except AssertionError as exc:
        # The evaluator's own check that the gold query ran.
        raise GoldQueryFailed(item.db_id, item.gold_sql) from exc
    return matched == 1


def _refused(executor: SqliteExecutor, db_id: str, sql: str) -> bool:
    vetted = executor.run(db_id, sql, timeout_ms=VET_TIMEOUT_MS, max_rows=0)
    return vetted.error_kind in REFUSED
