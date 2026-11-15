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
- The evaluator calls ``asyncio.run`` for every query, which raises inside a running
  event loop such as a notebook cell's. So the items run in a worker thread under one
  overall deadline, and each query keeps the vendored per-query timeout of 60 seconds.

The evaluator opens its own connections, read-only and with the executor's authorizer
and limits (``third_party/test_suite_eval/CHANGES.md``), so SQL that got past these
checks still can't write.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from sqlparse.exceptions import SQLParseError
from test_suite_eval import exec_eval
from test_suite_eval.parse import remove_distinct

from forge.core.types import ErrorKind
from forge.data.spider import db_path, resolve_data_dir
from forge.sql.executor import SqliteExecutor, strip_leading

DEFAULT_TIMEOUT_S = 3600.0
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


class OfficialEvalTimeout(TimeoutError):
    """The items took longer than the overall deadline."""


def official_exec_match(
    items: Sequence[tuple[str, str | None, str]],
    *,
    data_dir: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[bool]:
    """The official verdict for each ``(db_id, pred_sql, gold_sql)`` item, in order.

    Raises ``OfficialEvalTimeout`` when all of them take longer than ``timeout_s``; the
    worker then stops after the item it is on. A missing database raises
    ``DataNotFound``, and a failing gold query ``GoldQueryFailed``.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")
    root = resolve_data_dir(data_dir).resolve()
    jobs = [MatchItem(*item) for item in items]
    stop = threading.Event()
    outcome: list[list[bool] | BaseException] = []

    def work() -> None:
        try:
            outcome.append(_match_all(jobs, root, stop))
        except BaseException as exc:  # re-raised in the caller's thread below
            outcome.append(exc)

    # A daemon, so a worker still finishing its last query can't hold up interpreter exit.
    worker = threading.Thread(target=work, name="forge-official-eval", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        stop.set()
        raise OfficialEvalTimeout(
            f"official evaluation of {len(jobs)} items took longer than {timeout_s:g} s"
        )
    (result,) = outcome
    if isinstance(result, BaseException):
        raise result
    return result


def _match_all(items: Sequence[MatchItem], root: Path, stop: threading.Event) -> list[bool]:
    verdicts: list[bool] = []
    # Made here because sqlite3 connections belong to the thread that opened them.
    with SqliteExecutor(root) as executor:
        for item in items:
            if stop.is_set():
                break
            verdicts.append(_match(executor, root, item))
    return verdicts


def _match(executor: SqliteExecutor, root: Path, item: MatchItem) -> bool:
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
