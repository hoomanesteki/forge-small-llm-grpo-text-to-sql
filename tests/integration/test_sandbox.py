"""The sandbox pool: untrusted SQL in worker processes that are killed at a hard deadline."""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from forge.core.types import ExecResult
from forge.sql import sandbox
from forge.sql.sandbox import KILL_GRACE_MS, SandboxError, SandboxPool, SandboxTimeout
from tests import sandbox_jobs as jobs
from tests.sandbox_jobs import review_query

DB = "department_management"
RUNAWAY = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r"
# A billion rows from the 10-row head table: endless, but it loops, so SQLite checks for
# an interrupt between rows.
CARTESIAN = "SELECT count(*) FROM " + ", ".join(f"head AS t{i}" for i in range(9))


def _timed(
    pool: SandboxPool, items: list[tuple[str, str]], timeout_ms: int
) -> tuple[list[ExecResult], float]:
    start = time.perf_counter()
    results = pool.run_many(items, timeout_ms=timeout_ms)
    return results, (time.perf_counter() - start) * 1000


@pytest.mark.phase(1)
def test_pool_kills_queries_that_overrun_inside_one_row(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The hard deadline on its own: with FG-191's limits these ran 8.7 s and 18.5 s on the
    # laptop in process, against 300 ms and 2,000 ms timeouts.
    monkeypatch.setattr(sandbox, "_run_job", jobs.run_with_fg191_limits)
    caplog.set_level(logging.DEBUG, logger="forge.sql.sandbox")
    with SandboxPool(2) as pool:
        # Warm both workers first: one job would start only one, and the second worker's
        # start-up would then land inside the timed run below.
        assert [r.rows for r in pool.run_many([(DB, "SELECT 1")] * 2)] == [[(1,)], [(1,)]]
        results, elapsed_ms = _timed(pool, [(DB, review_query(100)), (DB, review_query(200))], 300)

        for result in results:
            assert (result.ok, result.error_kind) == (False, "timeout"), result
            assert result.error is not None
            assert result.error.startswith("killed after "), result.error
            assert result.error.endswith(" ms (timeout 300 ms)")
            assert 300 + KILL_GRACE_MS <= result.ms < 300 + 1000
        assert elapsed_ms < 300 + 1000
        assert pool.restarts == 2

        # Fresh workers take the next queries.
        after = pool.run_many([(DB, "SELECT count(*) FROM head")] * 2)
        assert [r.rows for r in after] == [[(10,)], [(10,)]]
    assert "2 restarts so far" in caplog.text
    assert "closed a sandbox pool of 2 workers after 2 restarts" in caplog.text


@pytest.mark.phase(1)
def test_pool_stops_runaway_and_cartesian_queries(spider_mini: Path) -> None:
    with SandboxPool(2) as pool:
        # Warm both workers, so the timed runs below don't pay for a worker start-up.
        assert [r.rows for r in pool.run_many([(DB, "SELECT 1")] * 2)] == [[(1,)], [(1,)]]
        # With today's limits the review's queries stop before they do any work.
        reviewed = [(DB, review_query(100)), (DB, review_query(200))]
        results, elapsed_ms = _timed(pool, reviewed, 300)
        too_big = ("other", "string or blob too big")
        assert [(r.error_kind, r.error) for r in results] == [too_big] * 2
        assert elapsed_ms < 300

        results, elapsed_ms = _timed(pool, [(DB, CARTESIAN), (DB, RUNAWAY)], 300)
        assert [(r.ok, r.error_kind) for r in results] == [(False, "timeout")] * 2
        assert all(300 <= r.ms < 300 + 1000 for r in results), [r.ms for r in results]
        assert elapsed_ms < 300 + 1000
        # Both loop, so the workers' own deadline stopped them and nothing was killed.
        assert pool.restarts == 0
        assert pool.run(DB, "SELECT count(*) FROM head").rows == [(10,)]


@pytest.mark.phase(1)
def test_pool_replaces_a_worker_that_dies(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SandboxPool(1) as pool:
        with pytest.raises(SandboxError, match="died with exit code 3"):
            pool.call(jobs.exit_with_three, timeout_s=5)
        assert pool.restarts == 1

        with monkeypatch.context() as patched:
            patched.setattr(sandbox, "_run_job", jobs.exit_with_three)
            died = pool.run(DB, "SELECT 1")
        assert (died.ok, died.error_kind, died.error) == (
            False,
            "other",
            "sandbox worker died with exit code 3",
        )
        assert pool.run(DB, "SELECT count(*) FROM head").rows == [(10,)]
        assert pool.restarts == 2


@pytest.mark.phase(1)
def test_pool_calls_run_in_a_worker(spider_mini: Path) -> None:
    with SandboxPool(1) as pool:
        pid = pool.call(jobs.worker_pid, timeout_s=5)
        assert pid != os.getpid()
        assert pool.call(jobs.worker_pid, timeout_s=5) == pid  # the worker stays up

        # A job's exception comes back with the worker's traceback.
        with pytest.raises(ValueError) as raised:
            pool.call(jobs.fail, "no good", timeout_s=5)
        assert str(raised.value) == "no good"
        (note,) = raised.value.__notes__
        assert note.startswith("in a sandbox worker:\n")
        assert "in fail\n    raise ValueError(message)" in note
        with pytest.raises(SandboxError, match="Unpicklable: one and two"):
            pool.call(jobs.fail_unpicklably, timeout_s=5)
        # A job that can't reach a worker fails here, and the worker is still fine.
        # Python 3.11 says "Can't pickle local object", 3.12 "Can't get local object".
        with pytest.raises(AttributeError, match=r"Can't (get|pickle) local object"):
            pool.call(lambda executor: None, timeout_s=5)
        assert pool.call(jobs.worker_pid, timeout_s=5) == pid

        start = time.monotonic()
        with pytest.raises(SandboxTimeout, match=r"killed after \d+ ms \(timeout 0\.2 s\)"):
            pool.call(jobs.sleep, 10.0, timeout_s=0.2)
        assert time.monotonic() - start < 1.0
        assert pool.call(jobs.worker_pid, timeout_s=5) != pid
        assert pool.restarts == 1
        with pytest.raises(ValueError, match="timeout_s must be positive"):
            pool.call(jobs.worker_pid, timeout_s=0)


@pytest.mark.phase(1)
def test_pool_reports_a_result_it_cannot_send_back(spider_mini: Path) -> None:
    with SandboxPool(1) as pool:
        pid = pool.call(jobs.worker_pid, timeout_s=5)
        # A job whose return value can't be pickled used to crash the worker, which came
        # back as "died with exit code 1". Now the pickling error is reported by name.
        with pytest.raises(SandboxError, match="result could not be sent back: Unsendable"):
            pool.call(jobs.returns_unsendable, timeout_s=5)
        # The worker stayed up: same pid, nothing restarted.
        assert pool.call(jobs.worker_pid, timeout_s=5) == pid
        assert pool.restarts == 0


@pytest.mark.phase(1)
def test_pool_starts_workers_for_work_and_stops_them_on_close(spider_mini: Path) -> None:
    with pytest.raises(ValueError, match="workers must be at least 1"):
        SandboxPool(0)
    pool = SandboxPool(4)
    assert pool.run_many([]) == []
    assert multiprocessing.active_children() == []
    # One query needs one worker.
    assert pool.run(DB, "SELECT 1").rows == [(1,)]
    assert len(multiprocessing.active_children()) == 1
    # Round trips stay cheap: about 0.2 ms each on the laptop; the bound is for slow CI.
    start = time.perf_counter()
    for _ in range(100):
        pool.run(DB, "SELECT count(*) FROM head")
    assert (time.perf_counter() - start) * 1000 / 100 < 25

    pool.close()
    assert multiprocessing.active_children() == []
    with pytest.raises(RuntimeError, match="the sandbox pool is closed"):
        pool.run(DB, "SELECT 1")
