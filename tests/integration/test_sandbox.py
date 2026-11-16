"""The sandbox pool: untrusted SQL in worker processes that are killed at a hard deadline."""

from __future__ import annotations

import logging
import multiprocessing
import os
import resource
import sys
import time
from pathlib import Path

import pytest

from forge.core.types import ExecResult
from forge.sql import sandbox
from forge.sql.sandbox import (
    KILL_GRACE_MS,
    OUT_OF_MEMORY,
    SandboxError,
    SandboxOutOfMemory,
    SandboxPool,
    SandboxTimeout,
    SandboxWorkerDied,
)
from tests import sandbox_jobs as jobs
from tests.sandbox_jobs import review_query

DB = "department_management"
RUNAWAY = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r"
# A billion rows from the 10-row head table: endless, but it loops, so SQLite checks for
# an interrupt between rows.
CARTESIAN = "SELECT count(*) FROM " + ", ".join(f"head AS t{i}" for i in range(9))
# 4,000-character strings aggregated over an endless CTE: SQLite's memory grows until the
# deadline, as no SQLite limit bounds it (ADR-0014).
HOARD = (
    "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) "
    "SELECT json_group_array(hex(zeroblob(2000))) FROM r"
)


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
    # The hard deadline on its own: with FG-191's limits these run for seconds in process,
    # against 300 ms and 2,000 ms timeouts (ADR-0014).
    monkeypatch.setattr(sandbox, "_run_job", jobs.run_with_fg191_limits)
    caplog.set_level(logging.DEBUG, logger="forge.sql.sandbox")
    with SandboxPool(2) as pool:
        # Both workers up and ready first, so no start-up lands inside the timed run.
        pool.start()
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
        # Both workers ready first, so the timed runs below don't pay for a start-up.
        pool.start()
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
        with pytest.raises(SandboxWorkerDied, match="died with exit code 3"):
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
def test_pool_reports_a_result_it_cannot_read_back(spider_mini: Path) -> None:
    with SandboxPool(1) as pool:
        pid = pool.call(jobs.worker_pid, timeout_s=5)
        # The result pickles in the worker but can't be rebuilt here. That used to raise a
        # bare TypeError and kill a healthy worker; the whole reply was read, so only the
        # reply is lost.
        with pytest.raises(
            SandboxError,
            match=r"reply could not be read back: TypeError: .*missing 1 required positional",
        ) as raised:
            pool.call(jobs.returns_unloadable, timeout_s=5)
        assert not isinstance(raised.value, SandboxWorkerDied)
        assert pool.call(jobs.worker_pid, timeout_s=5) == pid
        assert pool.restarts == 0


@pytest.mark.phase(1)
def test_pool_starts_workers_for_work_and_stops_them_on_close(spider_mini: Path) -> None:
    with pytest.raises(ValueError, match="workers must be at least 1"):
        SandboxPool(0)
    pool = SandboxPool(4)
    assert pool.run_many([]) == []
    assert multiprocessing.active_children() == []
    # One query needs one worker; start brings up the rest, and waits until they're ready.
    assert pool.run(DB, "SELECT 1").rows == [(1,)]
    assert len(multiprocessing.active_children()) == 1
    pool.start()
    assert len(multiprocessing.active_children()) == 4
    assert all(worker is not None and worker.ready for worker in pool._slots)
    # A worker that can't start fails start itself, not a later job.
    with (
        SandboxPool(1, imports=("forge_no_such_module",)) as broken,
        pytest.raises(SandboxError, match="exited while starting, with exit code 1"),
    ):
        broken.start()
    assert len(multiprocessing.active_children()) == 4
    # Round trips stay cheap: about 0.2 ms each on the laptop; the bound is for slow CI.
    start = time.perf_counter()
    for _ in range(100):
        pool.run(DB, "SELECT count(*) FROM head")
    assert (time.perf_counter() - start) * 1000 / 100 < 25

    pool.close()
    assert multiprocessing.active_children() == []
    with pytest.raises(RuntimeError, match="the sandbox pool is closed"):
        pool.run(DB, "SELECT 1")


@pytest.mark.phase(5)
def test_pool_replaces_a_worker_that_runs_out_of_memory(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="memory_limit must be positive or None"):
        SandboxPool(1, memory_limit=0)
    with SandboxPool(1) as pool:
        # SQLite's own heap cap gives a real SQLITE_NOMEM on macOS as well as Linux. The
        # query after it runs on the fresh worker, so a batch loses only the hoarder.
        with monkeypatch.context() as patched:
            patched.setattr(sandbox, "_run_job", jobs.run_under_heap_limit)
            hoarded, after = pool.run_many(
                [(DB, HOARD), (DB, "SELECT count(*) FROM head")], timeout_ms=300
            )
        assert (hoarded.ok, hoarded.error_kind, hoarded.error) == (False, "other", OUT_OF_MEMORY)
        assert after.rows == [(10,)]
        assert pool.restarts == 1

        with pytest.raises(SandboxOutOfMemory, match="out of memory") as raised:
            pool.call(jobs.raise_memory_error, timeout_s=5)
        # Official EX counts a worker's death as no match, and so this.
        assert isinstance(raised.value, SandboxWorkerDied)
        assert pool.restarts == 2
        assert pool.run(DB, "SELECT count(*) FROM head").rows == [(10,)]


@pytest.mark.phase(5)
@pytest.mark.skipif(sys.platform != "linux", reason="only Linux enforces RLIMIT_AS")
def test_worker_memory_cap_on_linux(spider_mini: Path) -> None:
    cap = 64 * 2**20
    with SandboxPool(1, memory_limit=cap) as pool:
        soft, hard = pool.call(jobs.address_space_limit, timeout_s=5)
        mapped = pool.call(jobs.address_space, timeout_s=5)
        assert soft == hard
        assert 0 < soft - mapped <= cap
        # Without the cap this query grows the worker by hundreds of MiB in 100 ms.
        start = time.perf_counter()
        (hoarded,) = pool.run_many([(DB, HOARD)], timeout_ms=500)
        assert (hoarded.ok, hoarded.error_kind, hoarded.error) == (False, "other", OUT_OF_MEMORY)
        assert time.perf_counter() - start < 5
        assert pool.restarts == 1
        assert pool.run(DB, "SELECT count(*) FROM head").rows == [(10,)]
        # The fresh worker has a cap of its own.
        soft, _ = pool.call(jobs.address_space_limit, timeout_s=5)
        assert 0 < soft - pool.call(jobs.address_space, timeout_s=5) <= cap
    with SandboxPool(1, memory_limit=None) as pool:
        assert pool.call(jobs.address_space_limit, timeout_s=5) == resource.getrlimit(
            resource.RLIMIT_AS
        )


@pytest.mark.phase(5)
@pytest.mark.skipif(sys.platform == "linux", reason="Linux caps the worker instead")
def test_worker_memory_is_not_capped_where_rlimit_as_is_not_enforced(spider_mini: Path) -> None:
    # macOS accepts RLIMIT_AS but ignores it, so the worker keeps its parent's limit and
    # the deadline's kill is the bound.
    with SandboxPool(1) as pool:
        assert pool.call(jobs.address_space_limit, timeout_s=5) == resource.getrlimit(
            resource.RLIMIT_AS
        )
