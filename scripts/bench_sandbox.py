"""Measure what the sandbox pool costs: per-query overhead and worker restart time.

Usage: FORGE_DATA_DIR=tests/fixtures/spider_mini uv run python scripts/bench_sandbox.py

The queries are every gold query of the data directory, run in process and then through
a pool. Overhead is the pool's time per query less the in-process time; restart time is
from a kill to a fresh worker's first answer. ADR-0014 quotes the output.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import time

from forge.data.spider import FAILING_GOLD_IDS, load_raw
from forge.sql.executor import SqliteExecutor
from forge.sql.sandbox import SandboxPool, SandboxTimeout


def _hold(executor: SqliteExecutor, seconds: float) -> None:
    time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5, help="passes over the gold queries")
    parser.add_argument("--restarts", type=int, default=10, help="kills to time")
    args = parser.parse_args()

    items = [(e.db_id, e.gold_sql) for e in load_raw() if e.id not in FAILING_GOLD_IDS]
    items *= args.rounds
    print(f"{len(items)} queries ({args.rounds} passes over {len(items) // args.rounds} golds)")

    with SqliteExecutor() as executor:
        start = time.perf_counter()
        for db_id, sql in items:
            executor.run(db_id, sql)
        inline_ms = (time.perf_counter() - start) * 1000 / len(items)
    print(f"in process, one at a time:      {inline_ms:.3f} ms per query")

    with SandboxPool(args.workers) as pool:
        start = time.perf_counter()
        pool.run(*items[0])
        print(f"pool start, first query:       {(time.perf_counter() - start) * 1000:.0f} ms")
        start = time.perf_counter()
        for db_id, sql in items:
            pool.run(db_id, sql)
        single_ms = (time.perf_counter() - start) * 1000 / len(items)
        print(
            f"pool.run, one at a time:       {single_ms:.3f} ms per query "
            f"({single_ms - inline_ms:+.3f} ms overhead)"
        )
        # A pass to start every worker, so the timed one doesn't wait on start-ups.
        pool.run_many(items)
        start = time.perf_counter()
        pool.run_many(items)
        batch_ms = (time.perf_counter() - start) * 1000 / len(items)
        print(f"pool.run_many, {args.workers} workers:     {batch_ms:.3f} ms per query (wall)")

        restarts = []
        for _ in range(args.restarts):
            with contextlib.suppress(SandboxTimeout):
                pool.call(_hold, 10.0, timeout_s=0.05)
            killed = time.perf_counter()
            pool.run(*items[0])
            restarts.append((time.perf_counter() - killed) * 1000)
        print(
            f"restart, kill to next answer:  {statistics.median(restarts):.0f} ms median, "
            f"{max(restarts):.0f} ms max of {len(restarts)}"
        )


if __name__ == "__main__":
    main()
