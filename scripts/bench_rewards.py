"""Time GRPO's rewards on a batch of 6 prompts with 6 replies each, as TRL calls them.

Usage: uv run python scripts/bench_rewards.py [--data-dir DIR] [--batches N] [--workers N]

The data directory defaults to the fixture, tests/fixtures/spider_mini. Each batch takes 6
of its train questions, spread over the split, and gives each the 6 replies of
``replies``: its gold query in a code block and after a line of prose, its first row
alone, a misspelt keyword, a constant and no SQL at all. Batches differ, so no batch
reuses the verdicts of the one before. Gold results come from the gold cache, built first
if missing, so they cost nothing per batch, as in training. A batch is timed from the
first reward function's call to the last one's return, with the pool already started.
``tests/integration/test_reward_pool.py`` times the same batches against FG-503's budget.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from forge.core.types import Example, ExecResult
from forge.data.spider import FAILING_GOLD_IDS
from forge.data.splits import load_train
from forge.eval.gold_cache import split_gold
from forge.train.rewards import Rewards, available_cpus, reward_pool, reward_workers

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "spider_mini"
PROMPTS = 6
GENERATIONS = 6

Scores = tuple[float, float, float, float]


def replies(example: Example) -> list[str]:
    """The ``GENERATIONS`` replies a policy might give ``example``, right ones and wrong."""
    bare = example.gold_sql.strip().removesuffix(";")
    return [
        f"```sql\n{bare}\n```",
        f"The query below answers it.\n```sql\n{bare}\n```",
        f"```sql\nSELECT * FROM ({bare}) LIMIT 1\n```",
        # Every Spider train gold query starts with an upper-case SELECT.
        f"```sql\n{bare.replace('SELECT', 'SELEC', 1)}\n```",
        "```sql\nSELECT 1\n```",
        "I can't answer that from the schema.",
    ]


def batch(examples: Sequence[Example], index: int = 0) -> list[tuple[Example, str]]:
    """Batch ``index``: ``PROMPTS`` questions spread over ``examples``, each with its
    replies, one after another as GRPO groups them."""
    step = max(1, len(examples) // PROMPTS)
    chosen = [examples[(i * step + index) % len(examples)] for i in range(PROMPTS)]
    return [(example, reply) for example in chosen for reply in replies(example)]


def score(rewards: Rewards, items: Sequence[tuple[Example, str]]) -> list[Scores]:
    """Each reply's four rewards, the functions called one after another as TRL calls
    them, with conversational completions and the dataset's columns as keywords."""
    columns: Mapping[str, list[str]] = {
        "example_id": [example.id for example, _ in items],
        "db_id": [example.db_id for example, _ in items],
        "gold_sql": [example.gold_sql for example, _ in items],
        "hardness": [example.hardness for example, _ in items],
    }
    prompts = [[{"role": "user", "content": example.question}] for example, _ in items]
    completions = [[{"role": "assistant", "content": reply}] for _, reply in items]
    outputs = [
        fn(prompts=prompts, completions=completions, completion_ids=None, **columns)
        for fn in rewards.functions
    ]
    return [(e, v, f, g) for e, v, f, g in zip(*outputs, strict=True)]


def timed(rewards: Rewards, items: Sequence[tuple[Example, str]]) -> float:
    """Seconds to score ``items``."""
    start = time.perf_counter()
    score(rewards, items)
    return time.perf_counter() - start


def train_questions(data_dir: Path) -> list[Example]:
    """The train questions the rewards can score: all but the known failing gold."""
    return [example for example in load_train(data_dir) if example.id not in FAILING_GOLD_IDS]


def train_gold(data_dir: Path) -> dict[str, ExecResult]:
    """Train's gold results from the gold cache, which this builds when it is missing."""
    return split_gold("train", data_dir=data_dir).results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=FIXTURE)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument(
        "--workers", type=int, default=None, help=f"default {reward_workers()}, one a CPU"
    )
    args = parser.parse_args()

    examples = train_questions(args.data_dir)
    gold = train_gold(args.data_dir)
    workers = reward_workers() if args.workers is None else args.workers
    start = time.perf_counter()
    with reward_pool(args.data_dir, workers=workers) as pool:
        up = time.perf_counter() - start
        print(f"pool of {workers} workers on {available_cpus()} CPUs up in {up:.2f} s")
        rewards = Rewards(gold, pool)
        seconds = [timed(rewards, batch(examples, index)) for index in range(args.batches)]
    size = PROMPTS * GENERATIONS
    print(
        f"{args.batches} batches of {size} completions: median {statistics.median(seconds):.3f} s, "
        f"worst {max(seconds):.3f} s, first {seconds[0]:.3f} s"
    )


if __name__ == "__main__":
    main()
