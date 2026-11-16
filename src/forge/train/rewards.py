"""GRPO's rewards: pay for SQL that answers the question and charge for the known
shortcuts, so the model learns the skill rather than the checker (FG-501).

``Rewards`` gives TRL four reward functions, in this order, and the weights
``GRPOConfig(reward_weights=...)`` sums them by, from the config's ``rewards`` section:

- ``execution_reward``: 1 when the query's rows match its gold query's under
  ``forge.sql.compare.results_match``, the comparator I5 measures against the official
  evaluator (ADR-0008), else 0. Weight ``rewards.execution``, 1.0 to start.
- ``validity_reward``: 1 when the query runs without error, else 0. Weight
  ``rewards.validity``, 0.1.
- ``format_reward``: 1 when the reply is one ```` ```sql ```` block holding a statement, with
  only whitespace around it, else 0. Weight ``rewards.format``, 0.05.
- ``guard_penalty``: the sum of ``rewards.guards``' penalties for the guards that fire,
  already in reward units, so its weight is 1:

  - ``non_select``, -1.0: the executor denies the query, as it denies anything but one
    ``SELECT`` or ``VALUES``; or, among several statements, one starts with a statement
    keyword the executor denies (``statement_keyword``), such as ``DELETE``. The pieces
    are cut and read as the executor reads them, so a write is charged even where sqlglot
    can't read the rest, as with a block comment left open.
  - ``multiple_statements``, -0.5: more than one statement, counted as the executor counts
    them (``statement_count``), so it fires exactly when the executor refuses the SQL as
    more than one statement.
  - ``constant_query``, -0.5: the query answers with constants where the gold query reads
    the database. It does when sqlglot finds no table in it, or when some of the rows it
    returns are constants: those of a ``SELECT`` it returns the rows of (either side of a
    ``UNION`` or ``INTERSECT``, the left of an ``EXCEPT``), or a row of a bare ``VALUES``,
    whose every column, subqueries included, names no column or ``*`` and calls no window
    function and no aggregate but ``min``, ``max`` or ``avg``, which return the constant
    they are given. So ``SELECT 3 FROM professor LIMIT 1`` costs what ``SELECT 3`` does. A
    common table expression, a subquery and a table-valued function such as ``json_each``
    are not tables, and a column of one counts as read from the database, whatever it
    holds. It is judged on the SQL alone, before the executor's result: ``SELECT 3`` costs
    the same whether or not 3 is the answer.
  - ``comment_payload``, -0.2: a comment over 40 characters, counted between its ``--``
    or ``/* */`` and without the whitespace at its ends. Comments are read as SQLite reads
    them (``forge.sql.executor.comments``), because sqlglot drops a comment that no token
    follows and can't read a block comment left open, which SQLite runs as a comment to the
    end.

The first three return 0 or 1 and the weights scale them, so TRL's
``rewards/<name>/mean`` reads as the share of completions that match, run or keep the
format; with the starting weights, the gold query in a code block scores 1.15 in all.
Execution, validity and format pay nothing once any guard fires, so a reply that sets one
off scores its penalties alone. They pay nothing either for a query that ran but that
sqlglot can't read, whose tables no one could check; such a query costs nothing. A reply in
which ``extract_sql`` finds no SQL scores 0 on all four.

Each function takes TRL's arguments, ``(prompts, completions, completion_ids=None,
**kwargs)``: ``kwargs`` holds the dataset's ``example_id``, ``db_id`` and ``gold_sql``
columns, one value a completion, which GRPO passes on with ``remove_unused_columns=False``.
A completion is the reply's text or, in a conversational dataset, a list of messages whose
first holds it as ``content``. The reply's ``<think>...</think>`` blocks are dropped before
anything reads it. ``gold`` maps example ids to their gold queries' results, from
``forge.eval.gold_cache.split_gold("train")``, which reads the file ``forge data
cache-gold`` builds, or builds it on first use.

Model SQL is untrusted, so it runs only in the ``SandboxPool`` (ADR-0014), with the eval
runner's timeout, 2 s a query, and row cap, which the gold cache's results allow for. A run
keeps one pool for all four functions, ``reward_pool()``: one worker a CPU, at most
``MAX_REWARD_WORKERS``, started before the first batch so no batch waits on start-ups, and
each worker's memory capped on Linux. A batch's queries run across its workers at once.
TRL calls the four functions one after another on the same batch, so the batch's queries
run once and the other three reuse its verdicts. Given the gold results the rewards are
pure (I3): the same completion of the same example gets the same rewards, unless its query
ends so near its deadline that one run times out and another doesn't.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import traverse_scope

from forge.core.types import ExecResult
from forge.eval.config import PRED_TIMEOUT_MS
from forge.eval.gold_cache import MAX_ROWS
from forge.sql.compare import order_matters, results_match
from forge.sql.executor import (
    QUERY_KEYWORDS,
    comments,
    split_statements,
    statement_count,
    statement_keyword,
)
from forge.sql.extract import drop_thinking, extract_sql
from forge.sql.sandbox import SandboxPool
from forge.train.backends import RewardFunc
from forge.train.config import RewardsConfig

log = logging.getLogger(__name__)

Guard = Literal["non_select", "multiple_statements", "constant_query", "comment_payload"]
# The order penalties are added in, so a sum comes out as the same float every time.
GUARDS: tuple[Guard, ...] = (
    "non_select",
    "multiple_statements",
    "constant_query",
    "comment_payload",
)
COMMENT_PAYLOAD_CHARS = 40
# Aggregates that return the constant they are given; any other one counts rows, as
# count(1) and sum(1) do.
_CONSTANT_AGGREGATES = (exp.Min, exp.Max, exp.Avg)
# The dataset columns the rewards read from TRL's keyword arguments.
COLUMNS = ("example_id", "db_id", "gold_sql")
# Kaggle's T4 machines have 4 CPUs. Rewards run between the trainer's steps, while it waits
# on them, so a worker on each CPU takes none from training.
MAX_REWARD_WORKERS = 4

_SQL_REPLY = re.compile(r"\s*```sql\b(?P<body>.*?)```\s*", re.DOTALL | re.IGNORECASE)

_BatchKey = tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class Verdict:
    """How the rewards read one completion."""

    sql: str | None  # what extract_sql found, None when it found nothing
    guards: tuple[Guard, ...] = ()  # those that fired, in GUARDS order
    valid: bool = False  # ran without error, and nothing withheld credit
    matches: bool = False  # its rows are its gold query's, and nothing withheld credit
    formatted: bool = False  # one sql block alone, and nothing withheld credit


class Rewards:
    """The reward functions of one GRPO run, as the module docstring describes them.

    ``gold`` holds the result of each dataset example's gold query by example id, and
    ``pool`` runs the completions' queries, a run's ``reward_pool()``; the caller owns the
    pool and closes it.
    """

    def __init__(
        self,
        gold: Mapping[str, ExecResult],
        pool: SandboxPool,
        config: RewardsConfig | None = None,
    ) -> None:
        self.gold = gold
        self.pool = pool
        self.config = RewardsConfig() if config is None else config
        guards = self.config.guards
        self._penalties: dict[Guard, float] = {
            "non_select": guards.non_select,
            "multiple_statements": guards.multiple_statements,
            "constant_query": guards.constant_query,
            "comment_payload": guards.comment_payload,
        }
        self._last: tuple[_BatchKey, tuple[Verdict, ...]] | None = None

    @property
    def functions(self) -> list[RewardFunc]:
        """The four reward functions for ``GRPOTrainer(reward_funcs=...)``."""
        return [self.execution_reward, self.validity_reward, self.format_reward, self.guard_penalty]

    @property
    def weights(self) -> list[float]:
        """``GRPOConfig.reward_weights``, in the order of ``functions``."""
        config = self.config
        return [config.execution, config.validity, config.format, 1.0]

    def execution_reward(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        completion_ids: object = None,
        **kwargs: object,
    ) -> list[float]:
        """1 for each completion whose query returns its gold query's rows, else 0."""
        return [float(v.matches) for v in self.judge(prompts, completions, kwargs)]

    def validity_reward(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        completion_ids: object = None,
        **kwargs: object,
    ) -> list[float]:
        """1 for each completion whose query runs without error, else 0."""
        return [float(v.valid) for v in self.judge(prompts, completions, kwargs)]

    def format_reward(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        completion_ids: object = None,
        **kwargs: object,
    ) -> list[float]:
        """1 for each reply that is one sql code block and nothing else, else 0."""
        return [float(v.formatted) for v in self.judge(prompts, completions, kwargs)]

    def guard_penalty(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        completion_ids: object = None,
        **kwargs: object,
    ) -> list[float]:
        """The sum of the penalties of the guards each completion sets off; 0 for none."""
        return [
            sum((self._penalties[guard] for guard in v.guards), 0.0)
            for v in self.judge(prompts, completions, kwargs)
        ]

    def judge(
        self,
        prompts: Sequence[object],
        completions: Sequence[object],
        columns: Mapping[str, object],
    ) -> list[Verdict]:
        """Each completion's verdict, from TRL's arguments; ``columns`` are its keywords.

        Raises ``ValueError`` when a column is missing or doesn't hold one string a
        completion, or an example has no gold result, ``TypeError`` for a completion that
        is neither text nor a list of messages, and ``DataNotFound`` from the pool for a
        database it doesn't have.
        """
        texts = [_reply_text(completion) for completion in completions]
        if len(prompts) != len(texts):
            raise ValueError(f"{len(prompts)} prompts came with {len(texts)} completions")
        example_ids, db_ids, gold_sqls = (_column(columns, name, len(texts)) for name in COLUMNS)
        key = tuple(zip(texts, example_ids, db_ids, gold_sqls, strict=True))
        if self._last is not None and self._last[0] == key:
            return list(self._last[1])
        golds = [self._gold(example_id) for example_id in example_ids]
        replies = [drop_thinking(text) for text in texts]
        sqls = [extract_sql(reply) for reply in replies]
        runnable = [(i, sql) for i, sql in enumerate(sqls) if sql is not None]
        ran = self.pool.run_many(
            [(db_ids[i], sql) for i, sql in runnable], timeout_ms=PRED_TIMEOUT_MS, max_rows=MAX_ROWS
        )
        results = dict(zip((i for i, _ in runnable), ran, strict=True))
        verdicts = [
            Verdict(sql=None)
            if sql is None
            else _verdict(replies[i], sql, results[i], golds[i], gold_sqls[i])
            for i, sql in enumerate(sqls)
        ]
        self._last = (key, tuple(verdicts))
        return verdicts

    def _gold(self, example_id: str) -> ExecResult:
        gold = self.gold.get(example_id)
        if gold is None:
            raise ValueError(
                f"no gold result for {example_id}; the gold results must cover every example "
                "of the dataset (forge.eval.gold_cache.gold_results)"
            )
        return gold


def reward_workers(cpus: int | None = None) -> int:
    """The reward pool's size: ``cpus``, by default the CPUs this process may run on, but
    at most ``MAX_REWARD_WORKERS``."""
    return max(1, min(MAX_REWARD_WORKERS, available_cpus() if cpus is None else cpus))


def available_cpus() -> int:
    """The CPUs this process may run on: on Linux its affinity mask, which a container's
    cpuset narrows and ``os.cpu_count`` doesn't see; elsewhere the machine's count."""
    if sys.platform == "linux":
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def reward_pool(data_dir: Path | None = None, *, workers: int | None = None) -> SandboxPool:
    """The sandbox pool a run's rewards share, ``reward_workers()`` workers by default,
    each one started and ready. The caller closes it, as a ``with`` block does.

    Raises ``SandboxError`` when a worker can't start.
    """
    pool = SandboxPool(reward_workers() if workers is None else workers, data_dir=data_dir)
    try:
        pool.start()
    except BaseException:
        pool.close()
        raise
    return pool


def _verdict(reply: str, sql: str, pred: ExecResult, gold: ExecResult, gold_sql: str) -> Verdict:
    """The verdict on ``sql``, found in ``reply``, whose result is ``pred``."""
    statements = _read(sql)
    count = statement_count(sql)
    fired: set[Guard] = set()
    if count > 1:
        fired.add("multiple_statements")
        if any(map(_denied_keyword, split_statements(sql))):
            fired.add("non_select")
    elif pred.error_kind == "denied":
        fired.add("non_select")
    reads = _reads_data(statements)
    if "non_select" not in fired and reads is False and _gold_reads_data(gold_sql):
        fired.add("constant_query")
    if any(len(text.strip()) > COMMENT_PAYLOAD_CHARS for text in comments(sql)):
        fired.add("comment_payload")
    guards = tuple(guard for guard in GUARDS if guard in fired)
    # A query that ran but whose tables sqlglot couldn't check can't be cleared of being a
    # constant, so it earns nothing, though nothing is charged.
    credit = not guards and not (pred.ok and reads is None)
    return Verdict(
        sql=sql,
        guards=guards,
        valid=credit and pred.ok,
        matches=credit and results_match(gold, pred, order_matters=_order_matters(gold_sql)),
        # A block of nothing but comments holds no query to be in the format.
        formatted=credit and count == 1 and _formatted(reply),
    )


def _reply_text(completion: object) -> str:
    """The text of a completion: itself, or its first message's content."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion and isinstance(completion[0], Mapping):
        content = completion[0].get("content")
        if isinstance(content, str):
            return content
    raise TypeError(
        "a completion is its text or a list of messages with the text as the first one's "
        f"content, not {completion!r:.100}"
    )


def _column(columns: Mapping[str, object], name: str, size: int) -> list[str]:
    """Dataset column ``name`` from TRL's keywords: ``size`` strings."""
    if name not in columns:
        raise ValueError(
            f"the rewards need the dataset's {name} column, which GRPO passes on only with "
            "remove_unused_columns=False"
        )
    values = columns[name]
    wrong = f"the {name} column must hold one string a completion, {size} in all"
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ValueError(wrong)
    strings = [value for value in values if isinstance(value, str)]
    if len(strings) != size or len(values) != size:
        raise ValueError(wrong)
    return strings


def _read(sql: str) -> list[exp.Expr] | None:
    """The statements sqlglot reads in ``sql``, or ``None`` when it can't read them."""
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    # sqlglot runs out of stack at about 47 nested parentheses and raises RecursionError.
    except (SqlglotError, RecursionError):
        return None
    # sqlglot 30 reads a comment beside a ``;`` as an ``exp.Semicolon`` statement, which
    # SQLite doesn't count; ``comments`` reads comments for the payload guard.
    return [
        statement
        for statement in statements
        if statement is not None and not isinstance(statement, exp.Semicolon)
    ]


def _denied_keyword(statement: str) -> bool:
    """Whether the executor would deny ``statement`` by its keyword, as not a query."""
    keyword = statement_keyword(statement)
    return keyword is not None and keyword not in QUERY_KEYWORDS


def _is_query(statement: exp.Expr) -> bool:
    # sqlglot reads a bare VALUES as a node of its own, but the executor runs it as a query.
    return isinstance(statement, exp.Query | exp.Values)


def _reads_a_table(statements: Sequence[exp.Expr] | None) -> bool | None:
    """Whether the queries read a table of the database: ``None`` unless sqlglot read the
    SQL as one or more queries and could follow their scopes.

    A table is a source in a query's scope that is a named ``exp.Table``: a name bound to a
    common table expression is that expression's scope, and a table-valued function has no
    name. A bare ``VALUES`` has no scope, so the queries inside it are read instead.
    """
    if not statements or not all(map(_is_query, statements)):
        return None
    queries = [
        query
        for statement in statements
        for query in (
            [statement] if isinstance(statement, exp.Query) else statement.find_all(exp.Query)
        )
    ]
    try:
        return any(
            isinstance(source, exp.Table) and isinstance(source.this, exp.Identifier)
            for query in queries
            for scope in traverse_scope(query)
            for source in scope.sources.values()
        )
    except (SqlglotError, RecursionError):
        return None


def _reads_data(statements: Sequence[exp.Expr] | None) -> bool | None:
    """Whether the queries answer with what they read from the database, rather than with
    constants (the module docstring's ``constant_query``): ``None`` when
    ``_reads_a_table`` is."""
    tables = _reads_a_table(statements)
    if statements is None or not tables:
        return tables
    return not any(map(_has_constant_rows, statements))


def _has_constant_rows(query: exp.Expr) -> bool:
    """Whether some of the rows ``query`` returns are constants it writes: those of a
    ``SELECT`` whose rows it returns or a row of a ``VALUES``, all of whose columns are."""
    # A stack rather than recursion, as a long chain of UNIONs nests as deep as it is long.
    stack = [query]
    while stack:
        node = stack.pop()
        if isinstance(node, exp.Subquery):
            stack.append(node.this)
        elif isinstance(node, exp.Except):
            # The right side's rows are taken away, never returned.
            stack.append(node.left)
        elif isinstance(node, exp.SetOperation):
            stack.extend((node.left, node.right))
        # A VALUES holds its rows as tuples of its columns.
        elif (
            isinstance(node, exp.Select) and node.selects and all(map(_is_constant, node.selects))
        ) or (isinstance(node, exp.Values) and any(map(_is_constant, node.expressions))):
            return True
    return False


def _is_constant(column: exp.Expr) -> bool:
    """Whether a column of a query's result holds what it writes rather than what it reads:
    nothing in it, subqueries included, names a column or ``*`` or calls a window function,
    or an aggregate that doesn't return the constant it is given."""
    return not any(
        isinstance(node, exp.Column | exp.Star | exp.Window)
        or (isinstance(node, exp.AggFunc) and not isinstance(node, _CONSTANT_AGGREGATES))
        for node in column.walk()
    )


@lru_cache(maxsize=16_384)
def _gold_reads_data(gold_sql: str) -> bool:
    """Whether the gold query answers with what it reads from the database; one sqlglot
    can't read is taken to, as all 7,787 train gold queries do."""
    reads = _reads_data(_read(gold_sql))
    if reads is None:
        log.warning("sqlglot can't read the gold query %r; taking it to read a table", gold_sql)
        return True
    return reads


@lru_cache(maxsize=16_384)
def _order_matters(gold_sql: str) -> bool:
    return order_matters(gold_sql)


def _formatted(reply: str) -> bool:
    """Whether the reply is one sql code block with only whitespace around it."""
    match = _SQL_REPLY.fullmatch(reply)
    return match is not None and "```" not in match["body"]
