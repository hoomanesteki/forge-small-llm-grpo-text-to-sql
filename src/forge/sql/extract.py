"""Pull the SQL query out of a model reply, by the rules in plan section 3.7.

1. Drop every ``<think>...</think>`` block. Qwen3 emits an empty one even in non-thinking
   mode, and a query drafted inside one is not the answer.
2. Take the last ```` ```sql ```` code block (the tag is case-insensitive). A bare
   ```` ``` ```` block is not an sql block; only its ``SELECT`` or ``WITH`` lines can
   count, through rule 3.
3. Otherwise take the last line that starts with ``SELECT`` or ``WITH``, ignoring case and
   indentation, and that sqlglot reads as a query, so a sentence such as "With this query
   you get..." after the query is passed over, though one short enough to parse, such as
   "Select it", is not. Only that line is taken, so an unfenced query split over several
   lines loses the lines after it.
4. Otherwise there is no SQL: ``None``.

The result has surrounding whitespace and one trailing semicolon removed. An empty
result, such as an empty last code block, is ``None`` too.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_SQL_BLOCK = re.compile(r"```sql\b(.*?)```", re.DOTALL | re.IGNORECASE)
_QUERY_LINE = re.compile(r"\s*(?:select|with)\b", re.IGNORECASE)


def extract_sql(text: str) -> str | None:
    """The reply's SQL query, or ``None`` when it has none."""
    text = _THINK_BLOCK.sub("", text)
    blocks = _SQL_BLOCK.findall(text)
    if blocks:
        return _clean(blocks[-1])
    for line in reversed(text.splitlines()):
        if _QUERY_LINE.match(line):
            sql = _clean(line)
            if sql is not None and _parses_as_query(sql):
                return sql
    return None


def _parses_as_query(sql: str) -> bool:
    """Whether sqlglot reads the line as a query; whether it runs is the executor's call."""
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    # sqlglot recurses per nesting level and runs out of stack at about 45 parentheses,
    # which no real query reaches.
    except (SqlglotError, RecursionError):
        return False
    return bool(statements) and isinstance(statements[0], exp.Query)


def _clean(sql: str) -> str | None:
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql or None
