"""Pull the SQL query out of a model reply, by the rules in plan section 3.7.

1. Drop every ``<think>...</think>`` block. Qwen3 emits an empty one even in non-thinking
   mode, and a query drafted inside one is not the answer.
2. Take the last ```` ```sql ```` code block (the tag is case-insensitive). A bare
   ```` ``` ```` block is not an sql block; only its ``SELECT`` or ``WITH`` lines can
   count, through rule 3.
3. Otherwise take the last line that starts with ``SELECT`` or ``WITH``, ignoring case and
   indentation. Only that line is taken, so an unfenced query split over several lines
   loses the lines after it.
4. Otherwise there is no SQL: ``None``.

The result has surrounding whitespace and one trailing semicolon removed. An empty
result, such as an empty last code block, is ``None`` too.
"""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_SQL_BLOCK = re.compile(r"```sql\b(.*?)```", re.DOTALL | re.IGNORECASE)
_QUERY_LINE = re.compile(r"\s*(?:select|with)\b", re.IGNORECASE)


def extract_sql(text: str) -> str | None:
    """The reply's SQL query, or ``None`` when it has none."""
    text = _THINK_BLOCK.sub("", text)
    blocks = _SQL_BLOCK.findall(text)
    if blocks:
        return _clean(blocks[-1])
    lines = [line for line in text.splitlines() if _QUERY_LINE.match(line)]
    if lines:
        return _clean(lines[-1])
    return None


def _clean(sql: str) -> str | None:
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql or None
