"""SQL extraction from model replies, by the plan's section 3.7 rules."""

from __future__ import annotations

import pytest

from forge.sql.extract import extract_sql


@pytest.mark.phase(1)
def test_last_code_block_wins() -> None:
    reply = (
        "A first try:\n```sql\nSELECT name FROM singer\n```\n"
        "Better, with the age:\n```SQL\nSELECT name, age\nFROM singer\nWHERE age > 30;\n```\n"
        "And a bare block, which is not an sql block:\n```\nSELECT 1\n```\n"
    )

    assert extract_sql(reply) == "SELECT name, age\nFROM singer\nWHERE age > 30"


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("reply", "sql"),
    [
        # No code block: the last SELECT or WITH line, whatever its case and indentation.
        ("Here it is:\nselect count(*) from singer;\nDone.", "select count(*) from singer"),
        ("SELECT 1\nThen the real one:\n   SELECT name FROM singer", "SELECT name FROM singer"),
        (
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t ;  \nThat uses a CTE.",
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        ),
        # A bare block is not an sql block, but its query line still counts.
        ("```\nSELECT name FROM singer;\n```", "SELECT name FROM singer"),
        # Only one line is taken: the rule is the last line, not the rest of the reply.
        ("SELECT name\nFROM singer", "SELECT name"),
        # A word that merely starts with SELECT or WITH is not a query line.
        ("SELECT id FROM t\nSelected columns: id\nWithout a filter.", "SELECT id FROM t"),
    ],
)
def test_fallback_select_line(reply: str, sql: str) -> None:
    assert extract_sql(reply) == sql


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("reply", "sql"),
    [
        # A sentence that starts with With or Select after the query is not the query.
        (
            "SELECT name FROM singer WHERE age > 30\n"
            "With this query you get the singers older than 30.",
            "SELECT name FROM singer WHERE age > 30",
        ),
        (
            "select count(*) from singer;\nSelect the singers older than 30.",
            "select count(*) from singer",
        ),
        (
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t\nWith that, the query is done.",
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
        ),
        # Prose alone is no query at all.
        ("With this query you get the singers.\nSelect the right table first.", None),
        # A line sqlglot can't follow is not taken for a query, and doesn't raise.
        ("SELECT " + "(" * 60 + "1" + ")" * 60, None),
    ],
)
def test_fallback_line_must_parse_as_a_query(reply: str, sql: str | None) -> None:
    assert extract_sql(reply) == sql


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   \n\t",
        "I cannot answer that from this schema.",
        "Selecting is not possible here.\nWithout more tables, no.",
        "```python\nprint('no query')\n```",
        # An empty last sql block holds no query, even after an earlier one.
        "```sql\nSELECT 1\n```\nNo, scrap that:\n```sql\n;\n```",
        # A query drafted inside a think block is not the answer.
        "<think>\nSELECT name FROM singer\n</think>\nI am not sure.",
    ],
)
def test_none_when_no_sql(reply: str) -> None:
    assert extract_sql(reply) is None


@pytest.mark.phase(1)
def test_think_block_is_ignored() -> None:
    # Qwen3 in non-thinking mode still opens with an empty think block.
    assert extract_sql("<think>\n\n</think>\n\n```sql\nSELECT 1;\n```") == "SELECT 1"
    reply = (
        "<think>\n```sql\nSELECT draft FROM t\n```\n</think>\n"
        "```sql\nSELECT name FROM singer\n```\n"
        "<think>\nSELECT another_draft FROM t\n</think>"
    )
    assert extract_sql(reply) == "SELECT name FROM singer"


@pytest.mark.phase(1)
@pytest.mark.parametrize(
    ("reply", "sql"),
    [
        ("```sql\n  SELECT 1;  \n```", "SELECT 1"),
        # One trailing semicolon goes; what is left is the executor's to judge.
        ("```sql\nSELECT 1;;\n```", "SELECT 1;"),
        ("```sql\nSELECT 1; SELECT 2;\n```", "SELECT 1; SELECT 2"),
        ("```sql\nSELECT ';'\n```", "SELECT ';'"),
        ("```sql SELECT 1```", "SELECT 1"),
    ],
)
def test_strips_one_trailing_semicolon(reply: str, sql: str) -> None:
    assert extract_sql(reply) == sql


@pytest.mark.phase(1)
def test_other_language_tags_are_not_sql_blocks() -> None:
    reply = "```sqlite\nSELECT a FROM t\n```\n```sql\nSELECT b FROM t\n```\n```sqlite\nnot sql\n```"

    assert extract_sql(reply) == "SELECT b FROM t"
