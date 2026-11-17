"""Plan section 3.8's GPU hour budget, read one way for the tests that check it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BudgetRow:
    job: str  # the job's name before any colon or parenthesis; Total for the total row
    model: str
    hours: float
    story: str


def budget_rows() -> list[BudgetRow]:
    """The rows of plan section 3.8's GPU hour budget, the total row last."""
    plan = (ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8")
    section = plan.split("\n### 3.8 ", 1)[1].split("\n### ", 1)[0]
    table = section.split("**GPU hour budget**", 1)[1].split("\n\n")[1]
    rows = []
    for line in table.splitlines()[2:]:
        job, model, hours, story = (cell.strip() for cell in line.strip("|").split("|"))
        number = re.search(r"[\d.]+", hours)
        assert number, line
        name = re.split(r"[:(]", job.strip("*"), maxsplit=1)[0].strip()
        rows.append(BudgetRow(name, model, float(number.group()), story))
    return rows
