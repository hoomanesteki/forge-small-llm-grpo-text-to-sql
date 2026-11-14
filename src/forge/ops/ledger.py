"""The compute ledger: one markdown row per GPU session, append-only.

Free GPU quota is the project's real budget, so every session is written down.
Only Kaggle hours count against the weekly quota; Colab and Modal hours still count
toward the project's GPU total, and Modal is where cash can be spent.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from forge.ops.logfile import locked_append
from forge.ops.paths import ledger_path

COLUMNS = ("date", "platform", "gpu", "hours", "run_id", "purpose", "cost_usd")
PLATFORMS = frozenset({"local-cpu", "kaggle", "colab", "modal"})
GPU_PLATFORMS = frozenset({"kaggle", "colab", "modal"})
QUOTA_PLATFORMS = frozenset({"kaggle"})
# Characters that would break a markdown table row.
_TABLE_BREAKERS = re.compile(r"[|\r\n\u2028\u2029\x85]")
HEADER = "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n"


@dataclass(frozen=True)
class LedgerRow:
    date: date
    platform: str
    gpu: str
    hours: float
    run_id: str
    purpose: str
    cost_usd: float = 0.0

    def validate(self) -> None:
        if self.platform not in PLATFORMS:
            raise ValueError(
                f"unknown platform {self.platform!r}; expected one of {sorted(PLATFORMS)}"
            )
        for name, number in (("hours", self.hours), ("cost_usd", self.cost_usd)):
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{name} must be a finite, non-negative number, got {number!r}")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        for name, text in (("run_id", self.run_id), ("gpu", self.gpu)):
            if _TABLE_BREAKERS.search(text):
                raise ValueError(f"{name} must not contain '|' or line breaks: {text!r}")

    def to_markdown(self) -> str:
        self.validate()
        cells = [
            self.date.isoformat(),
            self.platform,
            self.gpu or "-",
            f"{self.hours:.2f}",
            self.run_id,
            # Free text is the only field we repair rather than reject.
            re.sub(r"\s*[|\r\n\u2028\u2029\x85]+\s*", " / ", self.purpose).strip(),
            f"{self.cost_usd:.2f}",
        ]
        return "| " + " | ".join(cells) + " |"

    @classmethod
    def from_markdown(cls, line: str) -> LedgerRow:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(COLUMNS):
            raise ValueError(
                f"ledger row has {len(cells)} cells, expected {len(COLUMNS)}: {line!r}"
            )
        d, platform, gpu, hours, run_id, purpose, cost = cells
        return cls(
            date=date.fromisoformat(d),
            platform=platform,
            gpu="" if gpu == "-" else gpu,
            hours=float(hours),
            run_id=run_id,
            purpose=purpose,
            cost_usd=float(cost),
        )


def read_rows(path: Path | None = None) -> list[LedgerRow]:
    path = path or ledger_path()
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # Data rows start with an ISO date; the header and separator do not.
        if line.startswith("| 2"):
            rows.append(LedgerRow.from_markdown(line))
    return rows


def append(row: LedgerRow, path: Path | None = None) -> None:
    locked_append(
        path or ledger_path(), row.to_markdown(), preamble="# Compute ledger\n\n" + HEADER
    )


def has_run(run_id: str, path: Path | None = None) -> bool:
    return any(r.run_id == run_id for r in read_rows(path))


@dataclass(frozen=True)
class Summary:
    window_days: int
    quota_hours_window: float
    gpu_hours_total: float
    cash_total: float
    weekly_quota: float
    sessions: int

    @property
    def remaining_week(self) -> float:
        return max(0.0, self.weekly_quota - self.quota_hours_window)

    def render(self) -> str:
        return "\n".join(
            [
                f"Kaggle hours, last {self.window_days} days: {self.quota_hours_window:.2f} "
                f"of {self.weekly_quota:.0f} (remaining {self.remaining_week:.2f})",
                f"GPU hours, all time: {self.gpu_hours_total:.2f} over {self.sessions} sessions",
                f"Cash spent: ${self.cash_total:.2f}",
            ]
        )


def summary(window_days: int = 7, path: Path | None = None, today: date | None = None) -> Summary:
    today = today or date.today()
    start = today - timedelta(days=window_days - 1)
    rows = read_rows(path)
    from forge.ops.dotenv import forge_env

    weekly_quota = float(forge_env().get("FORGE_WEEKLY_GPU_HOURS", "30"))
    return Summary(
        window_days=window_days,
        quota_hours_window=sum(
            r.hours for r in rows if r.platform in QUOTA_PLATFORMS and start <= r.date <= today
        ),
        gpu_hours_total=sum(r.hours for r in rows if r.platform in GPU_PLATFORMS),
        cash_total=sum(r.cost_usd for r in rows),
        weekly_quota=weekly_quota,
        sessions=sum(r.platform in GPU_PLATFORMS for r in rows),
    )
