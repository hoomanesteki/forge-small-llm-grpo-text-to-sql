"""The compute ledger: one markdown row per GPU session, append-only.

Free GPU quota is the project's real budget, so every session is written down.
Only Kaggle hours count against the weekly quota; Colab and Modal hours still count
toward the project's GPU total, and Modal is where cash can be spent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from forge.ops.paths import ledger_path

COLUMNS = ("date", "platform", "gpu", "hours", "run_id", "purpose", "cost_usd")
GPU_PLATFORMS = frozenset({"kaggle", "colab", "modal"})
QUOTA_PLATFORMS = frozenset({"kaggle"})
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

    def to_markdown(self) -> str:
        cells = [
            self.date.isoformat(),
            self.platform,
            self.gpu or "-",
            f"{self.hours:.2f}",
            self.run_id,
            self.purpose.replace("|", "/").replace("\n", " "),
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
    path = path or ledger_path()
    if row.hours < 0 or row.cost_usd < 0:
        raise ValueError("hours and cost must be non-negative")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Compute ledger\n\n" + HEADER, encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + row.to_markdown() + "\n", encoding="utf-8")


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
    weekly_quota = float(os.environ.get("FORGE_WEEKLY_GPU_HOURS", "30"))
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
