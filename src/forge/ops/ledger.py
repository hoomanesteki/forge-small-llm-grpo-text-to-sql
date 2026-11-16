"""The compute ledger: one markdown row per GPU session, append-only.

Free GPU quota is the project's real budget, so every session is written down.
Only Kaggle hours count against the weekly quota; Colab and Modal hours still count
toward the project's GPU total, and Modal is where cash can be spent. Laptop sessions,
on its CPU or on Apple silicon's GPU (``local-mps``, ADR-0016), spend neither budget
and count toward neither.

A run's hours are its wall-clock time, from its manifest's ``started_at`` to its
``ended_at`` (``run_hours``): a session holds its GPU for all of it, loading and
uploading included.

``forge runs pull`` writes a run's row from its manifest (``record_run``) once the run
has ended, on any platform with a GPU, and only once a run id: the check and the append
share one lock. ``audit`` is invariant I7: every run on a budget platform has a row, a
row names its run's platform, and the cash spent stays within ``CASH_LIMIT_USD``.

Every date here is a UTC date, as run ids are (``forge.ops.clock``): a pulled row takes
the UTC day its run started, ``forge ledger add`` today's UTC date unless told another,
and ``summary``'s window ends on today's UTC date. A row's date and its run id's date
then agree, and a session that ended in the evening west of UTC counts this week at once.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from forge.core.types import RunManifest
from forge.ops.clock import utc_today
from forge.ops.logfile import locked_append
from forge.ops.paths import ledger_path

COLUMNS = ("date", "platform", "gpu", "hours", "run_id", "purpose", "cost_usd")
PLATFORMS = frozenset({"local-cpu", "local-mps", "kaggle", "colab", "modal"})
GPU_PLATFORMS = frozenset({"kaggle", "colab", "modal"})
QUOTA_PLATFORMS = frozenset({"kaggle"})
# Platforms whose runs hold a GPU, the laptop's through MPS among them. Their manifests
# count wall-clock hours as GPU hours; only GPU_PLATFORMS spend the free GPU budget.
GPU_RUN_PLATFORMS = GPU_PLATFORMS | {"local-mps"}
# Characters that would break a markdown table row.
_TABLE_BREAKERS = re.compile(r"[|\r\n\u2028\u2029\x85]")
HEADER = "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n"
PREAMBLE = "# Compute ledger\n\n" + HEADER
# The project's cash budget (plan section 4, I7); the GPUs themselves are free.
CASH_LIMIT_USD = 50.0


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


def run_hours(started_at: datetime, ended_at: datetime) -> float:
    """The wall-clock hours from ``started_at`` to ``ended_at``.

    Raises ``ValueError`` when the end comes first: a manifest that says so is corrupt.
    """
    seconds = (ended_at - started_at).total_seconds()
    if seconds < 0:
        raise ValueError(f"a run can't end ({ended_at}) before it starts ({started_at})")
    return seconds / 3600


def read_rows(path: Path | None = None) -> list[LedgerRow]:
    path = path or ledger_path()
    if not path.is_file():
        return []
    return _parse(path.read_text(encoding="utf-8"))


def _parse(text: str) -> list[LedgerRow]:
    # Data rows start with an ISO date; the header and separator do not.
    return [LedgerRow.from_markdown(line) for line in text.splitlines() if line.startswith("| 2")]


def append(row: LedgerRow, path: Path | None = None) -> None:
    locked_append(path or ledger_path(), row.to_markdown(), preamble=PREAMBLE)


def has_run(run_id: str, path: Path | None = None) -> bool:
    return any(r.run_id == run_id for r in read_rows(path))


class _Logged(Exception):
    """The ledger has a row for the run already."""


def append_once(row: LedgerRow, path: Path | None = None) -> bool:
    """Append ``row`` unless the ledger has a row for its run id; returns whether it did.

    The check runs under the append's lock, so two pulls of one run at once add one row.
    """

    def unlogged(before: str) -> None:
        if any(logged.run_id == row.run_id for logged in _parse(before)):
            raise _Logged

    try:
        locked_append(path or ledger_path(), row.to_markdown(), preamble=PREAMBLE, check=unlogged)
    except _Logged:
        return False
    return True


def purpose(manifest: RunManifest) -> str:
    """What a run was, for its row: its kind and model, its config's split, prompt version
    and limit when it names them, and how it ended."""
    words = f"{manifest.kind} {manifest.model_ref}"
    if manifest.adapter_ref:
        words += f" with {manifest.adapter_ref}"
    config = manifest.config
    if config.get("split"):
        words += f" on {config['split']}"
    for name, label in (("prompt_version", "prompt"), ("limit", "limit")):
        if config.get(name) is not None:
            words += f", {label} {config[name]}"
    return f"{words}, {manifest.status}"


def run_row(manifest: RunManifest) -> LedgerRow:
    """The ledger row of a run that has ended: the UTC date it started, where it ran and on
    which GPU, its wall-clock hours, what it was and its cost.

    Raises ``ValueError`` for a run that hasn't ended, whose hours aren't known yet.
    """
    if manifest.ended_at is None:
        raise ValueError(f"run {manifest.run_id} is {manifest.status}, so its hours aren't known")
    return LedgerRow(
        date=manifest.started_at.astimezone(UTC).date(),
        platform=manifest.platform,
        gpu=manifest.gpu or "",
        hours=run_hours(manifest.started_at, manifest.ended_at),
        run_id=manifest.run_id,
        purpose=purpose(manifest),
        cost_usd=manifest.cost_usd,
    )


def record_run(manifest: RunManifest, path: Path | None = None) -> tuple[LedgerRow | None, str]:
    """Append the row of a run that held a GPU, once; returns the row it appended, if any,
    and a line saying what it did.

    A ``local-cpu`` run gets no row, and a run that hasn't ended gets none yet: pull it
    again once it has.
    """
    run = manifest.run_id
    if manifest.platform not in GPU_RUN_PLATFORMS:
        return None, f"ledger: no row, a {manifest.platform} run holds no GPU"
    if manifest.ended_at is None:
        return None, f"ledger: no row yet, run {run} is {manifest.status}; pull it once it ends"
    row = run_row(manifest)
    if not append_once(row, path):
        return None, f"ledger: run {run} has its row already"
    return row, f"ledger: added {row.hours:.2f} h on {row.platform} for run {run}"


def audit(manifests: Iterable[RunManifest], rows: Sequence[LedgerRow]) -> list[str]:
    """How ``manifests`` and the ledger's ``rows`` break invariant I7, a line each.

    A run on a budget platform (``GPU_PLATFORMS``) needs a row. Any run with a row, a
    ``local-mps`` pilot among them, needs its row to name its platform. All the rows'
    cash together must stay within ``CASH_LIMIT_USD``.
    """
    logged: dict[str, list[LedgerRow]] = {}
    for row in rows:
        logged.setdefault(row.run_id, []).append(row)
    problems = []
    for manifest in manifests:
        run, platform = manifest.run_id, manifest.platform
        if platform in GPU_PLATFORMS and run not in logged:
            problems.append(f"run {run} ran on {platform}, and the ledger has no row for it")
        problems += [
            f"run {run} ran on {platform}, but its ledger row of {row.date} says {row.platform}"
            for row in logged.get(run, [])
            if row.platform != platform
        ]
    cash = sum(row.cost_usd for row in rows)
    if cash > CASH_LIMIT_USD:
        problems.append(f"the ledger spends ${cash:.2f}, over the ${CASH_LIMIT_USD:.0f} limit")
    return problems


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
    """The ledger's totals, with the Kaggle hours of the ``window_days`` UTC days that end
    on ``today``, by default today's UTC date."""
    today = today or utc_today()
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
