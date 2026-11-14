"""Parse the story blocks in docs/PLAN.md.

Story headings look like ``#### FG-101 · Spike: ...``. The line after carries the
fields: ``- **Epic:** E03 · **Points:** 2 · **Branch:** `spike/...` · **Depends on:** ...``.
Fenced code blocks are skipped so examples inside them never count as stories.

    uv run python scripts/plan_parser.py           # summary
    uv run python scripts/plan_parser.py --json    # every story as JSON
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "docs" / "PLAN.md"

STORY_RE = re.compile(r"^#### (FG-\d{3}) · (.+)$")
PHASE_RE = re.compile(r"^## Phase (\d+): (.+?)(?: \(tag (v[\d.]+)\))?$")
EPIC_ROW_RE = re.compile(r"^\| (E\d{2}) \| (.+?) \| (T\d) \| [\d, -]+ \| (\d+) \|$")
FIELDS_RE = re.compile(
    r"^- \*\*Epic:\*\* (E\d{2}) · \*\*Points:\*\* (\d+) · \*\*Branch:\*\* `([^`]+)`"
    r" · \*\*Depends on:\*\* (.+)$"
)
TEST_REF_RE = re.compile(r"`(tests/[^`:]+\.py(?:::[A-Za-z_][A-Za-z0-9_]*)+)`")
STORY_ID_RE = re.compile(r"FG-\d{3}")


@dataclass(frozen=True)
class Story:
    id: str
    title: str
    phase: int
    epic: str
    points: int
    branch: str
    depends_on: tuple[str, ...]
    tests: tuple[str, ...]
    gpu: bool
    body: str = field(repr=False, compare=False)

    @property
    def kind(self) -> str:
        if self.title.startswith("Spike:"):
            return "spike"
        return "gpu" if self.gpu else "story"


@dataclass(frozen=True)
class Phase:
    number: int
    name: str
    tag: str | None


@dataclass(frozen=True)
class Epic:
    id: str
    name: str
    theme: str
    points: int


@dataclass
class Plan:
    stories: list[Story]
    phases: list[Phase]
    epics: dict[str, Epic]

    def totals(self) -> dict[str, int]:
        return {
            "stories": len(self.stories),
            "points": sum(s.points for s in self.stories),
            "phases": len({s.phase for s in self.stories}),
            "gpu": sum(s.gpu for s in self.stories),
        }

    def by_phase(self, phase: int) -> list[Story]:
        return [s for s in self.stories if s.phase == phase]


def _strip_fences(lines: list[str]) -> list[str]:
    """Blank out fenced code blocks so line numbers stay aligned."""
    out, inside = [], False
    for line in lines:
        if line.lstrip().startswith("```"):
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return out


def _block_end(lines: list[str], start: int) -> int:
    for i in range(start + 1, len(lines)):
        if lines[i].startswith(("#### ", "### ", "## ")):
            return i
    return len(lines)


def parse_plan(text: str) -> Plan:
    raw = text.splitlines()
    lines = _strip_fences(raw)
    stories: list[Story] = []
    phases: list[Phase] = []
    epics: dict[str, Epic] = {}
    phase: int | None = None

    for i, line in enumerate(lines):
        if m := PHASE_RE.match(line):
            phase = int(m.group(1))
            phases.append(Phase(phase, m.group(2), m.group(3)))
            continue
        if m := EPIC_ROW_RE.match(line):
            epics[m.group(1)] = Epic(m.group(1), m.group(2), m.group(3), int(m.group(4)))
            continue
        m = STORY_RE.match(line)
        if not m:
            continue
        if phase is None:
            raise ValueError(f"story {m.group(1)} appears before any '## Phase N:' heading")
        end = _block_end(lines, i)
        block = lines[i + 1 : end]
        fields = next((FIELDS_RE.match(b) for b in block if b.startswith("- **Epic:**")), None)
        if fields is None:
            raise ValueError(f"story {m.group(1)} has no Epic/Points/Branch/Depends line")
        tests_line = next((b for b in block if b.startswith("- **Tests:**")), "")
        build_line = next((b for b in block if b.startswith("- **Build:**")), "")
        depends = fields.group(4).strip()
        stories.append(
            Story(
                id=m.group(1),
                title=m.group(2).strip(),
                phase=phase,
                epic=fields.group(1),
                points=int(fields.group(2)),
                branch=fields.group(3),
                depends_on=tuple(STORY_ID_RE.findall(depends)),
                tests=tuple(TEST_REF_RE.findall(tests_line)),
                gpu=build_line.removeprefix("- **Build:**").strip().startswith("**GPU job**"),
                body="\n".join(raw[i:end]).strip(),
            )
        )
    return Plan(stories=stories, phases=phases, epics=epics)


def load_plan(path: Path = DEFAULT_PLAN) -> Plan:
    return parse_plan(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--json", action="store_true", help="print every story as JSON")
    args = parser.parse_args(argv)

    plan = load_plan(args.plan)
    if args.json:
        rows = [{k: v for k, v in asdict(s).items() if k != "body"} for s in plan.stories]
        json.dump(rows, sys.stdout, indent=2)
        print()
        return 0

    t = plan.totals()
    print(f"{t['stories']} stories, {t['points']} points, {t['phases']} phases, {t['gpu']} GPU")
    points = Counter[int]()
    for s in plan.stories:
        points[s.phase] += s.points
    for ph in plan.phases:
        print(
            f"  phase {ph.number}: {len(plan.by_phase(ph.number)):>2} stories, "
            f"{points[ph.number]:>2} pts  {ph.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
