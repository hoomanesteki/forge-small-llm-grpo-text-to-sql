from __future__ import annotations

import pytest
from plan_parser import load_plan, parse_plan

SAMPLE = """\
## 6. Backlog

| E01 | Repo and delivery tooling | T6 | 0 | 5 |

## Phase 0: Foundation (tag v0.0.1)

#### FG-001 · Repo scaffold
- **Epic:** E01 · **Points:** 3 · **Branch:** `chore/fg-001-scaffold` · **Depends on:** none
- **Build:** make the repo.
- **Tests:** `tests/unit/test_a.py::test_one`, `tests/unit/test_a.py::TestX::test_two`

```markdown
#### FG-999 · Inside a code fence, not a story
```

## Phase 1: Data (tag v0.1.0)

#### FG-101 · Spike: data sources
- **Epic:** E01 · **Points:** 2 · **Branch:** `spike/fg-101-data` · **Depends on:** FG-001, FG-003
- **Build:** **GPU job**, budget 1 h. Measure things.
- **Tests:** none (release checklist)
"""


@pytest.mark.phase(0)
def test_parses_story_blocks() -> None:
    plan = parse_plan(SAMPLE)
    assert [s.id for s in plan.stories] == ["FG-001", "FG-101"]
    first, spike = plan.stories
    assert (first.phase, first.epic, first.points) == (0, "E01", 3)
    assert first.branch == "chore/fg-001-scaffold"
    assert first.depends_on == ()
    assert first.tests == (
        "tests/unit/test_a.py::test_one",
        "tests/unit/test_a.py::TestX::test_two",
    )
    assert not first.gpu and first.kind == "story"
    assert spike.depends_on == ("FG-001", "FG-003")
    assert spike.tests == ()
    assert spike.gpu and spike.kind == "spike"
    assert plan.epics["E01"].theme == "T6"
    assert [p.tag for p in plan.phases] == ["v0.0.1", "v0.1.0"]


@pytest.mark.phase(0)
def test_totals_match_plan() -> None:
    plan = load_plan()
    assert plan.totals() == {"stories": 56, "points": 190, "phases": 9, "gpu": 11}
    assert sum(e.points for e in plan.epics.values()) == 190
    for story in plan.stories:
        assert story.epic in plan.epics, story.id
