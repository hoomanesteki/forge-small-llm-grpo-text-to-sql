from __future__ import annotations

from pathlib import Path

import pytest
from check_story_tests import find_missing, resolve
from plan_parser import parse_plan

PLAN = """\
## Phase 0: Foundation (tag v0.0.1)

#### FG-001 · Something
- **Epic:** E01 · **Points:** 1 · **Branch:** `chore/x` · **Depends on:** none
- **Tests:** `tests/unit/test_real.py::test_here`, `tests/unit/test_real.py::test_gone`

## Phase 1: Later (tag v0.1.0)

#### FG-101 · Later thing
- **Epic:** E01 · **Points:** 1 · **Branch:** `feat/y` · **Depends on:** none
- **Tests:** `tests/unit/test_missing_file.py::test_any`
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_real.py").write_text(
        "def test_here():\n    pass\n\nclass TestGroup:\n    def test_method(self):\n        pass\n"
    )
    return tmp_path


@pytest.mark.phase(0)
def test_reports_missing_test(repo: Path) -> None:
    missing = find_missing(parse_plan(PLAN), phase=0, root=repo)
    assert [(m.story, m.ref, m.why) for m in missing] == [
        ("FG-001", "tests/unit/test_real.py::test_gone", "test_gone not defined"),
    ]
    later = find_missing(parse_plan(PLAN), phase=1, root=repo)
    assert {m.ref for m in later} == {
        "tests/unit/test_real.py::test_gone",
        "tests/unit/test_missing_file.py::test_any",
    }


@pytest.mark.phase(0)
def test_resolves_class_methods(repo: Path) -> None:
    assert resolve("tests/unit/test_real.py::TestGroup::test_method", repo) is None
    assert resolve("tests/unit/test_real.py::TestGroup::test_nope", repo) == "test_nope not defined"
    assert resolve("tests/unit/test_real.py::test_here::x", repo) == "test_here is not a class"
