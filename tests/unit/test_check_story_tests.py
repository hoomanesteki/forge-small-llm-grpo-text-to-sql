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
- **Tests:** `tests/unit/test_missing_file.py::test_any`, `tests/unit/test_real.py::test_here`
"""

TEST_FILE = """\
import pytest

@pytest.mark.phase(0)
def test_here():
    pass

@pytest.mark.phase(0)
class TestGroup:
    def test_method(self):
        pass

def test_unmarked():
    pass

@pytest.mark.skip(reason="later")
@pytest.mark.phase(0)
def test_skipped():
    pass

@pytest.mark.phase(2)
def test_from_phase_two():
    pass
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_real.py").write_text(TEST_FILE)
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
    ref = "tests/unit/test_real.py::TestGroup::test_method"
    assert resolve(ref, repo) is None
    assert resolve(ref, repo, story_phase=0) is None, "class marker applies to its methods"
    assert resolve("tests/unit/test_real.py::TestGroup::test_nope", repo) == "test_nope not defined"
    assert resolve("tests/unit/test_real.py::test_here::x", repo) == "test_here is not a class"


@pytest.mark.phase(0)
def test_flags_unusable_tests(repo: Path) -> None:
    assert resolve("tests/unit/test_real.py::test_unmarked", repo, 0) == "no phase marker"
    assert resolve("tests/unit/test_real.py::test_skipped", repo, 0) == "marked skip"
    assert resolve("tests/unit/test_real.py::test_from_phase_two", repo, 1) == (
        "marked phase 2, after the story's phase 1"
    )
    # An earlier test may satisfy a later story.
    assert resolve("tests/unit/test_real.py::test_here", repo, 3) is None


@pytest.mark.phase(0)
def test_module_level_pytestmark(tmp_path: Path) -> None:
    (tmp_path / "t.py").write_text(
        "import pytest\npytestmark = [pytest.mark.phase(1)]\ndef test_a():\n    pass\n"
    )
    assert resolve("t.py::test_a", tmp_path, 1) is None
    assert resolve("t.py::test_a", tmp_path, 0) == "marked phase 1, after the story's phase 0"
