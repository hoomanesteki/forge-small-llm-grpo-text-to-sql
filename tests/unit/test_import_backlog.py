from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from import_backlog import import_backlog
from plan_parser import load_plan


class FakeGh:
    """Records gh calls and remembers what was created, like the real repo would."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.issues: list[str] = []
        self.milestones: list[str] = []

    def __call__(self, args: Sequence[str]) -> str:
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["issue", "list"]:
            return json.dumps([{"title": t} for t in self.issues])
        if args[0] == "api" and args[1].endswith("milestones?state=all&per_page=100"):
            return json.dumps([{"title": t} for t in self.milestones])
        if args[0] == "api":
            self.milestones.append(args[args.index("-f") + 1].split("=", 1)[1])
        if args[:2] == ["issue", "create"]:
            self.issues.append(args[args.index("--title") + 1])
        return ""


@pytest.mark.phase(0)
def test_second_import_creates_nothing_new() -> None:
    plan = load_plan()
    fake = FakeGh()
    first = import_backlog(plan, run=fake)
    assert sum(a.startswith("issue ") for a in first) == 56
    assert sum(a.startswith("milestone ") for a in first) == 9
    second = import_backlog(plan, run=fake)
    assert not [a for a in second if a.startswith(("issue ", "milestone "))]
    assert len(fake.issues) == 56


@pytest.mark.phase(0)
def test_dry_run_makes_no_calls() -> None:
    def refuse(args: Sequence[str]) -> str:
        raise AssertionError(f"dry run called gh {args}")

    actions = import_backlog(load_plan(), run=refuse, dry_run=True)
    assert sum(a.startswith("issue ") for a in actions) == 56
