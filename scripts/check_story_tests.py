"""Check that every test a story names in docs/PLAN.md exists.

A gate runs ``check_story_tests.py --phase N``. It covers phases 0..N, so a test
deleted after its own gate still gets caught later. Test references look like
``tests/unit/test_x.py::test_name`` or ``tests/unit/test_x.py::TestClass::test_name``
and are resolved with ``ast``, without importing anything.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_parser import DEFAULT_PLAN, ROOT, Plan, load_plan


@dataclass(frozen=True)
class Missing:
    story: str
    ref: str
    why: str


def _defined_names(body: list[ast.stmt]) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def _marks(exprs: list[ast.expr]) -> dict[str, ast.expr]:
    """Map mark name to its expression for ``pytest.mark.<name>`` or ``pytest.mark.<name>(...)``."""
    found: dict[str, ast.expr] = {}
    for expr in exprs:
        target = expr.func if isinstance(expr, ast.Call) else expr
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "mark"
        ):
            found.setdefault(target.attr, expr)
    return found


def _module_marks(tree: ast.Module) -> list[ast.expr]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            value = node.value
            return list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]
    return []


def _phase_of(mark: ast.expr | None) -> int | None:
    if isinstance(mark, ast.Call) and len(mark.args) == 1:
        arg = mark.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            return arg.value
    return None


def resolve(ref: str, root: Path, story_phase: int | None = None) -> str | None:
    """Return None when the test exists and is usable, or a short reason when it isn't.

    With ``story_phase``, the test must also carry a phase marker no later than the
    story's phase (an earlier test may satisfy a later story) and must not be
    unconditionally skipped or marked xfail.
    """
    path_part, *names = ref.split("::")
    path = root / path_part
    if not path.is_file():
        return "file not found"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = _defined_names(tree.body)
    decorators: list[ast.expr] = _module_marks(tree)
    for depth, name in enumerate(names):
        node = scope.get(name)
        if node is None:
            return f"{name} not defined"
        assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        # Nearer decorators win over class and module marks.
        decorators = list(node.decorator_list) + decorators
        if depth < len(names) - 1:
            if not isinstance(node, ast.ClassDef):
                return f"{name} is not a class"
            scope = _defined_names(node.body)
    if story_phase is None:
        return None
    marks = _marks(decorators)
    for bad in ("skip", "xfail"):
        if bad in marks:
            return f"marked {bad}"
    test_phase = _phase_of(marks.get("phase"))
    if test_phase is None:
        return "no phase marker"
    if test_phase > story_phase:
        return f"marked phase {test_phase}, after the story's phase {story_phase}"
    return None


def find_missing(plan: Plan, phase: int, root: Path = ROOT, only: bool = False) -> list[Missing]:
    missing = []
    for story in plan.stories:
        if story.phase > phase or (only and story.phase != phase):
            continue
        for ref in story.tests:
            why = resolve(ref, root, story.phase)
            if why:
                missing.append(Missing(story.id, ref, why))
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that the plan's named tests exist.")
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--only", action="store_true", help="check phase N alone, not 0..N")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args(argv)

    plan = load_plan(args.plan)
    missing = find_missing(plan, args.phase, only=args.only)
    checked = [
        s
        for s in plan.stories
        if s.phase <= args.phase and (not args.only or s.phase == args.phase)
    ]
    refs = sum(len(s.tests) for s in checked)
    if missing:
        print(f"{len(missing)} of {refs} planned tests are missing:")
        for m in missing:
            print(f"  {m.story}  {m.ref}  ({m.why})")
        return 1
    print(f"all {refs} planned tests exist across {len(checked)} stories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
