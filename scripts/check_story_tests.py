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


def _defined_names(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def resolve(ref: str, root: Path) -> str | None:
    """Return None when the test exists, or a short reason when it doesn't."""
    path_part, *names = ref.split("::")
    path = root / path_part
    if not path.is_file():
        return "file not found"
    scope: dict[str, ast.AST] = _defined_names(ast.parse(path.read_text(encoding="utf-8")))
    for depth, name in enumerate(names):
        node = scope.get(name)
        if node is None:
            return f"{name} not defined"
        if depth < len(names) - 1:
            if not isinstance(node, ast.ClassDef):
                return f"{name} is not a class"
            scope = {
                n.name: n
                for n in node.body
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            }
    return None


def find_missing(plan: Plan, phase: int, root: Path = ROOT, only: bool = False) -> list[Missing]:
    missing = []
    for story in plan.stories:
        if story.phase > phase or (only and story.phase != phase):
            continue
        for ref in story.tests:
            why = resolve(ref, root)
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
