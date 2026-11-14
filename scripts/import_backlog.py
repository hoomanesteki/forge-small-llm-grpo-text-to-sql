"""Create GitHub milestones, labels and one issue per plan story.

Idempotent: issues are matched by the story id at the start of their title, labels
are created with --force, and milestones are only created when missing.

    uv run python scripts/import_backlog.py --dry-run
    uv run python scripts/import_backlog.py --phase 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_parser import DEFAULT_PLAN, Plan, Story, load_plan

Runner = Callable[[Sequence[str]], str]

LABEL_COLORS = {
    "phase": "1d76db",
    "theme": "5319e7",
    "epic": "0e8a16",
    "pts": "c5def5",
    "type": "fbca04",
}


def gh(args: Sequence[str]) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def story_labels(story: Story, plan: Plan) -> list[str]:
    theme = plan.epics[story.epic].theme if story.epic in plan.epics else None
    labels = [
        f"phase:{story.phase}",
        f"epic:{story.epic}",
        f"pts:{story.points}",
        f"type:{story.kind}",
    ]
    # A GPU spike (FG-301) is both; the gpu label is what budget reviews filter on.
    if story.gpu and story.kind != "gpu":
        labels.append("type:gpu")
    if theme:
        labels.insert(1, f"theme:{theme}")
    return labels


def all_labels(plan: Plan) -> dict[str, str]:
    """Label name -> description, for every label any story uses."""
    out: dict[str, str] = {}
    names = {ph.number: ph.name for ph in plan.phases}
    for story in plan.stories:
        for label in story_labels(story, plan):
            kind, value = label.split(":", 1)
            if kind == "phase":
                out[label] = f"Phase {value}: {names.get(int(value), '')}".strip()
            elif kind == "epic":
                out[label] = plan.epics[value].name if value in plan.epics else value
            elif kind == "theme":
                out[label] = f"Theme {value}"
            elif kind == "pts":
                out[label] = f"{value} story points"
            else:
                out[label] = f"{value} work item"
    out.setdefault("type:fix", "fix work item")
    return out


def issue_body(story: Story) -> str:
    deps = ", ".join(story.depends_on) or "none"
    return (
        f"{story.body}\n\n---\nImported from `docs/PLAN.md`. Phase {story.phase} · "
        f"{story.points} pts · depends on {deps}."
    )


def import_backlog(
    plan: Plan, *, run: Runner = gh, dry_run: bool = False, phase: int | None = None
) -> list[str]:
    """Return the actions taken (or planned, for a dry run)."""
    stories = [s for s in plan.stories if phase is None or s.phase == phase]
    actions: list[str] = []

    existing_titles: list[str] = []
    existing_milestones: set[str] = set()
    if not dry_run:
        issues = json.loads(
            run(["issue", "list", "--state", "all", "--limit", "500", "--json", "title"])
        )
        existing_titles = [i["title"] for i in issues]
        milestones = json.loads(
            run(["api", "repos/{owner}/{repo}/milestones?state=all&per_page=100"])
        )
        existing_milestones = {m["title"] for m in milestones}
    existing_ids = {t.split()[0] for t in existing_titles if t.startswith("FG-")}

    for ph in plan.phases:
        if phase is not None and ph.number != phase:
            continue
        title = f"Phase {ph.number}"
        if title in existing_milestones:
            continue
        actions.append(f"milestone {title}: {ph.name}")
        if not dry_run:
            run(
                [
                    "api",
                    "repos/{owner}/{repo}/milestones",
                    "-f",
                    f"title={title}",
                    "-f",
                    f"description={ph.name}",
                ]
            )

    for label, description in sorted(all_labels(plan).items()):
        actions.append(f"label {label}")
        if not dry_run:
            color = LABEL_COLORS[label.split(":", 1)[0]]
            run(
                [
                    "label",
                    "create",
                    label,
                    "--color",
                    color,
                    "--description",
                    description,
                    "--force",
                ]
            )

    for story in stories:
        if story.id in existing_ids:
            continue
        actions.append(f"issue {story.id} {story.title} [{', '.join(story_labels(story, plan))}]")
        if not dry_run:
            args = [
                "issue",
                "create",
                "--title",
                f"{story.id} {story.title}",
                "--body",
                issue_body(story),
                "--milestone",
                f"Phase {story.phase}",
            ]
            for label in story_labels(story, plan):
                args += ["--label", label]
            run(args)
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import plan stories as GitHub issues.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", type=int)
    args = parser.parse_args(argv)

    plan = load_plan(args.plan)
    actions = import_backlog(plan, dry_run=args.dry_run, phase=args.phase)
    issues = [a for a in actions if a.startswith("issue ")]
    for a in actions:
        if not a.startswith("label "):
            print(("would create " if args.dry_run else "created ") + a)
    stories = [s for s in plan.stories if args.phase is None or s.phase == args.phase]
    print(
        f"{len(stories)} stories, {sum(s.points for s in stories)} points, "
        f"{len({s.phase for s in stories})} phases, {sum(s.gpu for s in stories)} GPU; "
        f"{len(issues)} new issues"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
