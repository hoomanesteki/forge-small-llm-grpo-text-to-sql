"""Apply ADR-0012's rule to the candidates' val runs, as stored and with cut queries read whole.

Usage: FORGE_DATA_DIR=data/spider uv run python scripts/check_choice_extraction.py RUN [RUN ...]

Each RUN is a candidate's stored val run with prompt p1, its model one of the candidates
configs/models.yaml pins (``--models`` names another file). A reply with no ```sql block
is read by rule 3 of plan section 3.7, which keeps one line, so a query written over
several lines loses the rest (ADR-0007). The rule is applied twice: to the stored scores,
and to the same runs with each query rule 3 cut read whole, up to a blank line or a code
fence, and scored as a run scores a query (``read_whole`` in
scripts/count_pilot_extraction.py). For each reading it prints each candidate's official
EX with its interval and whether it leaves the headroom rule 2 asks for, the best eligible
candidate against the next as ``forge report compare`` compares two runs (the paired
change in points and McNemar's test on gold clusters), and the model the rule chooses.
Rule 2's other half, FG-301's GRPO memory check, passed for every candidate (ADR-0012,
Measured), so it isn't checked here. ADR-0012 quotes the output.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from count_pilot_extraction import read_whole

from forge.core.types import Score
from forge.data.spider import resolve_data_dir
from forge.eval.metrics import Estimate, Rate, compare, summarize
from forge.eval.report import ReportRun, _p, _points, _rate, load_run
from forge.eval.run_files import key

MODELS = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
# ADR-0012's rule: a candidate leaves visible headroom below this val official EX (rule 2),
# and the best two differ significantly when McNemar's p is below ALPHA (rule 3).
HEADROOM = 0.75
ALPHA = 0.05


@dataclass(frozen=True)
class Candidate:
    name: str  # the model's Hub repo
    params: int  # its safetensors parameters, as configs/models.yaml gives them
    run: ReportRun


@dataclass(frozen=True)
class Test:
    """The best eligible candidate against the next, as ``forge report compare`` puts it."""

    delta: Estimate  # the best's official EX less the next's, paired
    pvalue: float  # exact McNemar on ``unit``s
    unit: str


@dataclass(frozen=True)
class Choice:
    """What ADR-0012's rule makes of one reading of the candidates' scores."""

    rates: Mapping[str, Rate]  # official EX, by candidate
    ranked: tuple[str, ...]  # the eligible candidates, best first
    test: Test | None  # with two or more eligible
    chosen: str | None  # None when no candidate is eligible (rule 5)


def load_candidates(folders: Sequence[Path], *, models: Path, data: Path) -> list[Candidate]:
    """Each run in ``folders`` as a candidate, refusing a run that isn't a candidate's
    p1 val run or that a report couldn't read."""
    entries = yaml.safe_load(models.read_text(encoding="utf-8"))["candidates"].values()
    pinned = {f"{entry['repo']}@{entry['revision']}": entry for entry in entries}
    found = []
    for folder in folders:
        run = load_run(folder.name, runs_root=folder.parent, data_dir=data)
        model = run.stored.manifest.model_ref
        if model not in pinned:
            raise ValueError(f"run {run.run_id}'s model {model} is not a candidate in {models}")
        if (run.options.split, run.options.prompt_version) != ("val", "p1"):
            raise ValueError(
                f"run {run.run_id} answered {run.options.split} with prompt "
                f"{run.options.prompt_version}; rule 1 reads each candidate's p1 val run"
            )
        found.append(Candidate(pinned[model]["repo"], pinned[model]["params"], run))
    return found


def read_whole_scores(candidate: Candidate, data: Path) -> tuple[list[Score], int]:
    """The candidate's scores with each query rule 3 cut scored whole instead, and how
    many it cut."""
    run = candidate.run
    whole = {key(item.score): item.score for item in read_whole(run.stored, run.examples, data)}
    return [whole.get(key(score), score) for score in run.stored.scores], len(whole)


def apply_rule(candidates: Sequence[Candidate], scores: Mapping[str, Sequence[Score]]) -> Choice:
    """ADR-0012's rule on ``scores``, each candidate's by its name."""
    by_name = {candidate.name: candidate for candidate in candidates}
    rates = {}
    for candidate in candidates:
        official = summarize(scores[candidate.name], clusters=candidate.run.clusters).ex_official
        if official is None:
            raise ValueError(f"run {candidate.run.run_id} has samples without official EX")
        rates[candidate.name] = official
    eligible = [name for name, rate in rates.items() if rate.value < HEADROOM]
    ranked = tuple(sorted(eligible, key=lambda name: (-rates[name].value, by_name[name].params)))
    if len(ranked) < 2:
        return Choice(rates, ranked, None, ranked[0] if ranked else None)
    best, second = by_name[ranked[0]], by_name[ranked[1]]
    comparison = compare(scores[second.name], scores[best.name], clusters=best.run.clusters)
    pvalue = comparison.mcnemar.pvalue
    if pvalue is None:
        raise ValueError("McNemar's test needs one sample a question, as rule 3 compares them")
    if pvalue >= ALPHA:
        chosen = min((best, second), key=lambda candidate: candidate.params).name
    else:
        chosen = best.name
    test = Test(comparison.delta, pvalue, comparison.mcnemar.unit)
    return Choice(rates, ranked, test, chosen)


def describe(
    title: str,
    candidates: Sequence[Candidate],
    choice: Choice,
    read_whole_counts: Mapping[str, int] | None = None,
) -> list[str]:
    lines = [f"{title}:"]
    for candidate in candidates:
        rate = choice.rates[candidate.name]
        eligible = "eligible" if candidate.name in choice.ranked else "no headroom"
        line = (
            f"- {candidate.name} ({candidate.params / 1e9:.2f}B parameters): "
            f"official EX {_rate(rate)}, {eligible}"
        )
        if read_whole_counts is not None:
            line += f", {read_whole_counts[candidate.name]} queries read whole"
        lines.append(line)
    if choice.test is not None:
        best, second = choice.ranked[:2]
        delta = choice.test.delta
        lines.append(
            f"- {best} against {second}: {_points(delta.value)} points "
            f"[{_points(delta.low)}, {_points(delta.high)}], McNemar {_p(choice.test.pvalue)} "
            f"on {choice.test.unit}s"
        )
    if choice.chosen is None:
        lines.append("- chosen: none, as no candidate is eligible (rule 5)")
    elif choice.test is None:
        lines.append(f"- chosen: {choice.chosen}, the only eligible candidate")
    elif choice.chosen == choice.ranked[0]:
        lines.append(f"- chosen: {choice.chosen}, the best, ahead of the next by p below {ALPHA}")
    else:
        lines.append(
            f"- chosen: {choice.chosen}, the one with fewer parameters, as the best two "
            f"don't differ significantly (p of {ALPHA} or more)"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="each candidate's stored p1 val run")
    parser.add_argument("--models", type=Path, default=MODELS, help="the file pinning them")
    args = parser.parse_args(argv)
    data = resolve_data_dir().resolve()
    try:
        candidates = load_candidates(args.runs, models=args.models, data=data)
        stored = {c.name: list(c.run.stored.scores) for c in candidates}
        read = {c.name: read_whole_scores(c, data) for c in candidates}
        whole = {name: scores for name, (scores, _) in read.items()}
        cut = {name: count for name, (_, count) in read.items()}
        lines = [
            *describe("As stored", candidates, apply_rule(candidates, stored)),
            *describe(
                "With the queries rule 3 cut read whole",
                candidates,
                apply_rule(candidates, whole),
                cut,
            ),
        ]
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
