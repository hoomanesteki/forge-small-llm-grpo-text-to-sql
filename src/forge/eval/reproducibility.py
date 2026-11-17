"""Invariant I6: a greedy evaluation gives the same result when it is run again.

Two runs of one model with one config, on the same data and code, must agree within
``TOLERANCE_POINTS`` of official EX on the samples both scored (plan section 4, FG-307).
They may differ in ``limit`` alone: a run of the first 100 val questions repeats the start
of a full val run, since both take the questions in id order. ``compare_runs`` checks
that the two runs are such a pair, then says how far apart they are and on which samples
their verdicts and replies differ. It works on stored runs, so the same check covers two
CPU runs in a test and two GPU runs committed under ``docs/runs/``.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.core.types import RunManifest
from forge.eval.run_files import Key, StoredRun, key

TOLERANCE_POINTS = 0.5
# Settings a repeat may change: how many questions it answers, and whether its push went
# over the week's GPU budget, which changes no score.
_MAY_DIFFER = frozenset({"limit", "allow_over_budget"})


@dataclass(frozen=True)
class Reproducibility:
    """Two runs of one eval, compared on the samples both scored."""

    runs: tuple[str, str]
    samples: int
    hits: tuple[int, int]  # official EX matches of each run on those samples
    verdicts_differ: tuple[Key, ...]  # samples one run matches and the other doesn't
    replies_differ: tuple[Key, ...]  # samples whose reply text differs

    @property
    def official_ex(self) -> tuple[float, float]:
        """Each run's official EX on the compared samples, in percent."""
        return (100 * self.hits[0] / self.samples, 100 * self.hits[1] / self.samples)

    @property
    def gap_points(self) -> float:
        return abs(self.official_ex[0] - self.official_ex[1])

    @property
    def ok(self) -> bool:
        # In whole matches, so a gap of exactly the tolerance isn't lost to float error.
        return abs(self.hits[0] - self.hits[1]) * 100 <= TOLERANCE_POINTS * self.samples

    def describe(self) -> str:
        a, b = self.official_ex
        return (
            f"runs {self.runs[0]} and {self.runs[1]}: official EX {a:.1f}% and {b:.1f}% on "
            f"the {self.samples:,} samples both scored, {self.gap_points:.1f} points apart "
            f"(I6 allows {TOLERANCE_POINTS}); {len(self.verdicts_differ):,} verdicts and "
            f"{len(self.replies_differ):,} replies differ"
        )


def compare_runs(a: StoredRun, b: StoredRun) -> Reproducibility:
    """Compare two stored runs of the same eval, as I6 does.

    Raises ``ValueError`` unless they are the same model, adapter, commit, data and config
    but for ``limit``, and one run's samples are among the other's.
    """
    _check_same_eval(a.manifest, b.manifest)
    scores_a = {key(score): score for score in a.scores}
    scores_b = {key(score): score for score in b.scores}
    if not (scores_a.keys() <= scores_b.keys() or scores_b.keys() <= scores_a.keys()):
        raise ValueError(
            f"runs {a.manifest.run_id} and {b.manifest.run_id} answered different questions: "
            "a repeat answers the same ones, or the first of them"
        )
    shared = sorted(scores_a.keys() & scores_b.keys())
    hits = (
        sum(scores_a[k].ex_official is True for k in shared),
        sum(scores_b[k].ex_official is True for k in shared),
    )
    return Reproducibility(
        runs=(a.manifest.run_id, b.manifest.run_id),
        samples=len(shared),
        hits=hits,
        verdicts_differ=tuple(
            k for k in shared if scores_a[k].ex_official != scores_b[k].ex_official
        ),
        replies_differ=tuple(k for k in shared if a.generations[k].text != b.generations[k].text),
    )


def _check_same_eval(a: RunManifest, b: RunManifest) -> None:
    pair = f"runs {a.run_id} and {b.run_id}"
    for field in ("model_ref", "adapter_ref", "git_sha", "data_manifest_hash"):
        if getattr(a, field) != getattr(b, field):
            raise ValueError(
                f"{pair} differ in {field} ({getattr(a, field)} and {getattr(b, field)}), "
                "so they are not one eval run twice"
            )
    settings = set(a.config) | set(b.config)
    changed = sorted(
        name for name in settings - _MAY_DIFFER if a.config.get(name) != b.config.get(name)
    )
    if changed:
        raise ValueError(f"{pair} differ in {', '.join(changed)}: a repeat changes only the limit")
