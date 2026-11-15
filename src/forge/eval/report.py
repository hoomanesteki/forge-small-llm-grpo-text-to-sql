"""Reports (stage 10): what a run scored, from its stored files only.

Nothing in a report comes from a run's memory: ``load_run`` reads

- the run's manifest, scores and generations, through ``read_run``, which takes only a
  finished run whose files agree with each other and with the manifest;
- the questions and gold queries of its split, from data whose data manifest hash is
  the manifest's. Other data is refused.

Rates are official EX unless labeled fast (ADR-0008), so every sample needs an official
verdict. Intervals come from ``forge.eval.metrics``, with the questions that share a gold
query drawn together (``gold_clusters``).

A dev run's report shows the model's answers on dev, which belongs to the look the run
logged (ADR-0004): it needs the run's row in ``docs/dev-usage.md`` and adds none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from forge.core.types import Example, GenParams, Score, Split
from forge.data.manifest import data_manifest_hash
from forge.data.spider import load_raw, resolve_data_dir
from forge.data.splits import split_examples
from forge.eval.config import check_run_id
from forge.eval.metrics import Rate, Summary, gold_clusters, summarize
from forge.eval.run_files import RunCorrupt, StoredRun, read_run
from forge.eval.taxonomy import ERROR_CLASSES
from forge.ops.dev_usage import read_uses
from forge.ops.paths import dev_usage_path, runs_dir


class RunOptions(BaseModel, frozen=True):
    """What a report shows of a run's options, from its manifest's config."""

    backend: str
    split: Split
    limit: int | None
    prompt_version: str
    generation: GenParams


@dataclass(frozen=True)
class ReportRun:
    """A finished run with what a report adds to its files."""

    stored: StoredRun
    options: RunOptions
    examples: Mapping[str, Example]  # the questions it scored, by id
    clusters: Mapping[str, str]  # gold_clusters of those questions
    summary: Summary
    official: Rate  # summary.ex_official, which every report leads with

    @property
    def run_id(self) -> str:
        return self.stored.manifest.run_id


@dataclass(frozen=True)
class _Data:
    """The data reports read questions from, and the hash a run must have been made on."""

    root: Path
    hash: str
    splits: Mapping[Split, Sequence[Example]]


def load_run(
    run_id: str, *, runs_root: Path | None = None, data_dir: Path | None = None
) -> ReportRun:
    """The finished run ``run_id`` with the questions it answered, as a report reads it.

    Raises what ``read_run`` raises, ``ValueError`` for a run made on other data or a dev
    run whose look isn't logged, and ``RunCorrupt`` for scores a finished run can't hold.
    """
    return _load(run_id, _data(data_dir), runs_root or runs_dir())


def _data(data_dir: Path | None) -> _Data:
    root = resolve_data_dir(data_dir).resolve()
    return _Data(root, data_manifest_hash(root), split_examples(load_raw(root)))


def _load(run_id: str, data: _Data, runs_root: Path) -> ReportRun:
    """A finished run with its questions, once its scores fit the data and each other."""
    check_run_id(run_id)
    stored = read_run(runs_root / run_id)
    manifest = stored.manifest
    try:
        options = RunOptions.model_validate(manifest.config)
    except ValidationError as exc:
        raise RunCorrupt(f"run {run_id}'s config lacks what a report shows: {exc}") from exc
    if manifest.data_manifest_hash != data.hash:
        raise ValueError(
            f"run {run_id} was made on other data: its data manifest hash is "
            f"{manifest.data_manifest_hash[:12]}, and {data.root} hashes to {data.hash[:12]}; "
            "point FORGE_DATA_DIR at the data it used"
        )
    if options.split == "dev":
        _check_dev_use(run_id)
    questions = {example.id: example for example in data.splits[options.split]}
    examples: dict[str, Example] = {}
    for score in stored.scores:
        _check_score(run_id, score)
        if score.example_id not in questions:
            raise RunCorrupt(f"run {run_id} scored {score.example_id}, which {options.split} lacks")
        examples[score.example_id] = questions[score.example_id]
    clusters = gold_clusters(examples.values())
    summary = summarize(stored.scores, clusters=clusters)
    return ReportRun(stored, options, examples, clusters, summary, _official(summary.ex_official))


def _check_dev_use(run_id: str) -> None:
    path = dev_usage_path()
    if not any(logged == run_id for _, logged, _ in read_uses(path)):
        raise ValueError(
            f"run {run_id} answered dev questions, but {path} has no row for it; its report "
            "is part of the look the run logged (ADR-0004), so the row must be there"
        )


def _check_score(run_id: str, score: Score) -> None:
    """Refuse a score a finished run can't hold, since the report would misread it."""
    sample = f"run {run_id}'s {score.example_id} sample {score.sample_idx}"
    if score.ex_official is None:
        raise RunCorrupt(f"{sample} has no official verdict, though the run finished")
    if score.error_class is not None and score.error_class not in ERROR_CLASSES:
        raise RunCorrupt(f"{sample} has the unknown error class {score.error_class!r}")
    # A run classifies each miss of fast EX and nothing else (forge.eval.scoring).
    if score.ex_fast and score.error_class is not None:
        raise RunCorrupt(f"{sample} matches by fast EX but has the error class {score.error_class}")
    if not score.ex_fast and score.error_class is None:
        raise RunCorrupt(f"{sample} misses by fast EX but has no error class")


def _official(rate: Rate | None) -> Rate:
    if rate is None:
        raise RunCorrupt("a score has no official verdict")
    return rate
