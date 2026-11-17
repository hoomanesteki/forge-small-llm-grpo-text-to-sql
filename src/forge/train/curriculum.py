"""The GRPO dataset (FG-505): which train questions GRPO trains on.

GRPO learns from the spread of rewards within a group. A prompt the starting model always
solves, or never does, gives every reply of its group the same reward and so no
advantage, and the GPU time spent on it teaches nothing. The filter deals with that.

``forge data build-grpo --filter-from <run_id>`` reads a finished eval run of the
starting model on train questions, a subset of them such as the first 3,000 (``forge eval
run --split train --limit 3000``), made greedy, one reply a question and no retry, so each
score is the model's one best answer (``check_filter_run``). Of the questions it scored
(``select_rows``):

- those on a rowless database (``ROWLESS_DBS``) are left out, as everywhere in GRPO,
  since any query with an empty result matches their gold one;
- so are those whose gold query took longer than ``MAX_GOLD_MS`` when the train gold
  cache was built (``forge.eval.gold_cache``): a right reply about as slow risks the
  rewards' 2 s cap, and slow queries hold a step up;
- every question the reply got wrong stays;
- of those it got right, ``KEEP_SOLVED`` stay, rounded to the nearest whole question,
  against forgetting what the model knows: the ones whose ids hash lowest with the seed,
  so a seed keeps the same ones on any machine.

Right and wrong are fast EX (``Score.ex_fast``), the comparator the execution reward uses
(ADR-0008), since the reward is what a group's spread is made of.

The rows go to ``data/grpo/<run_id>/`` (``save_filter``): ``rows.jsonl``, a line for each
question kept, and ``summary.json`` beside it, with what was kept and left out by
hardness and the hash of it all. They are kept rather than picked again when a run
starts, because gold timings differ from one machine's cache to the next and a resumed
run must train on the rows it started with. ``read_filter`` reads them back, checked
against their hash and the data they were picked on. Only train is read: the eval run
must be of train questions, each of them in ``load_train()``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, ValidationError

from forge.core.hashing import canonical_json, sha256_hex
from forge.core.types import Example, ExecResult, Hardness
from forge.data.manifest import data_manifest_hash
from forge.data.spider import ROWLESS_DBS, resolve_data_dir
from forge.data.splits import assert_train_only, load_train
from forge.eval.config import check_run_id
from forge.eval.gold_cache import split_gold
from forge.eval.run_files import ADAPTER_DIGEST, StoredRun, read_run
from forge.ops.paths import repo_root, runs_dir

log = logging.getLogger(__name__)

KEEP_SOLVED = 0.3  # the share of the questions the eval solved that stay
MAX_GOLD_MS = 1000.0  # gold slower than this, in the gold cache, is left out
ROWS_FILE = "rows.jsonl"
SUMMARY_FILE = "summary.json"
LEVELS: tuple[Hardness, ...] = get_args(Hardness)


class FilterRow(BaseModel, frozen=True):
    """A question the filter kept, and what it knew of it."""

    example_id: str
    db_id: str
    hardness: Hardness
    solved: bool  # the eval's reply matched the gold result (fast EX)
    gold_ms: float  # the gold query's time in the train gold cache


class Tally(BaseModel, frozen=True):
    """What became of the scored questions of one hardness level, or of all of them."""

    scored: int
    rowless: int  # on a rowless database: left out
    slow_gold: int  # gold slower than max_gold_ms: left out
    wrong: int  # of the rest, those the reply got wrong: all kept
    solved: int  # of the rest, those it got right
    solved_kept: int  # of those, the ones kept
    kept: int  # wrong and solved_kept


class FilterSummary(BaseModel, frozen=True):
    """``summary.json``: where the rows came from, how they were picked, and what was
    kept by hardness."""

    hash: str  # of everything else here and the rows
    eval_run: str
    model_ref: str
    adapter_ref: str | None
    adapter_sha256: str | None  # the eval's adapter files, as its manifest records them
    data_manifest_hash: str
    keep_solved: float
    max_gold_ms: float
    seed: int
    by_hardness: dict[Hardness, Tally]
    total: Tally
    slow_gold_ids: list[str]


@dataclass(frozen=True)
class GrpoFilter:
    """The questions the filter kept, in the eval's order, and their summary."""

    rows: list[FilterRow]
    summary: FilterSummary


def default_root() -> Path:
    return repo_root() / "data" / "grpo"


def build_filter(
    run_id: str,
    *,
    seed: int = 0,
    runs_root: Path | None = None,
    data_dir: Path | None = None,
    cache_root: Path | None = None,
) -> GrpoFilter:
    """The questions the filter keeps of eval run ``run_id``'s, as the module docstring
    says, with ``seed`` drawing the solved ones that stay.

    The run is read from ``runs_root`` (``runs/``) and the gold timings from the train gold
    cache under ``cache_root`` (``forge.ops.paths.cache_dir()``), built if missing. Raises
    as ``read_run`` does for a run that isn't a finished eval run, and ``ValueError`` for
    one the filter can't read or a question train doesn't hold here.
    """
    check_run_id(run_id)
    root = resolve_data_dir(data_dir)
    stored = read_run((runs_dir() if runs_root is None else runs_root) / run_id)
    check_filter_run(stored, data_manifest_hash(root))
    gold = split_gold("train", data_dir=root, cache_root=cache_root).results
    return select_rows(stored, load_train(root), gold, seed=seed)


def check_filter_run(stored: StoredRun, data_hash: str) -> None:
    """Raise ``ValueError`` unless ``stored`` is an eval the filter can read: of train
    questions, greedy with one reply a question and no retry, on the data ``data_hash``
    names."""
    manifest = stored.manifest
    run, config = manifest.run_id, manifest.config
    split = config.get("split")
    if split != "train":
        raise ValueError(
            f"run {run} evaluated {split}, and the GRPO filter reads an eval of train "
            "questions only: training never reads val or dev (ADR-0004)"
        )
    generation = config.get("generation", {})
    temperature, n = generation.get("temperature"), generation.get("n")
    if temperature != 0 or n != 1:
        raise ValueError(
            f"run {run} drew {n} replies a question at temperature {temperature}, and the "
            "filter needs the model's greedy reply alone: --temperature 0 --n 1"
        )
    if config.get("retry"):
        raise ValueError(
            f"run {run} retried the queries that failed, so its scores are of second "
            "replies, and the filter needs first ones: --retry 0"
        )
    if manifest.data_manifest_hash != data_hash:
        raise ValueError(
            f"run {run} was made on data {manifest.data_manifest_hash[:12]}, and this data is "
            f"{data_hash[:12]}, so its example ids may name other questions"
        )


def select_rows(
    stored: StoredRun,
    train: Sequence[Example],
    gold: Mapping[str, ExecResult],
    *,
    seed: int = 0,
) -> GrpoFilter:
    """The questions the filter keeps of those ``stored`` scored, as the module docstring
    says: ``train`` is ``load_train()``'s questions, and ``gold`` their gold results.

    Raises ``ValueError`` for a score of a question that train doesn't hold or that has
    no gold result, and ``SplitLeak`` should a scored question not be a train one.
    """
    run = stored.manifest.run_id
    examples = {example.id: example for example in train}
    missing = [s.example_id for s in stored.scores if s.example_id not in examples]
    if missing:
        raise ValueError(
            f"run {run} scored {len(missing)} questions train doesn't hold here, such as "
            f"{missing[0]}"
        )
    scored = [examples[score.example_id] for score in stored.scores]
    assert_train_only(scored)
    fates: dict[str, str] = {}  # example id -> rowless, slow_gold, wrong or solved
    for example, score in zip(scored, stored.scores, strict=True):
        result = gold.get(example.id)
        if result is None:
            raise ValueError(f"{example.id} has no result in the train gold cache")
        if example.db_id in ROWLESS_DBS:
            fates[example.id] = "rowless"
        elif result.ms > MAX_GOLD_MS:
            fates[example.id] = "slow_gold"
        else:
            fates[example.id] = "solved" if score.ex_fast else "wrong"
    solved = [example.id for example in scored if fates[example.id] == "solved"]
    stay = set(lowest_hashes(solved, round_share(KEEP_SOLVED, len(solved)), seed))
    rows = [
        FilterRow(
            example_id=example.id,
            db_id=example.db_id,
            hardness=example.hardness,
            solved=fates[example.id] == "solved",
            gold_ms=gold[example.id].ms,
        )
        for example in scored
        if fates[example.id] == "wrong" or example.id in stay
    ]
    by_hardness = {
        level: _tally([e for e in scored if e.hardness == level], fates, stay) for level in LEVELS
    }
    manifest = stored.manifest
    summary = FilterSummary(
        hash="",
        eval_run=run,
        model_ref=manifest.model_ref,
        adapter_ref=manifest.adapter_ref,
        adapter_sha256=manifest.outputs.get(ADAPTER_DIGEST),
        data_manifest_hash=manifest.data_manifest_hash,
        keep_solved=KEEP_SOLVED,
        max_gold_ms=MAX_GOLD_MS,
        seed=seed,
        by_hardness=by_hardness,
        total=_tally(scored, fates, stay),
        slow_gold_ids=[e.id for e in scored if fates[e.id] == "slow_gold"],
    )
    return GrpoFilter(rows, summary.model_copy(update={"hash": filter_hash(summary, rows)}))


def _tally(examples: Sequence[Example], fates: Mapping[str, str], stay: set[str]) -> Tally:
    counts = Counter(fates[example.id] for example in examples)
    solved_kept = sum(1 for example in examples if example.id in stay)
    return Tally(
        scored=len(examples),
        rowless=counts["rowless"],
        slow_gold=counts["slow_gold"],
        wrong=counts["wrong"],
        solved=counts["solved"],
        solved_kept=solved_kept,
        kept=counts["wrong"] + solved_kept,
    )


def round_share(share: float, count: int) -> int:
    """``share`` of ``count``, rounded half up, the share read as the decimal it is written
    as, so float error can't move a boundary: 0.3 of 15 is 5."""
    return math.floor(Fraction(repr(share)) * count + Fraction(1, 2))


def lowest_hashes(example_ids: Sequence[str], size: int, seed: int) -> list[str]:
    """The ``size`` ids that hash lowest with ``seed``, in their order: a hash rather than
    a random draw, so they depend on no library's generator nor on the ids' order."""
    ranked = sorted(example_ids, key=lambda example_id: _rank(f"{seed}:{example_id}"))
    chosen = set(ranked[:size])
    return [example_id for example_id in example_ids if example_id in chosen]


def filter_hash(summary: FilterSummary, rows: Sequence[FilterRow]) -> str:
    """sha256 of the canonical JSON of the summary but its hash, and of the rows."""
    payload = {"summary": summary.model_dump(exclude={"hash"}), "rows": list(rows)}
    return sha256_hex(canonical_json(payload))


def save_filter(found: GrpoFilter, root: Path | None = None) -> Path:
    """Write the rows and their summary to ``<root>/<eval run id>/`` (default
    ``data/grpo``) and return that folder.

    A folder already there with the same hash holds these same rows, so it is kept as it
    is; one with another hash, the rows of other settings or other data, is refused. The
    files are written to a temporary folder beside it first, so a failed write never
    leaves half of them under the run's name.
    """
    root = default_root() if root is None else root
    target = root / found.summary.eval_run
    if target.exists():
        there = _read_folder(target).summary.hash
        if there != found.summary.hash:
            raise ValueError(
                f"{target} holds other rows of run {found.summary.eval_run} (hash {there}, "
                f"not {found.summary.hash}); remove it to build them again"
            )
        log.info("%s is already built", target)
        return target
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".grpo-", dir=root) as scratch:
        staged = Path(scratch) / target.name
        staged.mkdir()
        lines = "".join(row.model_dump_json() + "\n" for row in found.rows)
        (staged / ROWS_FILE).write_text(lines, encoding="utf-8")
        (staged / SUMMARY_FILE).write_text(
            found.summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        staged.rename(target)
    return target


def read_filter(
    run_id: str, *, root: Path | None = None, data_dir: Path | None = None
) -> GrpoFilter:
    """The rows ``save_filter`` kept for eval run ``run_id`` under ``root`` (default
    ``data/grpo``), checked against their hash and the data in ``data_dir``.

    Raises ``FileNotFoundError`` when they were never built here, and ``ValueError`` for
    files that can't be read, don't hash to their summary's hash, or were picked on
    other data.
    """
    check_run_id(run_id)
    folder = (default_root() if root is None else root) / run_id
    if not (folder / SUMMARY_FILE).is_file():
        raise FileNotFoundError(
            f"no GRPO rows of eval run {run_id} in {folder}: build them with "
            f"forge data build-grpo --filter-from {run_id}"
        )
    found = _read_folder(folder)
    if found.summary.eval_run != run_id:
        raise ValueError(f"{folder} holds the rows of run {found.summary.eval_run}, not {run_id}")
    data_hash = data_manifest_hash(resolve_data_dir(data_dir))
    if found.summary.data_manifest_hash != data_hash:
        raise ValueError(
            f"the GRPO rows in {folder} were picked on data "
            f"{found.summary.data_manifest_hash[:12]}, and this data is {data_hash[:12]}; "
            "build them again on this data"
        )
    return found


def _read_folder(folder: Path) -> GrpoFilter:
    try:
        summary = FilterSummary.model_validate_json((folder / SUMMARY_FILE).read_bytes())
        lines = (folder / ROWS_FILE).read_bytes().splitlines()
        rows = [FilterRow.model_validate_json(line) for line in lines]
    except ValidationError as exc:
        raise ValueError(f"{folder} holds GRPO rows that can't be read: {exc}") from exc
    if filter_hash(summary, rows) != summary.hash:
        raise ValueError(
            f"{folder / ROWS_FILE} and its summary don't hash to the summary's hash; build "
            "them again"
        )
    return GrpoFilter(rows, summary)


def render_summary(summary: FilterSummary) -> str:
    """The summary's tallies as a table, a row a hardness level and one for all."""
    names = list(Tally.model_fields)
    table = [["hardness", *(name.replace("_", " ") for name in names)]]
    for level, tally in [*summary.by_hardness.items(), ("all", summary.total)]:
        table.append([level, *(str(getattr(tally, name)) for name in names)])
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    return "\n".join(
        row[0].ljust(widths[0])
        + "".join(f"  {cell:>{width}}" for cell, width in zip(row[1:], widths[1:], strict=True))
        for row in table
    )


def _rank(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
