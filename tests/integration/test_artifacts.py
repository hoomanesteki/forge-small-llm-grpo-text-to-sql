"""Runs committed under docs/runs/: whole, made as their configs say, and rebuildable.

A GPU story's runs wait in ``docs/runs/pending.yaml`` while the accounts they need don't
exist. ``gpu_story_runs`` and ``kernel_training_runs`` find a story's runs by the configs
they were made from, and skip the checks on them only while the story is listed there;
``fg405_comparison`` does the same for FG-405's comparison of the main model.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
import yaml
from datasets import Dataset
from typer.testing import CliRunner

from forge.cli import app
from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.data.prompts import chat_token_counter, load_tokenizer
from forge.data.spider import FAILING_GOLD_IDS, ROWLESS_DBS, resolve_data_dir
from forge.data.splits import load_train, load_val, read_split_ids
from forge.eval.config import HOSTS, EvalConfig, read_config_file
from forge.eval.report import COMPARE_PREFIX, rebuild, write_run_report
from forge.eval.reproducibility import compare_runs
from forge.eval.run_files import ADAPTER_DIGEST, StoredRun, read_run
from forge.eval.runner import run_eval
from forge.infer.base import FakeBackend
from forge.ops.adapters import checkpoint_ref, digest_output
from forge.ops.dev_usage import read_uses
from forge.ops.hub import BEST, FINAL, LocalRunStore
from forge.ops.ledger import LedgerRow, append_once, read_rows, run_hours
from forge.ops.manifest import config_hash
from forge.train.config import load_train_config
from forge.train.curriculum import (
    KEEP_SOLVED,
    MAX_GOLD_MS,
    ROWS_FILE,
    SUMMARY_FILE,
    FilterRow,
    FilterSummary,
    GrpoFilter,
    filter_hash,
    lowest_hashes,
    round_share,
)
from forge.train.sft import PROMOTION_FILE, checkpoint_scores, fit_rows, promote_best
from forge.train.sft_data import build_sft_dataset
from tests.fakes import sft_run_folder
from tests.report_trace import GENERATED, NUMBER, untraced

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "docs" / "runs"
PENDING = RUNS / "pending.yaml"
EVAL_CONFIGS = ROOT / "configs" / "eval"
# The local pilot of FG-291 (ADR-0016), made from the commit its id ends with.
PILOT_RUN = "2026-11-15-qwen2-5-coder-0-5b-instruct-eval-da3eff7"
PILOT_CONFIG = ROOT / "configs" / "eval" / "pilot-qwen25coder05b-val.yaml"


@pytest.mark.phase(2)
def test_pilot_run_is_committed_whole() -> None:
    # read_run refuses a run that isn't finished, whose config doesn't hash to the
    # manifest's, or whose scores and generations don't pair up one to one.
    stored = read_run(RUNS / PILOT_RUN)
    manifest = stored.manifest
    assert (manifest.platform, manifest.config["device"]) == ("local-mps", "mps")
    # Made from a clean commit, the one in its id.
    assert re.fullmatch(r"[0-9a-f]{40}", manifest.git_sha), manifest.git_sha
    assert PILOT_RUN.endswith(manifest.git_sha[:7])

    # The committed pilot config, with the model pinned in configs/models.yaml.
    config = EvalConfig(**read_config_file(PILOT_CONFIG))
    pilot = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["pilot"]
    assert manifest.model_ref == config.model == f"{pilot['repo']}@{pilot['revision']}"
    for name in ("backend", "device", "split", "limit", "prompt_version", "retry"):
        assert manifest.config[name] == getattr(config, name), name
    assert manifest.config["generation"] == config.params.model_dump()

    # Every val question but the known failing golds, once each: val only, never dev.
    val = {example_id for example_id, _ in read_split_ids("val")} - FAILING_GOLD_IDS
    assert sorted(score.example_id for score in stored.scores) == sorted(val)

    report = stored.paths.report.read_text(encoding="utf-8")
    assert "\n| Platform | `local-mps` |\n" in report
    assert f"\n| Git commit | `{manifest.git_sha}` |\n" in report
    assert stored.paths.chart.is_file()
    assert f"({PILOT_RUN}/" in (RUNS / "README.md").read_text(encoding="utf-8")
    # Logged, on a platform that spends no GPU budget.
    rows = [
        row for row in read_rows(ROOT / "docs" / "compute-ledger.md") if row.run_id == PILOT_RUN
    ]
    assert [row.platform for row in rows] == ["local-mps"]


@pytest.mark.phase(2)
@pytest.mark.requires_full_data
def test_pilot_report_rebuilds_and_traces_to_its_files() -> None:
    data = resolve_data_dir()
    checks = rebuild(PILOT_RUN, runs_root=RUNS, data_dir=data)
    assert [check.path.name for check in checks] == ["val-report.md", "val-hardness.png"]
    report, chart = checks
    # The report as the stored files render it, with each of its numbers traced to them.
    assert (report.diff, report.skipped) == ((), None)
    assert untraced(report.path.read_text(encoding="utf-8"), RUNS / PILOT_RUN, data_dir=data) == []
    # The chart too, where this machine draws as the one that drew it did. Elsewhere, or
    # after a release of what draws it changes, its bytes can't be compared, and the test
    # says so instead of passing.
    if chart.skipped is not None:
        pytest.skip(f"the pilot's chart wasn't compared: {chart.skipped}")
    assert chart.diff == ()


# FG-491's SFT pilot on the laptop's GPU (ADR-0016): the run that trained the adapter,
# committed without its weights, and the adapter's val run, made with FG-291's config.
SFT_PILOT_RUN = "2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8"
SFT_PILOT_EVAL = "2026-11-16-qwen2-5-coder-0-5b-instruct-eval-db0d8d8"
SFT_PILOT_CONFIG = ROOT / "configs" / "sft" / "pilot-qwen25coder05b-mps.yaml"
# Where the adapter stays: the run's folder on the laptop that trained it.
SFT_PILOT_ADAPTER = f"runs/{SFT_PILOT_RUN}/{FINAL}"


LEDGER = ROOT / "docs" / "compute-ledger.md"
DEV_USAGE = ROOT / "docs" / "dev-usage.md"


def _ledger_rows(run_id: str, ledger: Path = LEDGER) -> list[LedgerRow]:
    return [row for row in read_rows(ledger) if row.run_id == run_id]


def _made_from_its_commit(manifest: RunManifest) -> None:
    # Made from a clean commit, the one its id ends with, on the laptop's GPU.
    assert re.fullmatch(r"[0-9a-f]{40}", manifest.git_sha), manifest.git_sha
    assert manifest.run_id.endswith(manifest.git_sha[:7])
    assert manifest.platform == "local-mps" and manifest.gpu
    # Logged, on a platform that spends no GPU budget, with the manifest's own hours to the
    # ledger's two places.
    assert manifest.ended_at is not None
    hours = run_hours(manifest.started_at, manifest.ended_at)
    (row,) = _ledger_rows(manifest.run_id)
    assert (row.platform, row.gpu) == ("local-mps", manifest.gpu)
    assert row.hours == pytest.approx(hours, abs=0.005)
    assert f"({manifest.run_id}/" in (RUNS / "README.md").read_text(encoding="utf-8")


@pytest.mark.phase(4)
def test_sft_pilot_run_is_committed_without_weights() -> None:
    root = RUNS / SFT_PILOT_RUN
    manifest = LocalRunStore(RUNS).get_manifest(SFT_PILOT_RUN)
    assert (manifest.kind, manifest.status) == ("sft", "finished")
    _made_from_its_commit(manifest)

    # The committed config, with the model pinned in configs/models.yaml, on the train split,
    # and a config hash over everything the manifest records.
    config = load_train_config(SFT_PILOT_CONFIG)
    assert config.sft is not None
    pilot = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["pilot"]
    assert manifest.model_ref == config.run.model_ref == f"{pilot['repo']}@{pilot['revision']}"
    sections = {"run", "data", "lora", "sft"}
    assert {name: manifest.config[name] for name in sections} == config.model_dump(
        mode="json", include=sections
    )
    assert manifest.config["data"]["split"] == "train"
    assert manifest.config["trainer"] == {"backend": "trl", "device": "mps", "dtype": "bfloat16"}
    assert manifest.config_hash == config_hash(manifest.config)
    stored = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    assert stored == manifest.config
    # The subset the config sets, of the rows that fit in max_length.
    sft_data = manifest.config["sft_data"]
    assert sft_data["trained"] == config.sft.subset_rows
    assert sft_data["trained"] <= sft_data["rows"] - sft_data["over_max_length"]

    # Its files, and not its weights: the adapter stays in runs/ where it was trained.
    assert manifest.outputs == {
        "config": "config.yaml",
        "metrics": "metrics.jsonl",
        "adapter": FINAL,
    }
    assert sorted(path.name for path in root.iterdir()) == [
        "config.yaml",
        "manifest.json",
        "metrics.jsonl",
    ]
    # A line every logging step to the last, each loss a number, then the trainer's summary.
    lines = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    steps = math.ceil(sft_data["trained"] / 16)
    every = config.sft.logging_steps
    losses = [line for line in lines if "loss" in line]
    assert [line["step"] for line in losses] == list(range(every, steps + 1, every))
    assert all(math.isfinite(line["loss"]) for line in losses)
    assert lines[-1]["step"] == steps and "train_runtime" in lines[-1]


@pytest.mark.phase(4)
@pytest.mark.requires_full_data
def test_sft_pilot_trained_on_the_rows_it_names() -> None:
    manifest = LocalRunStore(RUNS).get_manifest(SFT_PILOT_RUN)
    config = load_train_config(SFT_PILOT_CONFIG)
    assert config.sft is not None
    data = resolve_data_dir()
    assert manifest.data_manifest_hash == data_manifest_hash(data)
    # The rows forge data build-sft builds with the model's tokenizer, hashing as recorded.
    tokenizer = load_tokenizer(manifest.model_ref)
    built = build_sft_dataset(
        config.data.prompt_version, manifest.model_ref, chat_token_counter(tokenizer)
    )
    sft_data = manifest.config["sft_data"]
    assert (built.meta.hash, built.meta.counts.rows) == (sft_data["hash"], sft_data["rows"])
    fit = fit_rows(Dataset.from_list(list(built.rows)), tokenizer, config.sft, seed=config.run.seed)
    assert (fit.over_max_length, len(fit.rows)) == (
        sft_data["over_max_length"],
        sft_data["trained"],
    )
    # Train questions only: none from val's databases, which its adapter is scored on.
    trained = set(fit.rows["example_id"])
    assert trained <= {example.id for example in load_train()}
    assert not trained & {example_id for example_id, _ in read_split_ids("val")}


@pytest.mark.phase(4)
def test_sft_pilot_adapter_run_is_committed_whole() -> None:
    stored = read_run(RUNS / SFT_PILOT_EVAL)
    manifest = stored.manifest
    _made_from_its_commit(manifest)
    assert manifest.config["device"] == "mps"

    # The FG-291 pilot's settings, from its config, with the SFT run's final adapter: the
    # base run and this one differ in the adapter alone.
    config = replace(EvalConfig(**read_config_file(PILOT_CONFIG)), adapter=SFT_PILOT_ADAPTER)
    assert manifest.config == config.manifest_config()
    base = read_run(RUNS / PILOT_RUN).manifest
    assert manifest.config == {**base.config, "adapter": SFT_PILOT_ADAPTER}
    assert (manifest.model_ref, manifest.adapter_ref) == (base.model_ref, SFT_PILOT_ADAPTER)
    trained = LocalRunStore(RUNS).get_manifest(SFT_PILOT_RUN)
    assert trained.model_ref == manifest.model_ref
    assert trained.ended_at is not None and trained.ended_at <= manifest.started_at

    # Every val question but the known failing golds, once each: val only, never dev.
    val = {example_id for example_id, _ in read_split_ids("val")} - FAILING_GOLD_IDS
    assert sorted(score.example_id for score in stored.scores) == sorted(val)
    report = stored.paths.report.read_text(encoding="utf-8")
    assert "\n| Platform | `local-mps` |\n" in report
    assert f"\n| Git commit | `{manifest.git_sha}` |\n" in report
    assert f"\n| Adapter | `{SFT_PILOT_ADAPTER}` |\n" in report
    assert stored.paths.chart.is_file()
    # Its comparison with the base pilot, beside its report.
    assert (stored.paths.report.parent / f"compare-{PILOT_RUN}.md").is_file()


@pytest.mark.phase(4)
@pytest.mark.requires_full_data
def test_sft_pilot_reports_rebuild_and_trace_to_their_files() -> None:
    data = resolve_data_dir()
    checks = rebuild(SFT_PILOT_EVAL, runs_root=RUNS, data_dir=data)
    names = [check.path.name for check in checks]
    assert names == ["val-report.md", "val-hardness.png", f"compare-{PILOT_RUN}.md"]
    report, chart, comparison = checks
    assert (report.diff, report.skipped) == ((), None)
    assert (comparison.diff, comparison.skipped) == ((), None)
    text = report.path.read_text(encoding="utf-8")
    assert untraced(text, RUNS / SFT_PILOT_EVAL, data_dir=data) == []
    # The chart where this machine draws as the one that drew it did, as for FG-291's.
    if chart.skipped is not None:
        pytest.skip(f"the SFT pilot's chart wasn't compared: {chart.skipped}")
    assert chart.diff == ()


def pending_stories(pending: Path = PENDING) -> set[str]:
    entries = yaml.safe_load(pending.read_text(encoding="utf-8"))["pending"]
    return {entry["story"] for entry in entries}


def _made_from(config: Mapping[str, Any]) -> dict[str, Any]:
    # A push over the week's budget says so in its config, which changes no score.
    return {name: value for name, value in config.items() if name != "allow_over_budget"}


def gpu_story_runs(
    story: str,
    wanted: Mapping[str, Sequence[Path]],
    *,
    runs: Path = RUNS,
    pending: Path = PENDING,
) -> dict[str, StoredRun]:
    """The committed kernel run of each name in ``wanted``, made from one of its configs.

    Each must be under ``runs`` whole, with its report, and be the only run of its name.
    When any is missing the calling test skips while ``story`` is in ``pending``, and fails
    once it isn't: a story off that list has run its jobs.
    """
    found, missing = kernel_eval_runs(story, wanted, runs=runs)
    if missing:
        _missing(story, missing, runs=runs, pending=pending)
    return found


def kernel_eval_runs(
    story: str, wanted: Mapping[str, Sequence[Path]], *, runs: Path = RUNS
) -> tuple[dict[str, StoredRun], list[str]]:
    """``gpu_story_runs``'s runs, and the names of those ``runs`` lacks, without skipping."""
    store = LocalRunStore(runs)
    stored = [
        read_run(runs / run) for run in store.runs() if store.get_manifest(run).kind == "eval"
    ]
    kernel_runs = [run for run in stored if run.manifest.platform in HOSTS]
    found: dict[str, StoredRun] = {}
    missing: list[str] = []
    for name, configs in wanted.items():
        made = [EvalConfig(**read_config_file(path)).manifest_config() for path in configs]
        matches = [run for run in kernel_runs if _made_from(run.manifest.config) in made]
        ids = [run.manifest.run_id for run in matches]
        if len(matches) > 1:
            pytest.fail(f"{story}'s {name} is one run, but {runs} has {len(ids)}: {ids}")
        if not matches:
            missing.append(name)
            continue
        if not matches[0].paths.report.is_file():
            pytest.fail(f"run {ids[0]}, {story}'s {name}, is committed without its report")
        found[name] = matches[0]
    return found, missing


def _missing(story: str, names: Sequence[str], *, runs: Path, pending: Path) -> NoReturn:
    """Skip while ``story`` is in ``pending``, since its runs can't happen yet, and fail
    once it isn't: a story off that list has run its jobs."""
    if story in pending_stories(pending):
        pytest.skip(f"GPU run pending: {story}'s {', '.join(names)} ({pending.name})")
    pytest.fail(f"{story} is not in {pending.name}, but {runs} lacks its {', '.join(names)}")


@pytest.mark.phase(3)
def test_gpu_story_runs_skip_only_while_pending(spider_mini: Path, tmp_path: Path) -> None:
    runs, cache = tmp_path / "runs", tmp_path / "cache"
    config = tmp_path / "gold.yaml"
    config.write_text("model: gold\nbackend: fake\nlimit: 3\n", encoding="utf-8")
    other = tmp_path / "other.yaml"
    other.write_text("model: gold\nbackend: fake\nlimit: 4\n", encoding="utf-8")
    pending = tmp_path / "pending.yaml"
    pending.write_text("pending:\n  - story: FG-998\n    run: gold\n", encoding="utf-8")

    def make(run_id: str, host: str | None) -> None:
        options = {**read_config_file(config), "run_id": run_id, "host": host}
        run_eval(
            EvalConfig(**options), allow_dirty=True, runs_root=runs, cache_root=cache, workers=1
        )

    make("on-the-laptop", None)
    # Missing: skipped while the story is pending, failed once it isn't.
    with pytest.raises(pytest.skip.Exception, match=r"GPU run pending: FG-998's gold \("):
        gpu_story_runs("FG-998", {"gold": [config]}, runs=runs, pending=pending)
    with pytest.raises(pytest.fail.Exception, match=r"FG-999 is not in pending\.yaml, but"):
        gpu_story_runs("FG-999", {"gold": [config]}, runs=runs, pending=pending)

    # Only a kernel's run counts, and only with its report.
    make("on-kaggle", "kaggle")
    with pytest.raises(pytest.fail.Exception, match="on-kaggle, FG-999's gold, is committed"):
        gpu_story_runs("FG-999", {"gold": [config]}, runs=runs, pending=pending)
    write_run_report("on-kaggle", runs_root=runs)
    found = gpu_story_runs("FG-999", {"gold": [other, config]}, runs=runs, pending=pending)
    assert found["gold"].manifest.run_id == "on-kaggle"

    make("again", "kaggle")
    with pytest.raises(pytest.fail.Exception, match=r"gold is one run, but .* has 2"):
        gpu_story_runs("FG-999", {"gold": [config]}, runs=runs, pending=pending)


def _models() -> dict[str, Any]:
    models: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs" / "models.yaml").read_text(encoding="utf-8")
    )
    return models


CANDIDATES = list(_models()["candidates"])
(REFERENCE,) = _models()["reference"]
I6_MODEL = "qwen2-5-coder-1-5b"
# ADR-0012 leaves out a candidate whose val official EX leaves no visible headroom.
HEADROOM = 0.75
# FG-307's runs, each from one config, or for the chosen candidate from one of the
# candidates' (docs/runbook.md, "Baselines").
BASELINES = {
    **{f"{name} val": [EVAL_CONFIGS / f"{name}-val.yaml"] for name in CANDIDATES},
    "chosen val p2": [EVAL_CONFIGS / f"{name}-val-p2.yaml" for name in CANDIDATES],
    "reference val": [EVAL_CONFIGS / f"{REFERENCE}-val.yaml"],
    "I6 repeat": [EVAL_CONFIGS / f"{I6_MODEL}-val-100.yaml"],
    "chosen dev": [EVAL_CONFIGS / f"{name}-dev.yaml" for name in CANDIDATES],
    "reference dev": [EVAL_CONFIGS / f"{REFERENCE}-dev.yaml"],
}


def _official_ex(run: StoredRun) -> float:
    return sum(score.ex_official is True for score in run.scores) / len(run.scores)


def assert_p2_ran_for_the_chosen_model(runs: Mapping[str, StoredRun]) -> None:
    """Plan section 3.7 compares p1 and p2 for the chosen model (ADR-0007): the model
    ADR-0012's rule chose, which is the one evaluated on dev. That is the best eligible
    candidate, which needn't be the best p1 run, and it left headroom on val."""
    chosen_ref = runs["chosen dev"].manifest.model_ref
    assert runs["chosen val p2"].manifest.model_ref == chosen_ref
    (chosen_val,) = [
        runs[f"{name} val"]
        for name in CANDIDATES
        if runs[f"{name} val"].manifest.model_ref == chosen_ref
    ]
    assert _official_ex(chosen_val) < HEADROOM


@pytest.mark.phase(3)
def test_p2_is_checked_against_the_chosen_model() -> None:
    def run(model: str, ex: int) -> Any:
        """A stand-in for a stored run of ``model`` that matched ``ex`` of 10 questions."""
        scores = [SimpleNamespace(ex_official=i < ex) for i in range(10)]
        return SimpleNamespace(manifest=SimpleNamespace(model_ref=model), scores=scores)

    # The best p1 run scores 80%, past the headroom, so ADR-0012 chooses the next one.
    top, chosen, other = CANDIDATES
    runs = {
        f"{top} val": run("org/top", 8),
        f"{chosen} val": run("org/chosen", 6),
        f"{other} val": run("org/other", 4),
        "chosen dev": run("org/chosen", 5),
        "chosen val p2": run("org/chosen", 7),
    }
    assert_p2_ran_for_the_chosen_model(runs)
    # p2 on the best p1 run would fix the prompt on a model that is never trained.
    with pytest.raises(AssertionError):
        assert_p2_ran_for_the_chosen_model({**runs, "chosen val p2": run("org/top", 8)})
    # A chosen model without headroom breaks ADR-0012's rule.
    with pytest.raises(AssertionError):
        assert_p2_ran_for_the_chosen_model({**runs, f"{chosen} val": run("org/chosen", 9)})


@pytest.mark.phase(3)
def test_phase3_baseline_reports_present() -> None:
    runs = gpu_story_runs("FG-307", BASELINES)
    assert_p2_ran_for_the_chosen_model(runs)
    # I6 on the T4: the first 100 val questions again agree with the whole run's.
    i6 = compare_runs(runs["I6 repeat"], runs[f"{I6_MODEL} val"])
    assert i6.ok, i6.describe()
    # Each dev run spent one look, logged with its reason (dev looks 1 and 2 of 6).
    uses = read_uses(ROOT / "docs" / "dev-usage.md")
    for name in ("chosen dev", "reference dev"):
        run_id = runs[name].manifest.run_id
        assert [reason for _, used, reason in uses if used == run_id] == ["baseline"], name
    # The ADRs that wait on these runs are decided, and the chosen model is ADR-0012's.
    for adr in ("0012-model-choice", "0007-prompt-format"):
        text = (ROOT / "docs" / "adr" / f"{adr}.md").read_text(encoding="utf-8")
        assert "\n- Status: accepted" in text, adr
    chosen = runs["chosen dev"].manifest.model_ref.split("@")[0]
    adr = (ROOT / "docs" / "adr" / "0012-model-choice.md").read_text(encoding="utf-8")
    assert f"\n- Chosen model: `{chosen}`\n" in adr


@pytest.mark.phase(3)
@pytest.mark.requires_full_data
def test_phase3_baseline_reports_rebuild() -> None:
    data = resolve_data_dir()
    for name, run in gpu_story_runs("FG-307", BASELINES).items():
        checks = rebuild(run.manifest.run_id, runs_root=RUNS, data_dir=data)
        assert [check.diff for check in checks] == [()], name


# FG-404's Kaggle SFT runs (plan section 3.8): the pilot on the small candidate, and the
# main run on the chosen model, each from its config.
SFT_RUNS = {
    "pilot": ROOT / "configs" / "sft" / "qwen3-1p7b-pilot.yaml",
    "main": ROOT / "configs" / "sft" / "qwen3-4b-main.yaml",
}
# What a training run's manifest config holds of its file, under the config hash.
TRAIN_SECTIONS = {"run", "data", "lora", "sft"}


def kernel_training_runs(
    story: str,
    wanted: Mapping[str, Path],
    *,
    runs: Path = RUNS,
    pending: Path = PENDING,
) -> dict[str, RunManifest]:
    """The committed kernel training run of each name in ``wanted``, made from its config.

    Each must be the only one of its name under ``runs``. When any is missing the calling
    test skips while ``story`` is in ``pending``, and fails once it isn't, as for
    ``gpu_story_runs``.
    """
    found, missing = training_runs(story, wanted, runs=runs)
    if missing:
        _missing(story, missing, runs=runs, pending=pending)
    return found


def training_runs(
    story: str, wanted: Mapping[str, Path], *, runs: Path = RUNS
) -> tuple[dict[str, RunManifest], list[str]]:
    """``kernel_training_runs``'s runs, and the names of those ``runs`` lacks, without
    skipping."""
    store = LocalRunStore(runs)
    trained = [
        manifest
        for manifest in (store.get_manifest(run) for run in store.runs())
        if manifest.kind in ("sft", "grpo") and manifest.platform in HOSTS
    ]
    found: dict[str, RunManifest] = {}
    missing: list[str] = []
    for name, path in wanted.items():
        made = load_train_config(path).model_dump(mode="json", include=TRAIN_SECTIONS)
        matches = [m for m in trained if {k: m.config.get(k) for k in made} == made]
        ids = [manifest.run_id for manifest in matches]
        if len(matches) > 1:
            pytest.fail(f"{story}'s {name} is one run, but {runs} has {len(ids)}: {ids}")
        if not matches:
            missing.append(name)
            continue
        found[name] = matches[0]
    return found, missing


@pytest.mark.phase(4)
def test_kernel_training_runs_skip_only_while_pending(tmp_path: Path) -> None:
    config = load_train_config(SFT_RUNS["pilot"])
    made = config.model_dump(mode="json", include=TRAIN_SECTIONS)
    wanted = {"pilot": SFT_RUNS["pilot"]}
    runs, pending = tmp_path / "runs", tmp_path / "pending.yaml"
    pending.write_text("pending:\n  - story: FG-998\n    run: pilot\n", encoding="utf-8")
    model = config.run.model_ref

    # The same config trained on the laptop is no kernel run, and another config's is not it.
    sft_run_folder(runs, "on-the-laptop", model, [], config=made, platform="local-mps")
    other = {**made, "sft": {**made["sft"], "learning_rate": 1e-4}}
    sft_run_folder(runs, "other-config", model, [], config=other)
    with pytest.raises(pytest.skip.Exception, match=r"GPU run pending: FG-998's pilot \("):
        kernel_training_runs("FG-998", wanted, runs=runs, pending=pending)
    with pytest.raises(pytest.fail.Exception, match=r"FG-999 is not in pending\.yaml, but"):
        kernel_training_runs("FG-999", wanted, runs=runs, pending=pending)

    sft_run_folder(runs, "on-kaggle", model, [], config={**made, "sft_data": {"rows": 1}})
    found = kernel_training_runs("FG-999", wanted, runs=runs, pending=pending)
    assert found["pilot"].run_id == "on-kaggle"
    sft_run_folder(runs, "again", model, [], config=made)
    with pytest.raises(pytest.fail.Exception, match=r"pilot is one run, but .* has 2"):
        kernel_training_runs("FG-999", wanted, runs=runs, pending=pending)


def _checkpoint_of(run: StoredRun, trained: RunManifest) -> str | None:
    """The checkpoint of ``trained`` an eval run's adapter is, as a path in its run."""
    ref = checkpoint_ref(run.manifest.adapter_ref or "")
    return ref.rel if ref is not None and ref.run_id == trained.run_id else None


# The base models' full val configs, p1 and p2 (FG-307): an SFT run's checkpoints are each
# scored with the one of its model and prompt version, and the adapter alone added.
VAL_CONFIGS = [
    EVAL_CONFIGS / f"{name}-val{prompt}.yaml" for name in CANDIDATES for prompt in ("", "-p2")
]


def base_val_config(trained: RunManifest, configs: Sequence[Path] = VAL_CONFIGS) -> Path:
    """The full val config of the model ``trained`` trained, at the prompt version it
    trained on: what its checkpoints are scored with, and FG-307 scored the model with."""
    prompt = trained.config["data"]["prompt_version"]
    matches = [
        path
        for path in configs
        if (config := EvalConfig(**read_config_file(path))).model == trained.model_ref
        and config.prompt_version == prompt
    ]
    assert len(matches) == 1, f"{trained.run_id}: {matches} score its model at {prompt}"
    return matches[0]


def assert_sft_runs_scored(
    trained: Mapping[str, RunManifest],
    *,
    runs: Path = RUNS,
    configs: Sequence[Path] = VAL_CONFIGS,
    ledger: Path = LEDGER,
    dev_usage: Path = DEV_USAGE,
    val: Collection[str] | None = None,
) -> None:
    """Each kernel SFT run of ``trained`` committed under ``runs`` without its weights, its
    checkpoints scored from kernels on every question of ``val`` (the val split's), each
    with its base model's full val config and that checkpoint's files alone added, and
    best/ set from those scores (FG-404)."""
    if val is None:
        val = {example_id for example_id, _ in read_split_ids("val")} - FAILING_GOLD_IDS
    store = LocalRunStore(runs)
    evals = [read_run(runs / run) for run in store.runs() if store.get_manifest(run).kind == "eval"]
    dev_looks = {run_id for _, run_id, _ in read_uses(dev_usage)}
    for name, manifest in trained.items():
        root = runs / manifest.run_id
        assert (manifest.kind, manifest.status) == ("sft", "finished"), name
        assert manifest.config_hash == config_hash(manifest.config), name
        assert [row.platform for row in _ledger_rows(manifest.run_id, ledger)] == [
            manifest.platform
        ]
        # Committed without its weights, which the run store holds.
        assert (root / "config.yaml").is_file() and (root / "metrics.jsonl").is_file(), name
        assert not (root / "checkpoints").exists(), name

        # Its checkpoints on val, each from a kernel, with a report and a ledger row; none
        # of them on dev, and no dev look spent on them.
        base = EvalConfig(**read_config_file(base_val_config(manifest, configs)))
        scored = {_checkpoint_of(run, manifest): run for run in evals}
        scored.pop(None, None)
        assert scored, f"{name}: no eval run of its checkpoints"
        assert manifest.ended_at is not None, name
        for rel, run in scored.items():
            made = run.manifest
            # The base model's full val config, the base run's (FG-307), and the checkpoint
            # alone added, so the two differ in the adapter alone (FG-405).
            expected = replace(base, adapter=made.adapter_ref).manifest_config()
            assert _made_from(made.config) == expected, f"{rel}: not the base config with it"
            assert sorted(score.example_id for score in run.scores) == sorted(val), rel
            assert made.platform in HOSTS and manifest.ended_at <= made.started_at, rel
            # Scored on the files the run saved as that checkpoint.
            saved = manifest.outputs[digest_output(rel)]
            assert made.outputs[ADAPTER_DIGEST] == saved, f"{rel}: scored other weights"
            assert run.paths.report.is_file(), rel
            assert [row.platform for row in _ledger_rows(made.run_id, ledger)] == [made.platform]
            assert made.run_id not in dev_looks, rel
        expected_ckpts = {FINAL} if name == "pilot" else {manifest.outputs["kept_at_0.5"], FINAL}
        assert set(scored) == expected_ckpts, name

        # best/ set from those runs' own scores: the highest official EX among them.
        assert (manifest.outputs["best"], manifest.outputs["promotion"]) == (BEST, PROMOTION_FILE)
        record = json.loads((root / PROMOTION_FILE).read_text(encoding="utf-8"))
        ids = [run.manifest.run_id for run in scored.values()]
        scores, sources = checkpoint_scores(manifest.run_id, ids, base=base, runs_root=runs)
        assert (record["scores"], record["eval_runs"]) == (scores, sources), name
        assert scores[record["checkpoint"]] == max(scores.values()), name


@pytest.mark.phase(4)
def test_phase4_sft_runs_present() -> None:
    assert_sft_runs_scored(kernel_training_runs("FG-404", SFT_RUNS))


@pytest.mark.phase(4)
def test_sft_runs_are_scored_on_their_base_models_full_val(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A kernel SFT run, its checkpoints' kernel val runs and best/, made on the fixture with
    # the replies stood in for, in runs/; then committed, as the runbook says, to docs/runs.
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    work, committed = tmp_path / "runs", tmp_path / "docs-runs"
    model, sft_id = "org/model@abc123", "2026-11-01-model-sft-abc1234"
    half = "checkpoints/step-000121"
    base = tmp_path / "model-val.yaml"
    base.write_text(f"model: {model}\nbackend: vllm\nlimit: null\nprompt_version: p1\n")
    base_config = EvalConfig(**read_config_file(base))
    trained_on = {"data": {"prompt_version": "p1"}}
    trained = sft_run_folder(
        work,
        sft_id,
        model,
        [half, FINAL],
        config=trained_on,
        config_hash=config_hash(trained_on),
        started_at=datetime(2026, 11, 1, 8, 0, tzinfo=UTC),
        ended_at=datetime(2026, 11, 1, 10, 0, tzinfo=UTC),
    )
    LocalRunStore(work).put_manifest(
        trained.model_copy(update={"outputs": {**trained.outputs, "kept_at_0.5": half}})
    )
    for name in ("config.yaml", "metrics.jsonl"):
        (work / sft_id / name).write_text("{}\n", encoding="utf-8")
    examples = load_val()
    gold = {example.id: example.gold_sql for example in examples}
    ledger, dev_usage = tmp_path / "ledger.md", tmp_path / "dev-usage.md"

    def evaluate(run_id: str, rel: str, **changes: Any) -> None:
        """A kernel's val run of checkpoint ``rel`` with the base config, as ``changes``
        change it, answering with gold, then its report and ledger row."""
        options = {"adapter": str(work / sft_id / rel), "run_id": run_id, "host": "kaggle"}
        config = EvalConfig(**{**read_config_file(base), **options, **changes})
        replies = FakeBackend(
            lambda prompt, params, i: f"```sql\n{gold[prompt.example_id]}\n```", model_ref=model
        )
        run_eval(config, backend=replies, allow_dirty=True, runs_root=work, workers=1)
        write_run_report(run_id, runs_root=work)
        append_once(LedgerRow(date.today(), "kaggle", "Tesla T4", 0.3, run_id, "val"), ledger)

    evaluate("half-val", half)
    evaluate("final-val", FINAL)
    ids = ["half-val", "final-val"]
    scores, sources = checkpoint_scores(sft_id, ids, base=base_config, runs_root=work)
    promote_best(sft_id, scores, sources=sources, runs_root=work)
    append_once(LedgerRow(date.today(), "kaggle", "Tesla T4", 2.0, sft_id, "sft"), ledger)

    def commit(folder: Path, evals: Sequence[str]) -> RunManifest:
        """``folder`` as docs/runs: the eval runs whole, the SFT run without checkpoints."""
        for run_id in evals:
            shutil.copytree(work / run_id, folder / run_id)
        (folder / sft_id).mkdir(parents=True)
        for name in ("manifest.json", "config.yaml", "metrics.jsonl", PROMOTION_FILE):
            shutil.copyfile(work / sft_id / name, folder / sft_id / name)
        return LocalRunStore(folder).get_manifest(sft_id)

    def check(folder: Path, manifest: RunManifest) -> None:
        val = [example.id for example in examples]
        assert_sft_runs_scored(
            {"main": manifest},
            runs=folder,
            configs=[base],
            ledger=ledger,
            dev_usage=dev_usage,
            val=val,
        )

    check(committed, commit(committed, ids))

    # A part of val or other sampling for the final adapter, or other weights than the run
    # saved as it: a best/ chosen from such a run is refused, as promote refuses it.
    weights = work / sft_id / FINAL / "adapter_model.safetensors"
    saved = weights.read_text()
    weights.write_text("weights from elsewhere")
    evaluate("final-other-weights", FINAL)
    weights.write_text(saved)
    evaluate("final-part", FINAL, limit=3)
    evaluate("final-sampled", FINAL, temperature=0.7)
    for run_id, message in (
        ("final-part", "not the base config with it"),
        ("final-sampled", "not the base config with it"),
        ("final-other-weights", "scored other weights"),
    ):
        folder = tmp_path / run_id
        with pytest.raises(AssertionError, match=f"{FINAL}: {message}"):
            check(folder, commit(folder, ["half-val", run_id]))
        with pytest.raises(ValueError, match=f"run {run_id} "):
            checkpoint_scores(sft_id, [run_id], base=base_config, runs_root=work)


# FG-405: what SFT changed on val, from the comparisons of each committed val run of an SFT
# adapter with its base run. The SFT pilot's is FG-491's; the main model's, FG-405's own,
# waits in pending.yaml for FG-404's and FG-307's runs.
COMPARISON = RUNS / SFT_PILOT_EVAL / "eval" / f"{COMPARE_PREFIX}{PILOT_RUN}.md"
MAIN_COMPARISON = "sft-main-comparison"  # its name in pending.yaml
NOTES = ROOT / "docs" / "notes" / "sft.md"
# What in the notes or a report names something rather than measures it: code spans, link
# targets, story and ADR ids, and a model's size such as 0.5B.
_NAMES = re.compile(r"`[^`\n]*`|\]\([^)\n]*\)|\b(?:FG|ADR)-\d+\b|\b\d+(?:\.\d+)?B\b")
_COMPARISON_LINK = re.compile(rf"\]\(([^)\s]+/eval/{COMPARE_PREFIX}[^)\s/]+\.md)\)")


def sft_comparisons(runs: Path = RUNS) -> dict[str, list[Path]]:
    """The compare reports in the folder of each committed val run of an SFT run's
    checkpoint, by that run's id, for the runs that have one."""
    store = LocalRunStore(runs)
    manifests = [store.get_manifest(run) for run in store.runs()]
    trained = {manifest.run_id for manifest in manifests if manifest.kind == "sft"}
    found: dict[str, list[Path]] = {}
    for manifest in manifests:
        ref = checkpoint_ref(manifest.adapter_ref or "")
        if manifest.kind != "eval" or ref is None:
            continue
        reports = sorted((runs / manifest.run_id / "eval").glob(f"{COMPARE_PREFIX}*.md"))
        if ref.run_id in trained and reports:
            found[manifest.run_id] = reports
    return found


def main_comparison(runs: Path = RUNS) -> Path | None:
    """FG-405's comparison of the main model with and without SFT where FG-404's and
    FG-307's runs put it: in the folder of the val run of the main SFT run's
    checkpoints/best, the one its best.json names, against FG-307's run of the model's
    full val config, so the two differ in the adapter alone. None while any is missing."""
    trained, _ = training_runs("FG-404", {"main": SFT_RUNS["main"]}, runs=runs)
    record = runs / trained["main"].run_id / PROMOTION_FILE if trained else None
    if record is None or not record.is_file():
        return None
    promoted = json.loads(record.read_text(encoding="utf-8"))
    best_val = promoted["eval_runs"][promoted["checkpoint"]]
    base, _ = kernel_eval_runs("FG-307", {"base": [base_val_config(trained["main"])]}, runs=runs)
    if not base:
        return None
    report = runs / best_val / "eval" / f"{COMPARE_PREFIX}{base['base'].manifest.run_id}.md"
    return report if report.is_file() else None


def fg405_comparison(*, runs: Path = RUNS, pending: Path = PENDING) -> Path:
    """``main_comparison``, or a skip while FG-405 is in ``pending`` and a failure once it
    isn't, as for a GPU story's runs."""
    report = main_comparison(runs)
    if report is None:
        _missing("FG-405", [MAIN_COMPARISON], runs=runs, pending=pending)
    return report


@pytest.mark.phase(4)
def test_main_comparison_skips_only_while_fg405_is_pending(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    runs, pending, unlisted = tmp_path / "runs", tmp_path / "pending.yaml", tmp_path / "done.yaml"
    pending.write_text(f"pending:\n  - story: FG-405\n    run: {MAIN_COMPARISON}\n")
    unlisted.write_text("pending:\n  - story: FG-404\n    run: sft-main\n")

    def missing() -> None:
        with pytest.raises(
            pytest.skip.Exception, match=f"GPU run pending: FG-405's {MAIN_COMPARISON}"
        ):
            fg405_comparison(runs=runs, pending=pending)
        with pytest.raises(pytest.fail.Exception, match=r"FG-405 is not in done\.yaml, but"):
            fg405_comparison(runs=runs, pending=unlisted)

    missing()
    # FG-404's main run from its config, its final adapter's kernel val run promoted to best/.
    made = load_train_config(SFT_RUNS["main"]).model_dump(mode="json", include=TRAIN_SECTIONS)
    main = sft_run_folder(runs, "main-sft", made["run"]["model_ref"], [FINAL], config=made)
    base = base_val_config(main)
    gold = {example.id: example.gold_sql for example in load_val()}

    def evaluate(run_id: str, adapter: str | None) -> None:
        options = {**read_config_file(base), "run_id": run_id, "host": "kaggle"}
        replies = FakeBackend(lambda prompt, params, i: f"```sql\n{gold[prompt.example_id]}\n```")
        run_eval(
            EvalConfig(**options, adapter=adapter),
            backend=replies,
            allow_dirty=True,
            runs_root=runs,
            workers=1,
        )
        write_run_report(run_id, runs_root=runs)

    evaluate("best-val", str(runs / "main-sft" / FINAL))
    record = {"checkpoint": FINAL, "eval_runs": {FINAL: "best-val"}}
    (runs / "main-sft" / PROMOTION_FILE).write_text(json.dumps(record), encoding="utf-8")
    missing()
    # FG-307's run of the model's val config, and a comparison against another run.
    evaluate("base-val", None)
    stray = runs / "best-val" / "eval" / f"{COMPARE_PREFIX}another-run.md"
    stray.write_text("a comparison\n", encoding="utf-8")
    missing()

    report = runs / "best-val" / "eval" / f"{COMPARE_PREFIX}base-val.md"
    report.write_text("a comparison\n", encoding="utf-8")
    assert fg405_comparison(runs=runs, pending=pending) == report
    assert fg405_comparison(runs=runs, pending=unlisted) == report


def _number_list(text: str) -> list[str]:
    """The numbers ``text`` states, in order, as a report writes them, leaving out its
    names and a report's Generated time."""
    return NUMBER.findall(_NAMES.sub(" ", GENERATED.sub(" ", text)))


def _numbers(text: str) -> set[str]:
    return set(_number_list(text))


@pytest.mark.phase(4)
@pytest.mark.requires_full_data
def test_phase4_compare_report_rebuildable(monkeypatch: pytest.MonkeyPatch) -> None:
    comparisons = sft_comparisons()
    assert COMPARISON in comparisons.get(SFT_PILOT_EVAL, []), comparisons
    # Run a is the base model alone: no adapter, and run b's settings but those that change
    # no score, as checkpoint_scores reads them, so a report measures the adapter alone.
    scoreless = ("adapter", "allow_over_budget")
    store = LocalRunStore(RUNS)
    for run_id, reports in comparisons.items():
        sft = store.get_manifest(run_id)
        for report in reports:
            base = store.get_manifest(report.stem.removeprefix(COMPARE_PREFIX))
            assert (base.kind, base.adapter_ref, base.model_ref) == ("eval", None, sft.model_ref)
            for name in (base.config.keys() | sft.config.keys()).difference(scoreless):
                assert base.config.get(name) == sft.config.get(name), (report.name, name)
    # forge report rebuild renders each again from both runs' stored scores and diffs it
    # line for line with the committed file, the Generated time aside.
    monkeypatch.setenv("FORGE_RUNS_DIR", str(RUNS))
    for run_id, reports in comparisons.items():
        result = CliRunner().invoke(app, ["report", "rebuild", run_id])
        assert result.exit_code == 0, result.output
        for report in reports:
            assert f"{report}: matches its rebuild\n" in result.output, result.output
    # The main model's comparison, FG-405's subject, among them once it is committed.
    main = fg405_comparison()
    assert main in comparisons.get(main.parent.parent.name, []), comparisons


def reading_blocks(text: str, folder: Path) -> list[tuple[Path | None, str]]:
    """Each block of ``text`` between blank lines, with the comparison it reads: the last
    one linked in it or above it, as a path from ``folder``."""
    blocks: list[tuple[Path | None, str]] = []
    reading: Path | None = None
    for block in text.split("\n\n"):
        links = _COMPARISON_LINK.findall(block)
        if links:
            reading = (folder / links[-1]).resolve()
        blocks.append((reading, block))
    return blocks


def _bullets(block: str) -> tuple[str, list[str]]:
    """A block's text before its first bullet, and each bullet with the lines it wraps to."""
    lead, *bullets = re.split(r"(?:^|\n)- ", block)
    return lead, bullets


def bullet_lists(text: str, folder: Path) -> dict[Path, list[int]]:
    """The size of each bullet list in ``text``, by the comparison it reads."""
    lists: dict[Path, list[int]] = {}
    for reading, block in reading_blocks(text, folder):
        _, bullets = _bullets(block)
        if bullets:
            assert reading is not None, f"a list reads no comparison: {block[:60]!r}"
            lists.setdefault(reading, []).append(len(bullets))
    return lists


def report_lines(report: str) -> dict[str, list[list[str]]]:
    """The numbers of each line of a compare report, in order, by the labels a reader cites
    them under, casefolded: a table row's first cell, such as ``medium`` or
    ``schema_link``, for that row, and a section's heading, such as ``McNemar's test``, for
    every line under it."""
    lines: dict[str, list[list[str]]] = {}
    section = ""
    for line in report.splitlines():
        if line.startswith("## "):
            section = line.removeprefix("## ").strip().casefold()
            continue
        numbers = _number_list(line)
        row = line.split("|")[1].strip().casefold() if line.startswith("|") else ""
        for label in {section, row} - {""}:
            if numbers:
                lines.setdefault(label, []).append(numbers)
    return lines


# Where a clause of a bullet ends: a full stop, a semicolon or a colon before a space.
_CLAUSE_END = r"[.;:](?=\s|$)"


def citations(bullet: str, labels: Collection[str]) -> list[tuple[str | None, list[str]]]:
    """The numbers ``bullet`` states, in runs, each with the label of ``labels`` it named
    last before them, casefolded. A run ends at the next label and at a clause's end; a
    clause that names no label goes on with the one before it."""
    # Longest first, so "valid sql" wins over a shorter label it starts with; with no
    # labels, a pattern that never matches.
    named = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    named = named or "(?!)"
    token = re.compile(
        rf"(?P<name>{_NAMES.pattern})|(?<!\w)(?P<label>{named})(?!\w)"
        rf"|(?P<end>{_CLAUSE_END})|(?P<number>{NUMBER.pattern})",
        re.IGNORECASE,
    )
    runs: list[tuple[str | None, list[str]]] = []
    label: str | None = None
    numbers: list[str] = []
    for match in token.finditer(" ".join(bullet.split())):
        kind, found = match.lastgroup, match.group().strip("`").casefold()
        if kind == "number":
            numbers.append(match.group())
        elif kind != "name" or found in labels:
            if numbers:
                runs.append((label, numbers))
                numbers = []
            if kind != "end":
                label = found
    if numbers:
        runs.append((label, numbers))
    return runs


def miscited(bullet: str, report: str) -> list[str]:
    """Each run of numbers ``bullet`` states that isn't, in its order, on one line of
    ``report`` that the label it is cited under names; empty when all are."""
    lines = report_lines(report)
    problems = []
    for label, numbers in citations(bullet, lines):
        cited = ", ".join(numbers)
        if label is None:
            problems.append(f"{cited}: after no row or section of the report: {bullet[:60]!r}")
        elif not any(_in_order(numbers, line) for line in lines[label]):
            problems.append(f"{cited}: not on the report's {label} line(s), in that order")
    return problems


def _in_order(numbers: Sequence[str], line: Sequence[str]) -> bool:
    rest = iter(line)
    return all(number in rest for number in numbers)


@pytest.mark.phase(4)
def test_sft_notes_cite_only_report_numbers() -> None:
    text = NOTES.read_text(encoding="utf-8")
    # The notes read the comparisons they link, the SFT pilot's among them.
    linked = [(NOTES.parent / target).resolve() for target in _COMPARISON_LINK.findall(text)]
    assert COMPARISON in linked, linked
    assert all(path.is_file() for path in linked), linked
    # Three to five bullets a list, the pilot's and each one a later comparison adds.
    lists = bullet_lists(text, NOTES.parent)
    assert COMPARISON in lists, lists
    assert all(3 <= size <= 5 for sizes in lists.values() for size in sizes), lists
    # Every number comes from the comparison its block reads: in a bullet, from the row or
    # section the bullet names before it, in the report's order, and around the lists,
    # from anywhere in that report.
    problems = []
    for reading, block in reading_blocks(text, NOTES.parent):
        report = reading.read_text(encoding="utf-8") if reading is not None else ""
        lead, bullets = _bullets(block)
        problems += [f"{n} is not in {reading}" for n in sorted(_numbers(lead) - _numbers(report))]
        for bullet in bullets:
            problems += miscited(bullet, report)
    assert problems == []
    # And a list of its own on the main model's comparison, once that is committed.
    assert fg405_comparison().resolve() in lists, lists


@pytest.mark.phase(4)
def test_notes_check_ties_each_number_to_its_row() -> None:
    report = COMPARISON.read_text(encoding="utf-8")
    cited = "Only medium's interval clears zero, at +6.1 [+0.3, +12.4]"
    assert miscited(cited, report) == []
    # Numbers the report holds, but on another row, in another order or under no label:
    # each one is somewhere in the report, so only a check by row tells them apart.
    wrong = {
        "Only medium's interval clears zero, at +4.9 [+0.0, +9.9]": "medium",
        "schema_link failures fell from 266 to 2 (-63)": "schema_link",
        "where went from 22 to 13 (+9)": "where",
        "McNemar's test pairs clusters. The adapter alone is right on 71, the base on 115.": (
            "mcnemar's test"
        ),
    }
    for bullet, label in wrong.items():
        assert _numbers(bullet) <= _numbers(report), bullet
        assert miscited(bullet, report) == [
            f"{', '.join(_number_list(bullet))}: not on the report's {label} line(s), in that order"
        ], bullet
    unlabeled = "The pilot gained on 115 questions."
    assert miscited(unlabeled, report) == [
        f"115: after no row or section of the report: {unlabeled!r}"
    ]
    # A label in a code span names its row as one in the text does.
    assert miscited("`schema_link` fell from 266 to 203", report) == []


# FG-591's GRPO pilot on the laptop's GPU (ADR-0016): the SFT pilot's adapter on the first
# 3,000 train questions, whose scores picked the rows GRPO trains on.
GRPO_PILOT_CONFIG = ROOT / "configs" / "grpo" / "pilot-qwen25coder05b-mps.yaml"
GRPO_FILTER_EVAL = "2026-11-17-qwen2-5-coder-0-5b-instruct-eval-a27afde"
GRPO_FILTER_LIMIT = 3000
# Where forge data build-grpo kept the filter's rows, and forge train grpo read them.
GRPO_ROWS = RUNS / "grpo"


def grpo_filter_rows(folder: Path = GRPO_ROWS / GRPO_FILTER_EVAL) -> GrpoFilter:
    """The rows the filter kept, as committed, once they hash to their summary's hash;
    ``read_filter`` checks the data hash too, which needs the full data."""
    summary = FilterSummary.model_validate_json((folder / SUMMARY_FILE).read_bytes())
    lines = (folder / ROWS_FILE).read_bytes().splitlines()
    rows = [FilterRow.model_validate_json(line) for line in lines]
    assert filter_hash(summary, rows) == summary.hash, folder
    return GrpoFilter(rows, summary)


@pytest.mark.phase(5)
def test_grpo_pilot_filter_eval_is_committed_whole() -> None:
    stored = read_run(RUNS / GRPO_FILTER_EVAL)
    manifest = stored.manifest
    _made_from_its_commit(manifest)
    assert manifest.config["device"] == "mps"
    # FG-291's settings, greedy with one reply a question and no retry, on train questions
    # with the SFT pilot's adapter: the eval the filter reads (check_filter_run).
    pilot = EvalConfig(**read_config_file(PILOT_CONFIG))
    config = replace(pilot, split="train", limit=GRPO_FILTER_LIMIT, adapter=SFT_PILOT_ADAPTER)
    assert manifest.config == config.manifest_config()
    assert (config.params.temperature, config.params.n, config.retry) == (0.0, 1, 0)
    assert manifest.adapter_ref == SFT_PILOT_ADAPTER
    # The first 3,000 train questions in file order but the known failing golds, once each:
    # train only, so none of the val databases its adapter is scored on.
    train = [example_id for example_id, _ in read_split_ids("train")]
    first = [example_id for example_id in train if example_id not in FAILING_GOLD_IDS]
    assert [score.example_id for score in stored.scores] == first[:GRPO_FILTER_LIMIT]
    report = stored.paths.report.read_text(encoding="utf-8")
    assert "\n| Platform | `local-mps` |\n" in report
    assert f"\n| Git commit | `{manifest.git_sha}` |\n" in report
    assert f"\n| Adapter | `{SFT_PILOT_ADAPTER}` |\n" in report
    assert stored.paths.chart.is_file()

    # The rows it picked, beside the runs, from its scores and adapter: every question its
    # reply got wrong and 30% of those it got right, by hash, but those on a rowless
    # database and those whose gold the train cache timed over 1 s.
    picked = grpo_filter_rows()
    summary = picked.summary
    assert (summary.eval_run, summary.model_ref, summary.adapter_ref) == (
        GRPO_FILTER_EVAL,
        manifest.model_ref,
        SFT_PILOT_ADAPTER,
    )
    assert summary.adapter_sha256 == manifest.outputs[ADAPTER_DIGEST]
    assert summary.data_manifest_hash == manifest.data_manifest_hash
    assert (summary.keep_solved, summary.max_gold_ms) == (KEEP_SOLVED, MAX_GOLD_MS)
    dbs = dict(read_split_ids("train"))
    rowless = {score.example_id for score in stored.scores if dbs[score.example_id] in ROWLESS_DBS}
    left_out = rowless | set(summary.slow_gold_ids)
    rest = [score for score in stored.scores if score.example_id not in left_out]
    solved = [score.example_id for score in rest if score.ex_fast]
    stay = set(lowest_hashes(solved, round_share(KEEP_SOLVED, len(solved)), summary.seed))
    kept = [score for score in rest if not score.ex_fast or score.example_id in stay]
    assert [(row.example_id, row.solved) for row in picked.rows] == [
        (score.example_id, score.ex_fast) for score in kept
    ]
    assert (summary.total.scored, summary.total.kept) == (len(stored.scores), len(kept))
    assert f"(grpo/{GRPO_FILTER_EVAL}/" in (RUNS / "README.md").read_text(encoding="utf-8")
