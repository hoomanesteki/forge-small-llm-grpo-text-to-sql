"""Runs committed under docs/runs/: whole, made as their configs say, and rebuildable.

A GPU story's runs wait in ``docs/runs/pending.yaml`` while the accounts they need don't
exist. ``gpu_story_runs`` finds a story's runs by the configs they were made from, and
skips the checks on them only while the story is listed there.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from datasets import Dataset

from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.data.prompts import chat_token_counter, load_tokenizer
from forge.data.spider import FAILING_GOLD_IDS, resolve_data_dir
from forge.data.splits import load_train, read_split_ids
from forge.eval.config import HOSTS, EvalConfig, read_config_file
from forge.eval.report import rebuild, write_run_report
from forge.eval.reproducibility import compare_runs
from forge.eval.run_files import StoredRun, read_run
from forge.eval.runner import run_eval
from forge.ops.dev_usage import read_uses
from forge.ops.hub import BEST, FINAL, LocalRunStore
from forge.ops.ledger import LedgerRow, read_rows, run_hours
from forge.ops.manifest import config_hash
from forge.train.config import load_train_config
from forge.train.sft import PROMOTION_FILE, checkpoint_scores, fit_rows
from forge.train.sft_data import build_sft_dataset
from tests.fakes import sft_run_folder
from tests.report_trace import untraced

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


def _ledger_rows(run_id: str) -> list[LedgerRow]:
    return [row for row in read_rows(ROOT / "docs" / "compute-ledger.md") if row.run_id == run_id]


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
    if missing:
        _missing(story, missing, runs=runs, pending=pending)
    return found


def _missing(story: str, names: Sequence[str], *, runs: Path, pending: Path) -> None:
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
    if missing:
        _missing(story, missing, runs=runs, pending=pending)
    return found


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
    parts = (run.manifest.adapter_ref or "").split("/")
    if len(parts) < 3 or parts[-3] != trained.run_id or parts[-2] != "checkpoints":
        return None
    return "/".join(parts[-2:])


@pytest.mark.phase(4)
def test_phase4_sft_runs_present() -> None:
    trained = kernel_training_runs("FG-404", SFT_RUNS)
    store = LocalRunStore(RUNS)
    evals = [read_run(RUNS / run) for run in store.runs() if store.get_manifest(run).kind == "eval"]
    dev_looks = {run_id for _, run_id, _ in read_uses(ROOT / "docs" / "dev-usage.md")}
    for name, manifest in trained.items():
        root = RUNS / manifest.run_id
        assert (manifest.kind, manifest.status) == ("sft", "finished"), name
        assert manifest.config_hash == config_hash(manifest.config), name
        assert [row.platform for row in _ledger_rows(manifest.run_id)] == [manifest.platform]
        # Committed without its weights, which the run store holds.
        assert (root / "config.yaml").is_file() and (root / "metrics.jsonl").is_file(), name
        assert not (root / "checkpoints").exists(), name

        # Its checkpoints on val, each from a kernel, with a report and a ledger row; none
        # of them on dev, and no dev look spent on them.
        scored = {_checkpoint_of(run, manifest): run for run in evals}
        scored.pop(None, None)
        assert scored, f"{name}: no eval run of its checkpoints"
        for rel, run in scored.items():
            made = run.manifest
            assert made.config["split"] == "val" and made.platform in HOSTS, rel
            assert made.model_ref == manifest.model_ref, rel
            assert run.paths.report.is_file(), rel
            assert [row.platform for row in _ledger_rows(made.run_id)] == [made.platform], rel
            assert made.run_id not in dev_looks, rel
        expected = {FINAL} if name == "pilot" else {manifest.outputs["kept_at_0.5"], FINAL}
        assert set(scored) == expected, name

        # best/ set from those runs' own scores: the highest official EX among them.
        assert (manifest.outputs["best"], manifest.outputs["promotion"]) == (BEST, PROMOTION_FILE)
        record = json.loads((root / PROMOTION_FILE).read_text(encoding="utf-8"))
        ids = [run.manifest.run_id for run in scored.values()]
        scores, sources = checkpoint_scores(manifest.run_id, ids, runs_root=RUNS)
        assert (record["scores"], record["eval_runs"]) == (scores, sources), name
        assert scores[record["checkpoint"]] == max(scores.values()), name
