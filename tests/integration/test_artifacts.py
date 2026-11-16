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
from typer.testing import CliRunner

from forge.cli import app
from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.data.prompts import chat_token_counter, load_tokenizer
from forge.data.spider import FAILING_GOLD_IDS, load_checksums, resolve_data_dir
from forge.data.splits import load_train, read_split_ids
from forge.eval.config import HOSTS, EvalConfig, read_config_file
from forge.eval.report import COMPARE_PREFIX, rebuild, write_run_report
from forge.eval.reproducibility import compare_runs
from forge.eval.run_files import StoredRun, read_run
from forge.eval.runner import run_eval
from forge.ops.adapters import checkpoint_ref
from forge.ops.dev_usage import read_uses
from forge.ops.hub import BEST, FINAL, LocalRunStore
from forge.ops.ledger import LedgerRow, read_rows, run_hours
from forge.ops.manifest import config_hash
from forge.train.config import load_train_config
from forge.train.sft import PROMOTION_FILE, checkpoint_scores, fit_rows
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


# Integration checkpoint 3's Kaggle job (scripts/smoke/phase_3.sh), FG-303's criterion 4.
SMOKE_JOB = {"50-question eval": [EVAL_CONFIGS / "smoke-kaggle-50-val.yaml"]}
SPIKE_REPORTS = ROOT / "docs" / "spikes" / "kaggle-gpu"


def _kaggle_rows(run_id: str) -> list[str]:
    rows = read_rows(ROOT / "docs" / "compute-ledger.md")
    return [row.platform for row in rows if row.run_id == run_id]


@pytest.mark.phase(3)
def test_phase3_spike_reports_present() -> None:
    # The spike's kernel output holds no run, so its report and its push record are
    # committed instead, and its GPU hours are a ledger row added by hand.
    if "FG-301" in pending_stories():
        pytest.skip(f"GPU run pending: FG-301's spike ({PENDING.name})")
    pushed = json.loads((SPIKE_REPORTS / "kernels.json").read_text(encoding="utf-8"))
    report = json.loads((SPIKE_REPORTS / "spike-report.json").read_text(encoding="utf-8"))
    assert (pushed["job"], report["story"]) == ("spike", "FG-301")
    assert report["finished_at"] is not None
    assert pushed["code_sha"].startswith(pushed["run_id"].rsplit("-", 1)[1])
    assert _kaggle_rows(pushed["run_id"]) == ["kaggle"]


@pytest.mark.phase(3)
def test_phase3_kaggle_eval_job_present() -> None:
    (run,) = gpu_story_runs("FG-303", SMOKE_JOB).values()
    manifest = run.manifest
    assert (manifest.platform, manifest.gpu, manifest.status) == ("kaggle", "Tesla T4", "finished")
    # The first 50 val questions, once each: val only, never dev.
    val = [example_id for example_id, _ in read_split_ids("val")]
    first = [example_id for example_id in val if example_id not in FAILING_GOLD_IDS][:50]
    assert [score.example_id for score in run.scores] == first
    assert manifest.run_id.endswith(manifest.git_sha[:7])
    # The kernel's own record of the job came back with it.
    status = json.loads((run.paths.root / "status.json").read_text(encoding="utf-8"))
    assert (status["run_id"], status["status"], status["host"]) == (
        manifest.run_id,
        "finished",
        "kaggle",
    )
    assert _kaggle_rows(manifest.run_id) == ["kaggle"]
    # FG-302's datasets, as the kernel mounted them: the code dataset's version the push
    # named, holding the run's commit, and every verified Spider file.
    pushed = json.loads((run.paths.root / "kernels.json").read_text(encoding="utf-8"))
    owner, slug, _ = pushed["kernels"][0]["code"].split("/")
    stages = status["stages"]
    assert slug == "forge-code"
    assert stages["code"]["result"]["folder"].endswith(f"/{owner}/forge-code")
    assert stages["code"]["result"]["sha"] == manifest.git_sha
    assert stages["data"]["result"]["folder"].endswith(f"/{owner}/forge-spider")
    assert stages["data"]["result"]["files"] == len(load_checksums())


@pytest.mark.phase(3)
@pytest.mark.requires_full_data
def test_phase3_kaggle_eval_job_rebuilds() -> None:
    (run,) = gpu_story_runs("FG-303", SMOKE_JOB).values()
    report, chart = rebuild(run.manifest.run_id, runs_root=RUNS, data_dir=resolve_data_dir())
    assert (report.diff, report.skipped) == ((), None)
    # The kernel drew the chart on Linux; only a machine that draws as it did compares it.
    assert chart.diff == () or chart.skipped is not None


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
# The candidates' runs on val, which choose the model and the prompt, ran first.
VAL_BASELINES = {
    **{f"{name} val": [EVAL_CONFIGS / f"{name}-val.yaml"] for name in CANDIDATES},
    "chosen val p2": [EVAL_CONFIGS / f"{name}-val-p2.yaml" for name in CANDIDATES],
    "I6 repeat": [EVAL_CONFIGS / f"{I6_MODEL}-val-100.yaml"],
}
BASELINES = {
    **VAL_BASELINES,
    "reference val": [EVAL_CONFIGS / f"{REFERENCE}-val.yaml"],
    "chosen dev": [EVAL_CONFIGS / f"{name}-dev.yaml" for name in CANDIDATES],
    "reference dev": [EVAL_CONFIGS / f"{REFERENCE}-dev.yaml"],
}
# The runs after the choice, in the order they ran. Each is checked once it is committed,
# and the first one still missing skips the checks after it (docs/runs/pending.yaml).
LATER_BASELINES = [name for name in BASELINES if name not in VAL_BASELINES]


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


def _adr(name: str) -> str:
    return (ROOT / "docs" / "adr" / f"{name}.md").read_text(encoding="utf-8")


def _adr_chosen_model() -> str:
    chosen = re.search(r"\n- Chosen model: `([^`]+)`\n", _adr("0012-model-choice"))
    assert chosen, "ADR-0012 names no chosen model"
    return chosen.group(1)


@pytest.mark.phase(3)
def test_phase3_baseline_reports_present() -> None:
    val = gpu_story_runs("FG-307", VAL_BASELINES)
    # I6 on the T4: the first 100 val questions again agree with the whole run's, as
    # ADR-0012 quotes.
    i6 = compare_runs(val["I6 repeat"], val[f"{I6_MODEL} val"])
    assert i6.ok, i6.describe()
    assert " ".join(i6.describe().split(": ", 1)[1].split()) in " ".join(
        _adr("0012-model-choice").split()
    )
    for run in val.values():
        assert _kaggle_rows(run.manifest.run_id) == ["kaggle"], run.manifest.run_id
    # The ADRs the val runs decide are accepted, and p2 ran for the model ADR-0012 chose,
    # which left headroom.
    for adr in ("0012-model-choice", "0007-prompt-format"):
        assert "\n- Status: accepted\n" in _adr(adr), adr
    chosen = _adr_chosen_model()
    assert val["chosen val p2"].manifest.model_ref.split("@")[0] == chosen
    (chosen_val,) = [
        val[f"{name} val"]
        for name in CANDIDATES
        if val[f"{name} val"].manifest.model_ref.split("@")[0] == chosen
    ]
    assert _official_ex(chosen_val) < HEADROOM

    # The later runs, each once it is committed: the first still missing skips, naming
    # itself, while FG-307 waits in pending.yaml.
    runs = dict(val)
    uses = read_uses(ROOT / "docs" / "dev-usage.md")
    for name in LATER_BASELINES:
        runs |= gpu_story_runs("FG-307", {name: BASELINES[name]})
        manifest = runs[name].manifest
        assert _kaggle_rows(manifest.run_id) == ["kaggle"], name
        # Each dev run spent one look, logged with its reason (dev looks 1 and 2 of 6).
        if manifest.config["split"] == "dev":
            logged = [reason for _, used, reason in uses if used == manifest.run_id]
            assert logged == ["baseline"], name
        if name == "chosen dev":
            assert manifest.model_ref.split("@")[0] == chosen
            assert_p2_ran_for_the_chosen_model(runs)


def _report_rates(report: str) -> tuple[str, str, list[str]]:
    """A run report's model and prompt version, and the rates an ADR row quotes from it:
    official EX, fast EX and valid SQL with their intervals, then official EX at each
    hardness level, from easy to extra."""
    model = re.search(r"\n\| Model \| `([^`@]+)@", report)
    prompt = re.search(r"\n\| Prompt version \| `(\w+)` \|", report)
    assert model and prompt
    totals = {
        name: re.search(rf"\n\| {name} \| [\d,]+ \| [\d,]+ \| ([^|]+) \|", report)
        for name in ("official EX", "fast EX", "valid SQL")
    }
    levels = re.findall(
        r"\n\| (?:easy|medium|hard|extra) \| [\d,]+ \| [^|]+ \| ([\d.]+%) \[", report
    )
    assert all(totals.values()) and len(levels) == 4
    rates = [match.group(1).strip() for match in totals.values() if match]
    return model.group(1), prompt.group(1), rates + levels


@pytest.mark.phase(3)
def test_baseline_adrs_quote_the_reports() -> None:
    # Each row of ADR-0007's and ADR-0012's measured tables is its run's report, on val
    # or on dev: the prompt or the model it names first, then every rate in report order,
    # never typed by hand.
    rows = 0
    for name, names in (("0007-prompt-format", "prompt"), ("0012-model-choice", "model")):
        measured = _adr(name).split("\n## Measured\n", 1)[1].split("\n## ", 1)[0]
        for line in measured.splitlines():
            link = re.search(r"\]\(\.\./runs/([^/)]+)/eval/((?:val|dev)-report\.md)\)", line)
            if not line.startswith("| ") or link is None:
                continue
            report = (RUNS / link.group(1) / "eval" / link.group(2)).read_text(encoding="utf-8")
            model, prompt, rates = _report_rates(report)
            first, _, *quoted = (cell.strip() for cell in line.strip("|").split("|"))
            assert first == (prompt if names == "prompt" else f"`{model}`"), line
            assert quoted == rates[: len(quoted)], line
            rows += 1
    # p1 and p2 in ADR-0007; the candidates, the reference on val, and the chosen model and
    # the reference on dev in ADR-0012.
    assert rows == 8
    # ADR-0012's I5 lines, each after a link to the report it is quoted from, are that
    # report's own.
    i5 = re.findall(
        r"^- \[[^]]+\]\(\.\./runs/([^/)]+)/eval/(\w+-report\.md)\): (The checkers disagree .+)$",
        _adr("0012-model-choice"),
        flags=re.MULTILINE,
    )
    assert len(i5) == 6
    for run_id, report_name, line in i5:
        report = (RUNS / run_id / "eval" / report_name).read_text(encoding="utf-8")
        assert f"\n{line}\n" in report, line
    # ADR-0007 quotes the p1 and p2 compare report's lines as it wrote them.
    p2 = RUNS / "2026-11-16-qwen2-5-coder-1-5b-p2-c8f4b41" / "eval"
    compare = (p2 / "compare-2026-11-16-qwen2-5-coder-1-5b-val-c8f4b41.md").read_text(
        encoding="utf-8"
    )
    quotes = re.findall(r"^> (.+)$", _adr("0007-prompt-format"), flags=re.MULTILINE)
    assert quotes and all(quote in compare for quote in quotes)


def assert_rebuilds(runs: Mapping[str, StoredRun]) -> None:
    """Each run's report, chart and compare reports match their rebuild from its files."""
    data = resolve_data_dir()
    for name, run in runs.items():
        report, chart, *compares = rebuild(run.manifest.run_id, runs_root=RUNS, data_dir=data)
        assert (report.diff, report.skipped) == ((), None), name
        # Rendered on the laptop, so a Mac compares the chart; elsewhere it says it didn't.
        assert chart.diff == (), name
        assert [check.diff for check in compares] == [()] * len(compares), name


@pytest.mark.phase(3)
@pytest.mark.requires_full_data
def test_phase3_baseline_reports_rebuild() -> None:
    val = gpu_story_runs("FG-307", VAL_BASELINES)
    assert_rebuilds(val)
    # The p2 run holds its comparison with the chosen model's p1 run (ADR-0007).
    p1 = next(
        run
        for name, run in val.items()
        if name.endswith(" val")
        and run.manifest.model_ref == val["chosen val p2"].manifest.model_ref
    )
    compare = val["chosen val p2"].paths.report.parent / f"compare-{p1.manifest.run_id}.md"
    assert compare.is_file()
    for name in LATER_BASELINES:
        assert_rebuilds(gpu_story_runs("FG-307", {name: BASELINES[name]}))


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
    ref = checkpoint_ref(run.manifest.adapter_ref or "")
    return ref.rel if ref is not None and ref.run_id == trained.run_id else None


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


# FG-405: what SFT changed on val, from the comparisons of each committed val run of an SFT
# adapter with its base run. The SFT pilot's is FG-491's; FG-404's main run adds its own.
COMPARISON = RUNS / SFT_PILOT_EVAL / "eval" / f"{COMPARE_PREFIX}{PILOT_RUN}.md"
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


def _numbers(text: str) -> set[str]:
    """The numbers ``text`` states, as a report writes them, leaving out its names and
    a report's Generated time."""
    return set(NUMBER.findall(_NAMES.sub(" ", GENERATED.sub(" ", text))))


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


@pytest.mark.phase(4)
def test_sft_notes_cite_only_report_numbers() -> None:
    text = NOTES.read_text(encoding="utf-8")
    # The notes read the comparisons they link, the SFT pilot's among them.
    linked = [(NOTES.parent / target).resolve() for target in _COMPARISON_LINK.findall(text)]
    assert COMPARISON in linked, linked
    assert all(path.is_file() for path in linked), linked
    reported = set().union(*(_numbers(path.read_text(encoding="utf-8")) for path in linked))
    # Three to five bullets a list, the pilot's and each one a later comparison adds, and no
    # number in them or around them that those reports lack.
    sizes = [block.count("\n- ") + block.startswith("- ") for block in text.split("\n\n")]
    assert any(sizes) and all(3 <= size <= 5 for size in sizes if size), sizes
    assert sorted(_numbers(text) - reported) == []
