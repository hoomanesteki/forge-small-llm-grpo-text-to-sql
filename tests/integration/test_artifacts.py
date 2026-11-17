"""Runs committed under docs/runs/: whole, made as their configs say, and rebuildable.

A GPU story's runs wait in ``docs/runs/pending.yaml`` while the accounts they need don't
exist. ``gpu_story_runs`` finds a story's runs by the configs they were made from, and
skips the checks on them only while the story is listed there.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from forge.data.spider import FAILING_GOLD_IDS, resolve_data_dir
from forge.data.splits import read_split_ids
from forge.eval.config import HOSTS, EvalConfig, read_config_file
from forge.eval.report import rebuild, write_run_report
from forge.eval.reproducibility import compare_runs
from forge.eval.run_files import MANIFEST_FILE, StoredRun, read_run
from forge.eval.runner import run_eval
from forge.ops.dev_usage import read_uses
from forge.ops.ledger import read_rows
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
    stored = [
        read_run(folder)
        for folder in sorted(runs.iterdir())
        if folder.is_dir() and (folder / MANIFEST_FILE).is_file()
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
        if story in pending_stories(pending):
            pytest.skip(f"GPU run pending: {story}'s {', '.join(missing)} ({pending.name})")
        pytest.fail(f"{story} is not in {pending.name}, but {runs} lacks its {', '.join(missing)}")
    return found


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
# The candidates' runs on val, which choose the model and the prompt, ran first and are
# checked on their own while the dev and reference runs wait (docs/runs/pending.yaml).
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
    val = gpu_story_runs("FG-307", VAL_BASELINES)
    # I6 on the T4: the first 100 val questions again agree with the whole run's.
    i6 = compare_runs(val["I6 repeat"], val[f"{I6_MODEL} val"])
    assert i6.ok, i6.describe()
    for run in val.values():
        assert _kaggle_rows(run.manifest.run_id) == ["kaggle"], run.manifest.run_id

    # The rest skip, naming what is missing, while FG-307 waits in pending.yaml.
    runs = gpu_story_runs("FG-307", BASELINES)
    assert_p2_ran_for_the_chosen_model(runs)
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
    later = gpu_story_runs("FG-307", BASELINES)
    assert_rebuilds({name: run for name, run in later.items() if name not in VAL_BASELINES})
