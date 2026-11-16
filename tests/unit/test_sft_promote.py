"""``promote_best``: the checkpoint of an SFT run that val scores highest, copied to
``checkpoints/best`` and recorded beside the manifest, in ``runs/`` and in the run store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.ops.adapters import adapter_sha256, digest_output
from forge.ops.hub import BEST, FINAL, HubRunStore, LocalRunStore, RunNotFound
from forge.train.sft import PROMOTION_FILE, promote_best
from tests.fakes import FakeHub, hub_error, sft_run_folder

RUN = "2026-11-20-qwen3-4b-sft-abc1234"
MODEL = "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
HALF = "checkpoints/step-000242"
LATEST = "checkpoints/step-000450"
REPO = "owner/forge-runs"


def _files(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): p.read_text() for p in folder.rglob("*")}


def _never_sleep(seconds: float) -> None:
    raise AssertionError(f"waited {seconds} s for a call that should not be retried")


@pytest.mark.phase(4)
def test_promote_best_copies_the_checkpoint_val_scores_highest(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    before = sft_run_folder(runs, RUN, MODEL, [HALF, LATEST, FINAL])
    run_dir = runs / RUN
    (run_dir / HALF / "trainer_state.json").write_text('{"global_step": 242}')
    sources = {HALF: "half-on-val", FINAL: "final-on-val"}

    promotion = promote_best(RUN, {FINAL: 0.60, HALF: 0.62}, sources=sources, runs_root=runs)

    # The half-way checkpoint, copied whole; the one it came from stays as it was.
    assert promotion.checkpoint == HALF and promotion.best == run_dir / BEST
    assert _files(run_dir / BEST) == _files(run_dir / HALF)
    assert (run_dir / HALF / "adapter_model.safetensors").read_text() == f"weights of {HALF}"
    # best.json: what was promoted, every score by checkpoint order, and where each is from.
    record = json.loads((run_dir / PROMOTION_FILE).read_text())
    assert (
        record
        == promotion.record
        == {
            "run_id": RUN,
            "checkpoint": HALF,
            "metric": "official EX on val",
            "scores": {HALF: 0.62, FINAL: 0.60},
            "eval_runs": {HALF: "half-on-val", FINAL: "final-on-val"},
        }
    )
    # The manifest's outputs name both, and best/'s digest, the promoted checkpoint's;
    # nothing else of it changes.
    stored = LocalRunStore(runs).get_manifest(RUN)
    assert stored == promotion.manifest
    best = {"best": BEST, "promotion": PROMOTION_FILE, digest_output(BEST): ""}
    best[digest_output(BEST)] = before.outputs[digest_output(HALF)]
    assert stored.outputs == {**before.outputs, **best}
    assert best[digest_output(BEST)] == adapter_sha256(run_dir / BEST)
    assert stored.model_copy(update={"outputs": before.outputs}) == before

    # A tie goes to the later checkpoint, and best/ holds its files and no others.
    again = promote_best(RUN, {HALF: 0.61, FINAL: 0.61}, runs_root=runs)
    assert again.checkpoint == FINAL
    assert _files(run_dir / BEST) == _files(run_dir / FINAL)
    assert "eval_runs" not in json.loads((run_dir / PROMOTION_FILE).read_text())
    # By step, not by name: step 450 comes after step 242.
    assert promote_best(RUN, {LATEST: 0.5, HALF: 0.5}, runs_root=runs).checkpoint == LATEST
    # The staging folder is gone, and the run's checkpoints are what they were, and best/.
    assert sorted(p.name for p in run_dir.iterdir()) == [
        PROMOTION_FILE,
        "checkpoints",
        "manifest.json",
    ]
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == [
        "best",
        "final",
        "step-000242",
        "step-000450",
    ]


@pytest.mark.phase(4)
def test_promote_best_refuses_what_it_cant_take(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    sft_run_folder(runs, RUN, MODEL, [HALF, FINAL])
    refused = [
        ({}, "none can be promoted"),
        ({BEST: 0.5}, r"no checkpoint of run .* never checkpoints/best"),
        ({"checkpoints/step-12": 0.5}, "'checkpoints/step-12' is no checkpoint"),
        ({"eval/val-scores.jsonl": 0.5}, "is no checkpoint"),
        ({HALF: 1.5}, "scores 1.5; an EX is a share from 0 to 1"),
        ({HALF: float("nan")}, "scores nan"),
        ({HALF: True}, "scores True"),
        ({"checkpoints/step-000100": 0.5}, "has no adapter_config.json or adapter_model"),
    ]
    for scores, message in refused:
        with pytest.raises(ValueError, match=message):
            promote_best(RUN, scores, runs_root=runs)
    with pytest.raises(ValueError, match="the sources name"):
        promote_best(RUN, {HALF: 0.5, FINAL: 0.4}, sources={HALF: "one"}, runs_root=runs)
    # A checkpoint's folder must hold the files the run saved there, by their digest.
    (runs / RUN / HALF / "adapter_model.safetensors").write_text("other weights")
    with pytest.raises(ValueError, match=rf"{HALF} holds other files than run {RUN} saved"):
        promote_best(RUN, {HALF: 0.5, FINAL: 0.4}, runs_root=runs)
    undigested = "2026-11-20-qwen3-4b-sft-fed4321"
    sft_run_folder(runs, undigested, MODEL, [FINAL], outputs={"adapter": FINAL})
    with pytest.raises(ValueError, match="records no digest of checkpoints/final, so nothing"):
        promote_best(undigested, {FINAL: 0.5}, runs_root=runs)

    # Only a finished SFT run has checkpoints to choose from.
    stopped = "2026-11-20-qwen3-4b-sft-def5678"
    sft_run_folder(runs, stopped, MODEL, [HALF], status="stopped_time_guard")
    with pytest.raises(ValueError, match="is a stopped_time_guard sft run; a checkpoint is"):
        promote_best(stopped, {HALF: 0.5}, runs_root=runs)
    evaluated = "2026-11-20-qwen3-4b-eval-abc1234"
    sft_run_folder(runs, evaluated, MODEL, [FINAL], kind="eval")
    with pytest.raises(ValueError, match="is a finished eval run"):
        promote_best(evaluated, {FINAL: 0.5}, runs_root=runs)
    with pytest.raises(RunNotFound):
        promote_best("2026-11-20-missing-sft-abc1234", {FINAL: 0.5}, runs_root=runs)
    # Nothing was promoted by any of them.
    assert not (runs / RUN / BEST).exists() and not (runs / RUN / PROMOTION_FILE).exists()
    assert "best" not in LocalRunStore(runs).get_manifest(RUN).outputs


@pytest.mark.phase(4)
def test_promote_best_goes_to_the_run_store_manifest_last(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    sft_run_folder(runs, RUN, MODEL, [HALF, FINAL])
    fake = FakeHub()
    hub = HubRunStore(REPO, api=fake, sleep=_never_sleep)
    hub.put_folder(RUN, runs / RUN)

    # A push the Hub refuses fails the promotion, and the folder's manifest stays the
    # store's, so promoting again goes through.
    fake.fail["upload_folder"] = [hub_error(403)]
    with pytest.raises(OSError):
        promote_best(RUN, {HALF: 0.7, FINAL: 0.6}, runs_root=runs, store=hub)
    assert "best" not in LocalRunStore(runs).get_manifest(RUN).outputs
    fake.calls.clear()

    promotion = promote_best(RUN, {HALF: 0.7, FINAL: 0.6}, runs_root=runs, store=hub)

    prefix = f"runs/{RUN}"
    assert fake.files[f"{prefix}/{BEST}/adapter_model.safetensors"] == f"weights of {HALF}".encode()
    assert json.loads(fake.files[f"{prefix}/{PROMOTION_FILE}"]) == promotion.record
    assert hub.get_manifest(RUN) == promotion.manifest == LocalRunStore(runs).get_manifest(RUN)
    uploads = [kwargs["path_in_repo"] for name, kwargs in fake.calls if name.startswith("upload")]
    assert uploads == [f"{prefix}/{BEST}", f"{prefix}/{PROMOTION_FILE}", f"{prefix}/manifest.json"]

    # A store whose copy of the run differs from the folder's is not overwritten.
    other = sft_run_folder(tmp_path / "elsewhere", RUN, MODEL, [FINAL], gpu_hours=3.0)
    hub.put_manifest(other)
    with pytest.raises(ValueError, match=r"holds another manifest of run .* forge runs pull"):
        promote_best(RUN, {FINAL: 0.9}, runs_root=runs, store=hub)
    assert promote_best(RUN, {FINAL: 0.9}, runs_root=runs).checkpoint == FINAL
