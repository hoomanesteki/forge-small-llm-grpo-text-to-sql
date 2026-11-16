"""SFT end to end on the tiny random-weight model, through the commands ``make smoke`` runs
(FG-403): three steps of LoRA on the fixture's train questions with
configs/sft/tiny-smoke.yaml, the adapter and the run's manifest saved, and the adapter
evaluated on 10 val questions with ``forge eval run --adapter``.

The model is random, so the eval scores noise. What this checks is the wiring: the train
rows in, the loss on the replies, the adapter out where an eval loads it, and the hashes
that tie the run to its config and data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file
from typer.testing import CliRunner

from forge import cli
from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.data.splits import load_train
from forge.ops.manifest import config_hash
from forge.train.config import LORA_TARGETS
from forge.train.sft_data import META_FILE, read_meta
from tests.report_trace import read_stored

ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "configs" / "sft" / "tiny-smoke.yaml"


@pytest.mark.phase(4)
def test_sft_trains_saves_and_evaluates(
    spider_mini: Path, tiny_model_ref: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("FORGE_RUNS_DIR", str(runs))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    sft_dir = tmp_path / "sft"
    cli_runner = CliRunner()
    train = [
        *("train", "sft", "--config", str(SMOKE_CONFIG), "--run-id", "sft"),
        *("--sft-dir", str(sft_dir), "--allow-dirty"),
    ]

    trained = cli_runner.invoke(cli.app, train)
    assert trained.exit_code == 0, trained.output

    run_dir = runs / "sft"
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_bytes())
    assert (manifest.kind, manifest.status, manifest.platform) == ("sft", "finished", "local-cpu")
    assert manifest.model_ref == tiny_model_ref
    # The data hash: the rows' own, which names their folder, beside the Spider files'.
    (folder,) = sft_dir.iterdir()
    meta = read_meta(folder)
    assert folder.name == meta.hash
    assert manifest.config["sft_data"] == {"hash": meta.hash, "rows": meta.counts.rows}
    assert meta.counts.rows == len(load_train())  # the fixture's train questions, all kept
    assert manifest.data_manifest_hash == data_manifest_hash(spider_mini)
    # The config hash, over the config the run stored.
    assert manifest.config_hash == config_hash(manifest.config)
    assert manifest.config["sft"]["max_length"] == 2048
    # The trainer read the rows in memory, so its tokenizing left no cache files with them.
    assert not [p.name for p in folder.iterdir() if p.name.startswith("cache-")]
    assert (folder / META_FILE).is_file()

    # Three steps of loss, a checkpoint at step 2, and the adapter where the manifest says.
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [line["step"] for line in lines if "loss" in line] == [1, 2, 3]
    assert (run_dir / "checkpoints" / "step-000002" / "adapter_model.safetensors").is_file()
    adapter = run_dir / manifest.outputs["adapter"]
    assert adapter == run_dir / "checkpoints" / "final"
    weights = load_file(adapter / "adapter_model.safetensors")
    assert weights and all("lora_" in name for name in weights)
    # LoRA's B starts at zero; trained, it no longer is.
    assert any(t.abs().sum() > 0 for name, t in weights.items() if "lora_B" in name)
    settings = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert (settings["r"], settings["lora_alpha"]) == (16, 32)
    assert sorted(settings["target_modules"]) == sorted(LORA_TARGETS)

    evaluate = [
        *("eval", "run", "--backend", "hf", "--model", tiny_model_ref, "--adapter", str(adapter)),
        *("--split", "val", "--limit", "10", "--run-id", "sft-eval", "--workers", "2"),
        "--allow-dirty",
    ]
    evaluated = cli_runner.invoke(cli.app, evaluate)
    assert evaluated.exit_code == 0, evaluated.output

    stored = read_stored(runs / "sft-eval", spider_mini)
    assert stored.manifest.status == "finished"
    assert stored.manifest.adapter_ref == stored.manifest.config["adapter"] == str(adapter)
    assert len(stored.scores) == 10
    assert all(score.ex_official is not None for score in stored.scores)
    for reply in stored.generations.values():
        assert (reply.model_ref, reply.adapter_ref) == (tiny_model_ref, str(adapter))
        assert reply.text and reply.tokens_out >= 1
