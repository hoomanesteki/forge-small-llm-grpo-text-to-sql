"""``forge train sft`` with a stand-in backend: what an SFT run leaves in its folder and its
manifest, and what it refuses before any model loads. The smoke test trains for real."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from datasets import Dataset, load_from_disk
from transformers import TrainerCallback, TrainerControl, TrainerState
from typer.testing import CliRunner

from forge import cli
from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.data.splits import load_train
from forge.ops.hub import LocalRunStore
from forge.ops.manifest import config_hash
from forge.train import sft
from forge.train.backends import ADAPTER_DIR, Precision
from forge.train.config import LoraConfig, SftConfig, TrainConfig, parse_train_config
from forge.train.sft import RunExists, open_rows, platform_for, run_sft
from forge.train.sft_data import build_sft_dataset, read_meta, save_sft
from tests.fakes import WordTokenizer

RUN = "2026-11-15-tiny-sft-abc1234"
MODEL = "org/tiny@0123456789abcdef0123456789abcdef01234567"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeBackend:
    """A ``TrainBackend`` that trains nothing: ``sft`` drives the callbacks through
    ``steps`` steps as a trainer does, saving where they ask, and writes a stand-in
    adapter. Each step takes ``step_s`` on ``clock``; step ``fail_at`` raises."""

    name = "fake"
    precision = Precision("float32", use_cpu=True)

    def __init__(
        self,
        *,
        device: str = "cpu",
        steps: int = 3,
        fail_at: int | None = None,
        clock: Clock | None = None,
        step_s: float = 1.0,
    ) -> None:
        self.device = device
        self.steps = steps
        self.fail_at = fail_at
        self.clock = clock
        self.step_s = step_s
        self.loaded: list[tuple[str, LoraConfig, bool, int]] = []
        self.rows: list[dict[str, Any]] = []

    def load(
        self, model_ref: str, lora: LoraConfig, *, load_in_4bit: bool, seed: int
    ) -> tuple[Any, Any]:
        self.loaded.append((model_ref, lora, load_in_4bit, seed))
        return object(), WordTokenizer()

    def sft(
        self,
        model: Any,
        tok: Any,
        dataset: Dataset,
        cfg: SftConfig,
        callbacks: Sequence[TrainerCallback],
        *,
        output_dir: str | Path,
        seed: int,
    ) -> str:
        self.rows = dataset.to_list()
        out = Path(output_dir)
        args: Any = SimpleNamespace(output_dir=str(out))
        state, control = TrainerState(max_steps=self.steps), TrainerControl()
        for callback in callbacks:
            callback.on_train_begin(args, state, control)
        while state.global_step < self.steps and not control.should_training_stop:
            if self.clock is not None:
                self.clock.now += self.step_s
            state.global_step += 1
            if state.global_step == self.fail_at:
                raise RuntimeError(f"the loss is NaN at step {state.global_step}")
            for callback in callbacks:
                callback.on_step_end(args, state, control)
            for callback in callbacks:
                callback.on_log(args, state, control, logs={"loss": 1.0 / state.global_step})
            if control.should_save:
                saved = out / f"checkpoint-{state.global_step}"
                saved.mkdir(parents=True)
                (saved / "adapter_model.safetensors").write_text(f"step {state.global_step}")
                for callback in callbacks:
                    callback.on_save(args, state, control)
                control.should_save = False
        for callback in callbacks:
            callback.on_train_end(args, state, control)
        adapter = out / ADAPTER_DIR
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_text(f"{state.global_step} steps")
        return str(adapter)

    def grpo(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("an SFT run trains no GRPO")


def _config(**sections: Any) -> TrainConfig:
    return parse_train_config(
        {
            "run": {"kind": "sft", "model_ref": MODEL, "seed": 7},
            "sft": {"max_steps": 3, "logging_steps": 1},
            "ops": {"checkpoint_every": 2},
            **sections,
        }
    )


def _train(
    tmp_path: Path, backend: FakeBackend, config: TrainConfig | None = None, **kwargs: Any
) -> sft.SftRun:
    return run_sft(
        config or _config(),
        run_id=kwargs.pop("run_id", RUN),
        allow_dirty=True,
        runs_root=tmp_path / "runs",
        sft_root=tmp_path / "sft",
        backend=backend,
        env={},
        **kwargs,
    )


def _stored(tmp_path: Path, run_id: str = RUN) -> RunManifest:
    return LocalRunStore(tmp_path / "runs").get_manifest(run_id)


@pytest.mark.phase(4)
def test_sft_run_writes_its_adapter_and_manifest(spider_mini: Path, tmp_path: Path) -> None:
    backend = FakeBackend()
    config = _config()

    run = _train(tmp_path, backend, config)

    root = tmp_path / "runs" / RUN
    assert run.root == root and run.stopped_at is None
    assert run.adapter == root / "checkpoints" / "final"
    assert (run.adapter / "adapter_model.safetensors").read_text() == "3 steps"
    assert backend.loaded == [(MODEL, config.lora, False, 7)]
    manifest = run.manifest
    assert manifest == _stored(tmp_path)
    assert (manifest.kind, manifest.status, manifest.run_id) == ("sft", "finished", RUN)
    assert (manifest.model_ref, manifest.seed, manifest.platform) == (MODEL, 7, "local-cpu")
    assert manifest.gpu is None and manifest.ended_at is not None
    assert manifest.outputs == {
        "config": "config.yaml",
        "metrics": "metrics.jsonl",
        "adapter": "checkpoints/final",
    }

    # The data hash: the rows' own, the name of the folder they were saved to, beside the
    # Spider files' in data_manifest_hash.
    meta = read_meta(run.sft_data)
    assert run.sft_data == tmp_path / "sft" / meta.hash
    assert (meta.prompt_version, meta.tokenizer) == ("p1", MODEL)
    assert manifest.config["sft_data"] == {"hash": meta.hash, "rows": meta.counts.rows}
    assert manifest.data_manifest_hash == data_manifest_hash(spider_mini)
    # The config hash covers what can change the adapter, the trainer's precision among it,
    # and not the ops section.
    assert manifest.config_hash == config_hash(manifest.config)
    assert manifest.config["trainer"] == {"backend": "fake", "device": "cpu", "dtype": "float32"}
    assert config.sft is not None
    assert manifest.config["sft"] == config.sft.model_dump(mode="json")
    assert manifest.config["lora"]["target"] == list(config.lora.target)
    assert "ops" not in manifest.config
    stored = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    assert stored == manifest.config

    # It trained on the saved rows, all of them from the train split.
    assert backend.rows == load_from_disk(str(run.sft_data)).to_list()
    assert len(backend.rows) == meta.counts.rows
    assert {row["example_id"] for row in backend.rows} <= {e.id for e in load_train()}
    # A checkpoint every 2 steps, and the trainer's logs, one line each.
    assert LocalRunStore(tmp_path / "runs").latest_checkpoint(RUN) == "checkpoints/step-000002"
    lines = (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2, 3]

    # The same rows the next time are the folder already saved, under the same hash.
    again = _train(tmp_path, FakeBackend(), config, run_id="2026-11-15-tiny-sft-def5678")
    assert again.sft_data == run.sft_data
    assert again.manifest.config_hash == manifest.config_hash
    assert [p.name for p in (tmp_path / "sft").iterdir()] == [meta.hash]


@pytest.mark.phase(4)
def test_sft_run_refuses_before_any_model_loads(spider_mini: Path, tmp_path: Path) -> None:
    (tmp_path / "runs" / RUN).mkdir(parents=True)
    backend = FakeBackend()
    with pytest.raises(RunExists, match=f"run {RUN} is in"):
        _train(tmp_path, backend)
    cuda = FakeBackend(device="cuda")
    with pytest.raises(ValueError, match=r"no platform its manifest can name \(ADR-0016\)"):
        _train(tmp_path, cuda, run_id="2026-11-15-tiny-sft-def5678")
    grpo = parse_train_config({"run": {"kind": "grpo", "model_ref": MODEL}, "grpo": {}})
    with pytest.raises(ValueError, match="forge train sft trains sft runs"):
        _train(tmp_path, backend, grpo, run_id="2026-11-15-tiny-sft-def5678")
    with pytest.raises(ValueError, match="names a folder"):
        _train(tmp_path, backend, run_id="../elsewhere")
    assert backend.loaded == cuda.loaded == []
    assert not (tmp_path / "sft").exists()


@pytest.mark.phase(4)
def test_local_platforms() -> None:
    assert platform_for("cpu") == "local-cpu"
    assert platform_for("mps") == "local-mps"
    with pytest.raises(ValueError, match="through a Kaggle job"):
        platform_for("cuda")


@pytest.mark.phase(4)
def test_failed_training_leaves_a_failed_run(spider_mini: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="the loss is NaN at step 3"):
        _train(tmp_path, FakeBackend(fail_at=3))

    root = tmp_path / "runs" / RUN
    manifest = _stored(tmp_path)
    assert manifest.status == "failed" and manifest.ended_at is not None
    assert manifest.outputs == {
        "config": "config.yaml",
        "metrics": "metrics.jsonl",
        "error": "error.txt",
    }
    error = (root / "error.txt").read_text(encoding="utf-8")
    assert error.startswith("Traceback") and "RuntimeError: the loss is NaN at step 3" in error
    assert not (root / "checkpoints" / "final").exists()
    # The checkpoint before the failure stays.
    assert LocalRunStore(tmp_path / "runs").latest_checkpoint(RUN) == "checkpoints/step-000002"


@pytest.mark.phase(4)
def test_time_guard_stop_keeps_its_checkpoint_and_no_final(
    spider_mini: Path, tmp_path: Path
) -> None:
    clock = Clock()
    # 60 s, and steps of 25 s: after the second only 10 s are left, less than a step.
    config = _config(sft={"max_steps": 5}, ops={"time_guard_minutes": 1.0})
    run = _train(tmp_path, FakeBackend(steps=5, clock=clock, step_s=25), config, clock=clock)

    assert run.stopped_at == 2 and run.adapter is None
    assert run.manifest.status == "stopped_time_guard" == _stored(tmp_path).status
    assert "adapter" not in run.manifest.outputs
    assert not (run.root / "checkpoints" / "final").exists()
    checkpoint = run.root / "checkpoints" / "step-000002" / "adapter_model.safetensors"
    assert checkpoint.read_text() == "step 2"


@pytest.mark.phase(4)
def test_rows_must_hash_to_their_folder(spider_mini: Path, tmp_path: Path) -> None:
    def build(version: str) -> Any:
        return build_sft_dataset(version, MODEL, lambda prompt: 1)

    p1, p2 = build("p1"), build("p2")
    folder = save_sft(p1, tmp_path)
    assert open_rows(folder, p1.meta).to_list() == p1.rows
    # Rows that are not the ones the meta hashed, as a folder edited by hand would hold.
    with pytest.raises(ValueError, match="not to its name; remove the folder"):
        open_rows(folder, p2.meta)


@pytest.mark.phase(4)
def test_train_sft_command(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    chosen: list[tuple[str, str]] = []

    def select_backend(name: str, *, device: str, max_seq_length: int) -> FakeBackend:
        chosen.append((name, device))
        assert max_seq_length == 2048  # ADR-0010's, not TRL's 1,024
        return FakeBackend()

    monkeypatch.setattr(sft, "select_backend", select_backend)
    config = tmp_path / "sft.yaml"
    config.write_text(
        f"run: {{kind: sft, model_ref: '{MODEL}', backend: trl, device: cpu}}\n"
        "sft: {max_steps: 3}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    args = ["train", "sft", "--config", str(config), "--sft-dir", str(tmp_path / "sft")]

    done = runner.invoke(cli.app, [*args, "--run-id", RUN, "--allow-dirty"])

    assert done.exit_code == 0, done.output
    assert chosen == [("trl", "cpu")]
    meta = read_meta(next((tmp_path / "sft").iterdir()))
    root = tmp_path / "runs" / RUN
    assert done.stdout.splitlines() == [
        f"run {RUN}: {root}",
        f"sft data {tmp_path / 'sft' / meta.hash}: {meta.counts.rows} rows, hash {meta.hash}",
        "platform local-cpu, backend fake on cpu in float32",
        f"adapter {root / 'checkpoints' / 'final'}",
    ]
    taken = runner.invoke(cli.app, [*args, "--run-id", RUN, "--allow-dirty"])
    assert taken.exit_code == 1 and f"error: run {RUN} is in {root} already" in taken.stderr

    config.write_text(f"run: {{kind: sft, model_ref: '{MODEL}'}}\nsft: {{max_step: 3}}\n")
    misspelt = runner.invoke(cli.app, [*args, "--allow-dirty"])
    assert misspelt.exit_code == 1
    assert "sft.max_step: Extra inputs are not permitted" in misspelt.stderr
    assert chosen == [("trl", "cpu")]
