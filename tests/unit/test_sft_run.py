"""``forge train sft`` with a stand-in backend: what an SFT run leaves in its folder and its
manifest, and what it refuses before any model loads. The smoke test trains for real."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from datasets import Dataset, load_from_disk
from transformers import TrainerCallback, TrainerControl, TrainerState
from typer.testing import CliRunner

import forge.train
from forge import cli
from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.data.splits import load_train
from forge.ops.adapters import adapter_sha256, digest_output
from forge.ops.hub import FINAL, LocalRunStore
from forge.ops.manifest import config_hash
from forge.train import sft
from forge.train.backends import ADAPTER_DIR, Precision
from forge.train.config import LoraConfig, SftConfig, TrainConfig, parse_train_config
from forge.train.sft import RunExists, open_rows, platform_for, run_sft
from forge.train.sft_data import build_sft_dataset, read_meta, save_sft
from tests.fakes import WordTokenizer

ROOT = Path(__file__).resolve().parents[2]
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
                (saved / "adapter_config.json").write_text('{"r": 16}')
                (saved / "adapter_model.safetensors").write_text(f"step {state.global_step}")
                for callback in callbacks:
                    callback.on_save(args, state, control)
                control.should_save = False
        for callback in callbacks:
            callback.on_train_end(args, state, control)
        adapter = out / ADAPTER_DIR
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text('{"r": 16}')
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
    # With the digest of each checkpoint the run keeps, which an eval of one records too.
    half = "checkpoints/step-000002"
    assert manifest.outputs == {
        "config": "config.yaml",
        "metrics": "metrics.jsonl",
        "adapter": "checkpoints/final",
        digest_output(FINAL): adapter_sha256(run.adapter),
        digest_output(half): adapter_sha256(root / half),
    }
    assert manifest.outputs[digest_output(FINAL)] != manifest.outputs[digest_output(half)]

    # The data hash: the rows' own, the name of the folder they were saved to, beside the
    # Spider files' in data_manifest_hash.
    meta = read_meta(run.sft_data)
    assert run.sft_data == tmp_path / "sft" / meta.hash
    assert (meta.prompt_version, meta.tokenizer) == ("p1", MODEL)
    assert manifest.config["sft_data"] == {
        "hash": meta.hash,
        "rows": meta.counts.rows,
        "over_max_length": 0,
        "trained": meta.counts.rows,
    }
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


def _lengths(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {row["example_id"]: sft.row_tokens(WordTokenizer(), row) for row in rows}


@pytest.mark.phase(4)
def test_rows_over_max_length_are_left_out_not_cut(spider_mini: Path, tmp_path: Path) -> None:
    everything = FakeBackend()
    whole = _train(tmp_path, everything, run_id="2026-11-15-tiny-sft-all1234")
    lengths = _lengths(everything.rows)
    # A limit some rows pass, one of them by exactly its length.
    limit = sorted(lengths.values())[len(lengths) // 2]
    over = sorted(example_id for example_id, n in lengths.items() if n > limit)
    assert over and limit in lengths.values()

    backend = FakeBackend()
    run = _train(tmp_path, backend, _config(sft={"max_steps": 3, "max_length": limit}))

    # The rows that fit reach the trainer as they were saved, none of them shortened, and
    # the manifest counts the rest.
    fitting = [row for row in everything.rows if lengths[row["example_id"]] <= limit]
    assert backend.rows == fitting
    assert run.manifest.config["sft_data"] == {
        **whole.manifest.config["sft_data"],
        "over_max_length": len(over),
        "trained": len(fitting),
    }
    # Counted as TRL's SFTTrainer encodes a row: prompt and reply through the chat template.
    row = everything.rows[0]
    tokenizer = WordTokenizer()
    assert sft.row_tokens(tokenizer, row) == len(
        tokenizer.apply_chat_template(row["prompt"] + row["completion"])["input_ids"]
    )

    shortest = min(lengths.values())
    with pytest.raises(ValueError, match=f"all {len(lengths)} SFT rows are over max_length"):
        _train(
            tmp_path,
            FakeBackend(),
            _config(sft={"max_length": shortest - 1}),
            run_id="2026-11-15-tiny-sft-none123",
        )


@pytest.mark.phase(4)
def test_subset_is_drawn_by_hash_and_counted(spider_mini: Path, tmp_path: Path) -> None:
    everything = FakeBackend()
    _train(tmp_path, everything, run_id="2026-11-15-tiny-sft-all1234")
    ids = [row["example_id"] for row in everything.rows]
    size = len(ids) // 3

    backend = FakeBackend()
    run = _train(tmp_path, backend, _config(sft={"max_steps": 3, "subset_rows": size}))

    # The ids that hash lowest with the run's seed, 7, kept in the saved order.
    def rank(example_id: str) -> str:
        return hashlib.sha256(f"7:{example_id}".encode()).hexdigest()

    drawn = set(sorted(ids, key=rank)[:size])
    assert backend.rows == [row for row in everything.rows if row["example_id"] in drawn]
    assert run.manifest.config["sft_data"]["trained"] == size
    assert run.manifest.config["sft"]["subset_rows"] == size
    # Another seed draws other rows. A row's draw doesn't depend on the others: without a
    # row the subset left out, the same rows are drawn.
    assert sft.subset(ids, size, 8) != sft.subset(ids, size, 7)
    undrawn = next(example_id for example_id in ids if example_id not in drawn)
    rest = [example_id for example_id in ids if example_id != undrawn]
    assert {rest[position] for position in sft.subset(rest, size, 7)} == drawn

    with pytest.raises(ValueError, match=f"subset_rows {len(ids) + 1} is more than the {len(ids)}"):
        _train(
            tmp_path,
            FakeBackend(),
            _config(sft={"subset_rows": len(ids) + 1}),
            run_id="2026-11-15-tiny-sft-big1234",
        )


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


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


@pytest.mark.phase(4)
def test_sft_run_checks_its_tree_once(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A clean one-commit checkout, the Forge root, that the run starts in.
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("FORGE_ROOT", str(repo))
    monkeypatch.delenv("FORGE_LEDGER", raising=False)
    monkeypatch.delenv("FORGE_DEV_USAGE", raising=False)
    monkeypatch.chdir(repo)
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "T"],
    ):
        _git(repo, *args)
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "one")

    class TreeChanges(FakeBackend):
        """Changes the checkout while the model loads, as another process in it can."""

        def __init__(self, change: Callable[[], None]) -> None:
            super().__init__()
            self.change = change

        def load(
            self, model_ref: str, lora: LoraConfig, *, load_in_4bit: bool, seed: int
        ) -> tuple[Any, Any]:
            self.change()
            return super().load(model_ref, lora, load_in_4bit=load_in_4bit, seed=seed)

    def train(change: Callable[[], None]) -> RunManifest:
        checked = _git(repo, "rev-parse", "HEAD")
        run = run_sft(
            _config(),
            runs_root=tmp_path / "runs",
            sft_root=tmp_path / "sft",
            backend=TreeChanges(change),
            env={},
        )
        # The run's id and its manifest name the commit checked before the load.
        assert run.manifest.git_sha == checked, "the manifest names a later commit"
        assert run.manifest.run_id.endswith(checked[:7])
        assert run.manifest.status == "finished"
        return run.manifest

    def commit() -> None:
        (repo / "a.txt").write_text("two\n")
        _git(repo, "commit", "-q", "-am", "two")

    # A commit lands during the load, and then a new file: neither fails the run late.
    first = train(commit)
    second = train(lambda: (repo / "b.txt").write_text("new\n"))
    assert first.git_sha != second.git_sha


@pytest.mark.phase(4)
def test_a_kernel_run_takes_its_hosts_platform(spider_mini: Path, tmp_path: Path) -> None:
    # On a T4 in a kernel job, the run is the host's, which kaggle/entry.py names.
    run = _train(tmp_path, FakeBackend(device="cuda"), host="kaggle")
    assert (run.manifest.platform, run.manifest.status) == ("kaggle", "finished")
    assert run.manifest.config["trainer"]["device"] == "cuda"


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
    # The checkpoint it saved before the failure, by its digest.
    saved = "checkpoints/step-000002"
    assert manifest.outputs == {
        "config": "config.yaml",
        "metrics": "metrics.jsonl",
        "error": "error.txt",
        digest_output(saved): adapter_sha256(root / saved),
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
def test_sft_run_keeps_its_checkpoint_half_way(spider_mini: Path, tmp_path: Path) -> None:
    # FG-404's main run: the checkpoint at 50% and the final adapter, both for val to score.
    config = _config(
        sft={"max_steps": 6}, ops={"checkpoint_every": 2, "keep_checkpoints_at": [0.5]}
    )
    run = _train(tmp_path, FakeBackend(steps=6), config)

    folders = sorted(p.name for p in (run.root / "checkpoints").iterdir())
    # Step 3 outlives retention, which keeps the latest two of the cadence's beside it.
    assert folders == ["final", "step-000003", "step-000004", "step-000006"]
    assert (run.root / "checkpoints/step-000003/adapter_model.safetensors").read_text() == "step 3"
    kept = ["final", "step-000003", "step-000004", "step-000006"]
    assert run.manifest.outputs == {
        "config": "config.yaml",
        "metrics": "metrics.jsonl",
        "adapter": "checkpoints/final",
        "kept_at_0.5": "checkpoints/step-000003",
        **{
            digest_output(f"checkpoints/{name}"): adapter_sha256(run.root / "checkpoints" / name)
            for name in kept
        },
    }
    # Kept checkpoints change no weight, so the config hash is a run's without them.
    plain = _train(
        tmp_path,
        FakeBackend(steps=6),
        _config(sft={"max_steps": 6}, ops={"checkpoint_every": 2}),
        run_id="2026-11-15-tiny-sft-def5678",
    )
    assert plain.manifest.config_hash == run.manifest.config_hash
    assert "kept_at_0.5" not in plain.manifest.outputs


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
        f"trained on {meta.counts.rows} rows; 0 were over max_length 2048 and left out",
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


@pytest.mark.phase(4)
def test_training_commands_name_the_cpu_extra_they_miss(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The packages the commands name are the cpu extra's, as pyproject.toml lists them.
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = project["project"]["optional-dependencies"]["cpu"]
    names = tuple(re.split(r"[<>=!~ \[;]", spec, maxsplit=1)[0] for spec in extra)
    assert names == cli.CPU_EXTRA
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    config = tmp_path / "sft.yaml"
    config.write_text(f"run: {{kind: sft, model_ref: '{MODEL}', device: cpu}}\nsft: {{}}\n")
    commands = {
        "build-sft": ["data", "build-sft", "--prompt-version", "p1", "--tokenizer", MODEL],
        "train": ["train", "sft", "--config", str(config), "--run-id", RUN, "--allow-dirty"],
        "promote": ["runs", "promote", RUN, "--config", str(config), "--eval", "e"],
    }

    def without(package: str, *names: str, fresh: Sequence[str] = ()) -> None:
        with monkeypatch.context() as patch:
            # None in sys.modules makes `import <package>` fail as it does where it isn't
            # installed; the modules in ``fresh`` import it at their top, so they are
            # imported again.
            patch.setitem(sys.modules, package, None)
            for module in fresh:
                patch.delitem(sys.modules, module)
                patch.delattr(forge.train, module.rpartition(".")[2])
            for name in names:
                result = CliRunner().invoke(cli.app, commands[name])
                assert isinstance(result.exception, SystemExit), (name, result.exception)
                assert (name, result.exit_code, result.stderr) == (
                    name,
                    1,
                    f"error: {package} is not installed; it comes with the cpu extra: run "
                    "`make setup`\n",
                )

    # datasets, which the SFT modules import at their top, and torch, which the backend
    # imports once the run is under way: one error line either way, never a traceback.
    without("datasets", *commands, fresh=("forge.train.sft", "forge.train.sft_data"))
    without("torch", "train")
    assert not (tmp_path / "runs").exists()
