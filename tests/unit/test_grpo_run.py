"""``forge train grpo`` before it trains: the rows it builds from train, the adapter it
starts from, and the runs it refuses to start or resume, with a stand-in backend. The
smoke tests train for real."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.data.prompts import build_prompt
from forge.data.splits import load_train, load_val
from forge.eval.run_files import RunConflict
from forge.ops.hub import LocalRunStore, RunNotFound
from forge.ops.manifest import code_sha
from forge.train.backends import Precision
from forge.train.config import LoraConfig, TrainConfig, parse_train_config
from forge.train.grpo import grpo_rows, run_grpo, start_adapter
from forge.train.sft import RunExists
from tests.fakes import WordTokenizer, sft_run_folder

MODEL = "org/tiny@0123456789abcdef0123456789abcdef01234567"
RUN = "2026-11-20-tiny-grpo-abc1234"
SFT = "2026-11-19-tiny-sft-abc1234"


class FakeBackend:
    """A ``TrainBackend`` that loads a stand-in and must never train."""

    name = "fake"
    device = "cpu"
    precision = Precision("float32", use_cpu=True)

    def __init__(self) -> None:
        self.loaded: list[str | Path | None] = []

    def load(
        self,
        model_ref: str,
        lora: LoraConfig,
        *,
        load_in_4bit: bool,
        seed: int,
        adapter: str | Path | None = None,
    ) -> tuple[Any, Any]:
        self.loaded.append(adapter)
        return object(), WordTokenizer()

    def sft(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("a GRPO run trains no SFT")

    def grpo(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("these runs are refused before they train")


def _config(**grpo: Any) -> TrainConfig:
    return parse_train_config({"run": {"kind": "grpo", "model_ref": MODEL}, "grpo": grpo})


@pytest.mark.phase(5)
def test_grpo_rows_are_train_prompts_with_their_columns(spider_mini: Path) -> None:
    rows, meta = grpo_rows(_config(), WordTokenizer(), data_dir=spider_mini)

    train = {example.id: example for example in load_train()}
    assert rows.column_names == ["prompt", "example_id", "db_id", "gold_sql", "hardness"]
    assert len(rows) == meta["rows"] == len(train) == 40
    for row in rows:
        example = train[row["example_id"]]
        assert row["prompt"] == list(build_prompt(example, "p1").messages)
        assert (row["db_id"], row["gold_sql"], row["hardness"]) == (
            example.db_id,
            example.gold_sql,
            example.hardness,
        )
    # Train alone: no val database's question reaches the rewards.
    assert not set(rows["db_id"]) & {example.db_id for example in load_val()}
    assert meta == {**meta, "train": 40, "over_limit": 0, "duplicates": 0}
    again, same = grpo_rows(_config(), WordTokenizer(), data_dir=spider_mini)
    assert same["hash"] == meta["hash"] and again.to_list() == rows.to_list()

    # A prompt still over max_prompt_length after the overflow policy is left out and
    # counted, here rather than in TRL, which no longer has the setting; so is one the
    # policy cut down to hide a table its gold query reads.
    shortest = min(
        len(WordTokenizer().apply_chat_template(row["prompt"])["input_ids"]) for row in rows
    )
    short, counted = grpo_rows(
        _config(max_prompt_length=shortest), WordTokenizer(), data_dir=spider_mini
    )
    assert 0 < len(short) < len(rows) and counted["over_limit"] > 0
    assert len(rows) - len(short) == counted["over_limit"] + counted["hidden_gold_tables"]
    assert counted["hash"] != meta["hash"]


@pytest.mark.phase(5)
def test_start_adapter_checks_the_run_it_comes_from(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    runs = tmp_path / "runs"
    made = sft_run_folder(runs, SFT, MODEL, ["checkpoints/best"])
    best = runs / SFT / "checkpoints" / "best"

    start = start_adapter(str(best), MODEL)
    assert (start.folder, start.parent_run) == (best, SFT)
    assert start.sha256 == made.outputs["adapter_sha256:checkpoints/best"]
    # Another model's checkpoint, and other files than the run saved there, are refused.
    with pytest.raises(ValueError, match=f"which trained {MODEL}, and this run trains org/other"):
        start_adapter(str(best), "org/other")
    (best / "adapter_model.safetensors").write_text("other weights")
    with pytest.raises(ValueError, match=f"holds other files than run {SFT} saved as"):
        start_adapter(str(best), MODEL)
    # A run made before runs recorded digests: the files there, and a warning saying so.
    store = LocalRunStore(runs)
    store.put_manifest(made.model_copy(update={"outputs": {"adapter": "checkpoints/final"}}))
    with caplog.at_level(logging.WARNING, logger="forge.train.grpo"):
        assert start_adapter(str(best), MODEL).parent_run == SFT
    assert f"run {SFT} records no digest of checkpoints/best" in caplog.text
    # A checkpoint needs its run's manifest beside it; any other folder is just an adapter.
    (runs / SFT / "manifest.json").unlink()
    with pytest.raises(RunNotFound, match="which needs the run's manifest beside it"):
        start_adapter(str(best), MODEL)
    loose = tmp_path / "adapter"
    best.rename(loose)
    assert start_adapter(str(loose), MODEL).parent_run is None
    with pytest.raises(FileNotFoundError, match="there is no such folder here"):
        start_adapter(str(best), MODEL)


def _stored(tmp_path: Path, **changes: Any) -> None:
    """A stored GRPO run that the time guard stopped, made on this tree and data."""
    fields: dict[str, Any] = {
        "run_id": RUN,
        "kind": "grpo",
        "git_sha": code_sha(allow_dirty=True),
        "config": {},
        "config_hash": "1" * 64,
        "data_manifest_hash": data_manifest_hash(),
        "model_ref": MODEL,
        "platform": "local-cpu",
        "seed": 0,
        "started_at": datetime(2026, 11, 20, 8, 0, tzinfo=UTC),
        "status": "stopped_time_guard",
        **changes,
    }
    LocalRunStore(tmp_path / "runs").put_manifest(RunManifest(**fields))


@pytest.mark.phase(5)
def test_grpo_run_refuses_before_it_trains(spider_mini: Path, tmp_path: Path) -> None:
    backend = FakeBackend()

    def run(config: TrainConfig | None = None, **options: Any) -> None:
        run_grpo(
            config or _config(),
            allow_dirty=True,
            runs_root=tmp_path / "runs",
            cache_root=tmp_path / "cache",
            backend=backend,
            env={},
            **options,
        )

    sft = parse_train_config({"run": {"kind": "sft", "model_ref": MODEL}, "sft": {}})
    with pytest.raises(ValueError, match="forge train grpo trains grpo runs"):
        run(sft, run_id=RUN)
    with pytest.raises(ValueError, match="name the run to resume"):
        run(resume=True)
    with pytest.raises(RunNotFound, match=f"no run {RUN}"):
        run(run_id=RUN, resume=True)
    for changes, message in (
        ({"kind": "eval"}, f"run {RUN} is a eval run; forge train grpo resumes grpo runs"),
        ({"status": "finished"}, f"run {RUN} has finished"),
        ({"git_sha": "f" * 40}, f"run {RUN} was made with commit {'f' * 40}"),
    ):
        _stored(tmp_path, **changes)
        with pytest.raises(RunConflict, match=message):
            run(run_id=RUN, resume=True)
    # Nothing loaded for any of those, and a run already there isn't started again.
    assert backend.loaded == []
    with pytest.raises(RunExists, match="pass --resume or a new --run-id"):
        run(run_id=RUN)
    assert backend.loaded == []

    # Another config's run is refused once its hash is known, before the folder changes.
    _stored(tmp_path)
    before = sorted(path.name for path in (tmp_path / "runs" / RUN).iterdir())
    with pytest.raises(RunConflict, match=f"run {RUN} was made with config hash {'1' * 64}"):
        run(run_id=RUN, resume=True)
    assert backend.loaded == [None]
    assert sorted(path.name for path in (tmp_path / "runs" / RUN).iterdir()) == before
