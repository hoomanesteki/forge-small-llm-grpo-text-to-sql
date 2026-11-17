"""Training configs from YAML: what loads, and what is refused before any model loads."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from forge.data.prompts import MAX_PROMPT_TOKENS
from forge.eval.config import EvalConfig, read_config_file, read_config_sections
from forge.train.backends import precision_for
from forge.train.config import (
    LORA_TARGETS,
    GrpoConfig,
    TrainConfig,
    load_train_config,
    parse_train_config,
)

ROOT = Path(__file__).resolve().parents[2]
# Each SFT config, and the configs/models.yaml entry whose model it trains.
SFT_MODELS = {
    "tiny-smoke.yaml": ("tiny",),
    "pilot-qwen25coder05b-mps.yaml": ("pilot",),
    "qwen3-1p7b-pilot.yaml": ("candidates", "qwen3-1-7b"),
    "qwen3-4b-main.yaml": ("candidates", "qwen3-4b"),
}
LOCAL_PILOT = "pilot-qwen25coder05b-mps.yaml"
# The Kaggle runs' GPU hours, plan section 3.8: the SFT pilot and the main run.
KAGGLE_HOURS = {"qwen3-1p7b-pilot.yaml": 1.0, "qwen3-4b-main.yaml": 2.5}

# Plan section 3.10's GRPO file, without the sections later stories add.
GRPO_FILE = """\
run:
  kind: grpo
  model_ref: Qwen/Qwen3-4B@0123abc
  seed: 42
data:
  prompt_version: p1
  split: train
lora:
  r: 16
  alpha: 32
  dropout: 0.0
  target: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
grpo:
  num_generations: 6
  max_prompt_length: 1536
  max_completion_length: 256
  learning_rate: 5e-6
  per_device_train_batch_size: 6
  gradient_accumulation_steps: 4
  max_steps: 600
  beta: library_default
ops:
  checkpoint_every: 50
  time_guard_minutes: 675
"""

SFT_FILE = """\
run: {kind: sft, model_ref: trl-internal-testing/tiny-Qwen2ForCausalLM-2.5, device: cpu}
sft: {max_steps: 3, per_device_train_batch_size: 2, gradient_accumulation_steps: 1}
ops: {est_gpu_hours: 0.5}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _sft(**sections: Any) -> dict[str, Any]:
    return {"run": {"kind": "sft", "model_ref": "org/model"}, "sft": {}, **sections}


def _grpo(**grpo: Any) -> dict[str, Any]:
    return {"run": {"kind": "grpo", "model_ref": "org/model"}, "grpo": grpo}


@pytest.mark.phase(4)
def test_config_validation(tmp_path: Path) -> None:
    config = load_train_config(_write(tmp_path, GRPO_FILE))
    assert config.run.kind == "grpo" and config.run.seed == 42 and config.sft is None
    assert config.grpo is not None
    # 5e-6 is text to PyYAML; the loader reads the number it means.
    assert config.grpo.learning_rate == 5e-6
    assert config.grpo.beta == "library_default"
    assert config.grpo.generation_batch == 24
    assert config.lora.target == LORA_TARGETS
    assert config.ops.time_guard_minutes == 675.0
    assert config.max_seq_length == 1536 + 256

    sft = load_train_config(_write(tmp_path, SFT_FILE))
    assert sft.sft is not None and sft.sft.max_steps == 3
    assert (sft.sft.learning_rate, sft.sft.lr_scheduler_type) == (2e-4, "cosine")  # ADR-0010
    assert sft.run.device == "cpu" and sft.data.prompt_version == "p1"
    assert sft.ops.est_gpu_hours == 0.5 and sft.ops.checkpoint_every == 50

    refused = [
        (
            GRPO_FILE.replace("max_completion_length: 256", "max_completion_length: 513"),
            r"grpo\.max_completion_length: Input should be less than or equal to 512",
        ),
        (
            GRPO_FILE.replace("num_generations: 6", "num_generations: 5"),
            r"num_generations 5 must divide the generation batch.* = 6 x 4 = 24",
        ),
        (
            GRPO_FILE.replace("checkpoint_every", "checkpoint_evry"),
            r"ops\.checkpoint_evry: Extra inputs are not permitted",
        ),
        (
            GRPO_FILE.replace("split: train", "split: val"),
            r"data\.split: training reads the train split only, never 'val'",
        ),
        (
            GRPO_FILE.replace("max_steps: 600", "max_steps: true"),
            r"grpo\.max_steps: Input should be a valid integer",
        ),
        (GRPO_FILE + "ops: {checkpoint_every: 10}\n", "'ops' is set twice"),
    ]
    for text, message in refused:
        with pytest.raises(ValueError, match=message):
            load_train_config(_write(tmp_path, text))


@pytest.mark.phase(4)
@pytest.mark.parametrize(
    ("data", "message"),
    [
        (_sft(lora={"r": 0}), r"lora\.r: Input should be greater than or equal to 1"),
        (_sft(lora={"dropout": 1.0}), r"lora\.dropout: Input should be less than 1"),
        (_sft(lora={"target": []}), r"lora\.target: LoRA needs at least one target module"),
        (_sft(lora={"target": ["q_proj", "q_proj"]}), "target names q_proj more than once"),
        (_sft(data={"prompt_version": "p9"}), "unknown prompt version 'p9'"),
        (_sft(sft={"learning_rate": 0}), r"sft\.learning_rate: Input should be greater than 0"),
        (_sft(sft={"max_steps": 2.5}), r"sft\.max_steps: Input should be a valid integer"),
        (_sft(sft={"warmup_ratio": 1}), r"sft\.warmup_ratio: Input should be less than 1"),
        (_sft(ops={"time_guard_minutes": float("inf")}), "Input should be a finite number"),
        (_sft(ops={"checkpoint_every": 0}), r"ops\.checkpoint_every: Input should be greater"),
        (_sft(ops={"keep_checkpoints_at": [1.0]}), r"keep_checkpoints_at\.0: Input should be less"),
        (
            _sft(ops={"keep_checkpoints_at": [0.0]}),
            r"keep_checkpoints_at\.0: Input should be greater",
        ),
        (_sft(ops={"keep_checkpoints_at": [0.5, 0.5]}), r"each share is kept once, got \[0\.5,"),
        (_grpo(num_generations=1), r"grpo\.num_generations: Input should be greater than or"),
        (_grpo(max_completion_length=0), r"grpo\.max_completion_length: Input should be greater"),
        (_grpo(beta=-0.1), r"grpo\.beta.*greater than or equal to 0"),
        (_grpo(beta="none"), r"grpo\.beta.*Input should be 'library_default'"),
        (_grpo(gpu_memory_utilization=1.5), r"gpu_memory_utilization: Input should be less"),
        (_grpo(max_prompt_length=0), r"grpo\.max_prompt_length: Input should be greater"),
        ({"run": {"kind": "sft", "model_ref": "org/model"}}, "needs the sft section"),
        ({**_sft(), "grpo": {}}, "run.kind is sft, so the file can't have the grpo section"),
        ({"run": {"kind": "eval", "model_ref": "org/model"}, "sft": {}}, r"run\.kind: Input"),
        ({"run": {"kind": "sft", "model_ref": ""}, "sft": {}}, r"run\.model_ref: String should"),
        ({"sft": {}}, r"run: Field required"),
        (
            _sft(
                run={"kind": "sft", "model_ref": "org/model", "backend": "unsloth", "device": "mps"}
            ),
            "backend unsloth trains on CUDA only, not on device mps",
        ),
        (
            _sft(
                run={"kind": "sft", "model_ref": "org/model", "load_in_4bit": True, "device": "cpu"}
            ),
            "load_in_4bit needs bitsandbytes on CUDA, not device cpu",
        ),
        (_sft(run={"kind": "sft", "model_ref": "org/model", "device": "tpu"}), r"run\.device"),
    ],
)
def test_config_refuses(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_train_config(data, where="test.yaml")


@pytest.mark.phase(4)
def test_config_defaults_are_the_adr_starting_values() -> None:
    config = parse_train_config(_grpo())
    assert isinstance(config, TrainConfig) and config.grpo is not None
    grpo = config.grpo
    assert (grpo.num_generations, grpo.max_completion_length, grpo.max_steps) == (6, 256, 600)
    assert grpo.max_prompt_length == MAX_PROMPT_TOKENS
    assert (config.lora.r, config.lora.alpha, config.lora.dropout) == (16, 32, 0.0)
    assert (config.run.backend, config.run.device, config.run.load_in_4bit) == (
        "auto",
        "auto",
        False,
    )
    # The CUDA-only options are fine where the device may be CUDA.
    cuda = {"kind": "sft", "model_ref": "org/model", "backend": "unsloth", "load_in_4bit": True}
    assert parse_train_config(_sft(run=cuda)).run.load_in_4bit


@pytest.mark.phase(4)
def test_config_file_must_be_a_yaml_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not valid YAML"):
        load_train_config(_write(tmp_path, "run: [kind: sft"))
    with pytest.raises(ValueError, match="must be a mapping of sections"):
        load_train_config(_write(tmp_path, "- run\n- sft\n"))
    with pytest.raises(OSError):
        load_train_config(tmp_path / "missing.yaml")
    # Every wrong field is named, after the file it is in.
    text = SFT_FILE.replace("max_steps: 3", "max_steps: 0").replace("0.5}", "0.5, seed: 1}")
    with pytest.raises(ValueError, match=r"config\.yaml: sft\.max_steps: .*; ops\.seed: Extra"):
        load_train_config(_write(tmp_path, text))


@pytest.mark.phase(4)
def test_a_section_with_nothing_under_it_is_called_empty(tmp_path: Path) -> None:
    # YAML reads `grpo:` alone as null: the file has the section, and it is empty.
    run = "run: {kind: grpo, model_ref: o/m}\n"
    with pytest.raises(
        ValueError,
        match=r"config\.yaml: top level: the grpo section is empty; write its settings under "
        r"it, or `grpo: \{\}` for the starting values$",
    ):
        load_train_config(_write(tmp_path, run + "grpo:\n"))
    config = load_train_config(_write(tmp_path, run + "grpo: {}\n"))
    assert config.grpo == GrpoConfig()
    # The trainer the run doesn't use is null in a dump, as a kernel job's options carry it.
    dumped = config.model_dump(mode="json")
    assert dumped["sft"] is None and parse_train_config(dumped) == config
    # Any other section, each named; run has no starting values to take.
    with pytest.raises(ValueError, match=r": lora: the section is empty; .*; ops: the section"):
        load_train_config(_write(tmp_path, run + "lora:\ngrpo: {}\nops:\n"))
    required = r": run: the section is empty; write its settings under it$"
    with pytest.raises(ValueError, match=required):
        load_train_config(_write(tmp_path, "run:\ngrpo: {}\n"))
    # A section left out is still told apart from an empty one.
    with pytest.raises(ValueError, match=r"run\.kind is grpo, so the file needs the grpo section"):
        load_train_config(_write(tmp_path, run))


@pytest.mark.phase(4)
def test_sft_configs_start_from_adr_0010() -> None:
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    paths = {path.name: path for path in (ROOT / "configs" / "sft").glob("*.yaml")}
    assert sorted(paths) == sorted(SFT_MODELS)
    for name, keys in SFT_MODELS.items():
        config = load_train_config(paths[name])
        sft = config.sft
        assert sft is not None and config.run.kind == "sft", name
        entry = models
        for key in keys:
            entry = entry[key]
        # The model at the commit configs/models.yaml pins, so the runs train what was measured.
        assert config.run.model_ref == f"{entry['repo']}@{entry['revision']}", name
        assert (config.lora.r, config.lora.alpha, config.lora.dropout) == (16, 32, 0.0), name
        assert config.lora.target == LORA_TARGETS, name
        assert (sft.learning_rate, sft.lr_scheduler_type, sft.warmup_ratio) == (
            2e-4,
            "cosine",
            0.03,
        ), name
        # 2,048 tokens, not TRL's 1,024, which would cut long rows' replies (FG-401). The
        # laptop pilot takes 1,536 for its memory, and its longer rows are left out (FG-491).
        length = 1536 if name == LOCAL_PILOT else 2048
        assert config.max_seq_length == sft.max_length == length, name
        assert sft.gradient_checkpointing, name
        if name == "tiny-smoke.yaml":
            # Three steps on the CPU, what make smoke trains, with a checkpoint among them.
            assert (config.run.backend, config.run.device, sft.max_steps) == ("trl", "cpu", 3)
            assert config.ops.checkpoint_every < sft.max_steps
            continue
        # One epoch at an effective batch of 16 rows.
        rows_a_step = sft.per_device_train_batch_size * sft.gradient_accumulation_steps
        assert (sft.num_train_epochs, sft.max_steps, rows_a_step) == (1.0, None, 16), name
        if name in KAGGLE_HOURS:
            # A kernel job's: CUDA, a 4-bit base, the whole train split, and the plan's
            # estimate for the budget.
            assert (config.run.backend, config.run.device) == ("auto", "auto"), name
            assert config.run.load_in_4bit and sft.subset_rows is None, name
            assert config.ops.est_gpu_hours == KAGGLE_HOURS[name], name
        else:
            # The local pilot on Apple silicon's GPU (ADR-0016), a row at a time for the
            # laptop's memory, on a subset of whole steps: its timing probe put an epoch
            # past the 2.5 h it may take (FG-491).
            assert (config.run.backend, config.run.device) == ("trl", "mps"), name
            assert not config.run.load_in_4bit
            assert sft.per_device_train_batch_size == 1
            assert sft.subset_rows is not None and sft.subset_rows % rows_a_step == 0


@pytest.mark.phase(4)
def test_main_sft_run_keeps_its_half_way_checkpoint() -> None:
    # FG-404: the main run's checkpoints at 50% and at its end, checkpoints/final, are
    # evaluated on val and the better one promoted; the pilot shows the setup works, and
    # its end alone is evaluated.
    main = load_train_config(ROOT / "configs" / "sft" / "qwen3-4b-main.yaml")
    pilot = load_train_config(ROOT / "configs" / "sft" / "qwen3-1p7b-pilot.yaml")
    assert (main.ops.keep_checkpoints_at, pilot.ops.keep_checkpoints_at) == ((0.5,), ())
    # Shares in any order, kept in order.
    shares = parse_train_config(_sft(ops={"keep_checkpoints_at": [0.75, 0.25]}))
    assert shares.ops.keep_checkpoints_at == (0.25, 0.75)


@pytest.mark.phase(4)
def test_sft_runs_wait_in_pending_within_the_story_budget() -> None:
    pending = yaml.safe_load((ROOT / "docs" / "runs" / "pending.yaml").read_text(encoding="utf-8"))
    entries = {entry["run"]: entry for entry in pending["pending"] if entry["story"] == "FG-404"}
    # Each val entry scores an SFT run's checkpoints: its final adapter, and one for each
    # share of the run it keeps.
    scores = {"sft-pilot-val": "sft-pilot", "sft-main-checkpoints-val": "sft-main"}
    assert sorted(entries) == sorted([*scores, *scores.values()])
    for evaluated, trained in scores.items():
        (sft_path,) = entries[trained]["configs"]
        config = load_train_config(ROOT / sft_path)
        assert entries[trained]["est_gpu_hours"] == config.ops.est_gpu_hours, trained
        (eval_path,) = entries[evaluated]["configs"]
        _, ops = read_config_sections(ROOT / eval_path)
        assert ops.est_gpu_hours is not None
        # The base model's FG-307 val config, which the runbook adds the adapter to.
        assert EvalConfig(**read_config_file(ROOT / eval_path)).model == config.run.model_ref
        checkpoints = 1 + len(config.ops.keep_checkpoints_at)
        assert entries[evaluated]["est_gpu_hours"] == pytest.approx(
            ops.est_gpu_hours * checkpoints
        ), evaluated
    # The story's own budget, 4.5 h in its block in the plan, which the configs' estimates
    # fill: 1 h for the pilot and 0.3 h for its val run, 2.5 h for the main run and 0.7 h
    # for its checkpoints. Plan section 3.8's rows give FG-404 4.4 h: the pilot, the main
    # run and 0.90 h for the main run's 2 checkpoint evals, with no row for the pilot's.
    assert sum(entry["est_gpu_hours"] for entry in entries.values()) == pytest.approx(4.5)


ADR_0010 = ROOT / "docs" / "adr" / "0010-lora-sft-starting-point.md"
# FG-491's committed SFT pilot, trained from configs/sft/pilot-qwen25coder05b-mps.yaml.
PILOT_SFT_MANIFEST = (
    ROOT / "docs" / "runs" / "2026-11-16-qwen2-5-coder-0-5b-instruct-sft-db0d8d8" / "manifest.json"
)
# The starting values a config states, or leaves to ADR-0010 by omission.
STARTING_SFT = {
    "learning_rate",
    "num_train_epochs",
    "lr_scheduler_type",
    "warmup_ratio",
    "max_length",
}


# What ADR-0010 names rather than states as a number: code spans, story and ADR ids, dates
# and phases.
_ADR_NAMES = re.compile(r"`[^`\n]*`|\b(?:FG|ADR)-\d+\b|\b\d{4}-\d{2}-\d{2}\b|\bPhase \d+\b")
# A number as prose writes one, such as 16, 0.0 or 2,048, and not the digits in a word such
# as float32, bf16 or T4.
_ADR_NUMBER = re.compile(r"(?<![\w.,])\d(?:[\d,.]*\d)?(?!\w)")
# The MPS row's value: the dtype from macOS 14 on, then the one before.
_MPS_CELL = re.compile(r"(\w+) where .* \(macOS 14 and later\), (\w+) otherwise")


def _quotes(text: str) -> list[str]:
    """The ``key: value`` settings ``text`` quotes in code spans, as a YAML or JSON line."""
    return [quote for quote in re.findall(r"`([^`\n]+)`", text) if ": " in quote]


def _lines(text: str) -> set[str]:
    """The settings of a YAML or JSON file, a line each, as a quote would give them."""
    return {line.split("#", 1)[0].strip().rstrip(",") for line in text.splitlines()}


def _stated(text: str) -> set[float]:
    """The numbers ``text`` states outside its names."""
    found = _ADR_NUMBER.findall(_ADR_NAMES.sub(" ", text))
    return {float(number.replace(",", "")) for number in found}


def _values(data: object) -> Iterator[float]:
    """Each number a parsed YAML or JSON document holds as a value, at any depth: not the
    digits of a string such as a digest or a time, and no flag."""
    if isinstance(data, dict):
        for value in data.values():
            yield from _values(value)
    elif isinstance(data, list):
        for value in data:
            yield from _values(value)
    elif isinstance(data, int | float) and not isinstance(data, bool):
        yield float(data)


def check_adr_0010(adr: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert that ADR-0010's text ``adr`` says what the SFT configs, ``precision_for`` and
    the committed SFT pilot did."""
    assert "- Status: accepted" in adr
    rows = {
        cells[1].strip(): cells[2].strip()
        for cells in (line.split("|") for line in adr.splitlines() if line.startswith("| "))
    }
    # The base's dtype on each device is the one precision_for trains in: on MPS, each
    # branch of the row is the dtype on that side of macOS 14.
    assert rows["Base on the CPU"] == precision_for("cpu").dtype
    branches = _MPS_CELL.fullmatch(rows["Base on MPS"])
    assert branches is not None, rows["Base on MPS"]
    trains: dict[bool, str] = {}
    for macos_14 in (True, False):
        with monkeypatch.context() as patch:
            patch.setattr(torch.backends.mps, "is_macos_or_newer", lambda *_, newer=macos_14: newer)
            trains[macos_14] = precision_for("mps").dtype
    assert branches.groups() == (trains[True], trains[False]), "the MPS row's branches"
    # The pilot trained on an Apple M4, which runs macOS 15 or later.
    pilot = PILOT_SFT_MANIFEST.read_text(encoding="utf-8")
    manifest = json.loads(pilot)
    assert manifest["config"]["trainer"]["dtype"] == branches.group(1)

    # A config that leaves a starting value out is named as taking it by omission, with the
    # settings it does state quoted from its file.
    decision = adr.split("## Decision", 1)[1].split("\n|", 1)[0]
    for name in SFT_MODELS:
        path = ROOT / "configs" / "sft" / name
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        omits = "lora" not in raw or not STARTING_SFT.issubset(raw["sft"])
        assert (f"`configs/sft/{name}`" in decision) == omits, name
    smoke = _lines((ROOT / "configs" / "sft" / "tiny-smoke.yaml").read_text(encoding="utf-8"))
    assert _quotes(decision) and set(_quotes(decision)) <= smoke

    # The note on the pilot quotes its config and manifest, and states no number that
    # neither the table above nor a value of either file holds.
    note = adr.split("## Notes", 1)[1]
    config = (ROOT / "configs" / "sft" / LOCAL_PILOT).read_text(encoding="utf-8")
    misquoted = sorted(set(_quotes(note)) - _lines(config) - _lines(pilot))
    assert _quotes(note) and not misquoted, f"the note quotes {misquoted}, in neither file"
    table = "\n".join(line for line in adr.splitlines() if line.startswith("| "))
    held = _stated(table) | set(_values(yaml.safe_load(config))) | set(_values(manifest))
    unheld = sorted(_stated(note) - held)
    assert not unheld, f"the note states {unheld}, which the table, config and manifest lack"


@pytest.mark.phase(4)
def test_adr_0010_says_what_the_configs_and_the_pilot_did(monkeypatch: pytest.MonkeyPatch) -> None:
    check_adr_0010(ADR_0010.read_text(encoding="utf-8"), monkeypatch)


@pytest.mark.phase(4)
def test_adr_0010_check_fails_on_swapped_branches_and_wrong_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adr = ADR_0010.read_text(encoding="utf-8")
    newer, older = "bfloat16 where", "float32 otherwise"
    # A count only text supplies: 20 is in the config's comments and the manifest's config
    # hash, 41 in its end time.
    wrong = {
        "the MPS row's branches": adr.replace(newer, "float32 where").replace(
            older, "bfloat16 otherwise"
        ),
        r"the note states \[20\.0\]": adr.replace("the same 16", "the same 20"),
        r"the note states \[41\.0\]": adr.replace("the same 16", "the same 41"),
        "the note quotes": adr.replace('`"over_max_length": 53`', '`"over_max_length": 52`'),
    }
    for message, text in wrong.items():
        assert text != adr, message
        with pytest.raises(AssertionError, match=message):
            check_adr_0010(text, monkeypatch)
