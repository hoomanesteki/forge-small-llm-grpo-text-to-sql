"""The GRPO dataset (FG-505): the filter that picks GRPO's rows from a greedy eval of
train questions, ``forge data build-grpo`` that keeps them, the training run that reads
them back, and the curriculum's order as TRL's own dataloader takes it."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from accelerate import skip_first_batches
from datasets import Dataset
from transformers import Qwen2Config, Qwen2ForCausalLM
from typer.testing import CliRunner

from forge import cli
from forge.core.types import ExecResult, Hardness, Split
from forge.data.spider import ROWLESS_DBS
from forge.data.splits import load_train
from forge.eval.config import EvalConfig
from forge.eval.gold_cache import split_gold
from forge.eval.run_files import StoredRun, read_run
from forge.eval.runner import EvalRun, run_eval
from forge.infer.base import FakeBackend
from forge.train.backends import TrlBackend, grpo_settings
from forge.train.config import (
    CurriculumConfig,
    GrpoConfig,
    load_train_config,
    parse_train_config,
)
from forge.train.curriculum import (
    KEEP_SOLVED,
    LEVELS,
    MAX_GOLD_MS,
    ROWS_FILE,
    SUMMARY_FILE,
    check_filter_run,
    curriculum_order,
    filter_hash,
    read_filter,
    save_filter,
    select_rows,
    warmup_steps,
)
from forge.train.grpo import grpo_rows
from tests.fakes import WordTokenizer
from tests.tiny_hf import byte_tokenizer

MODEL = "org/tiny-sft@0123456789abcdef0123456789abcdef01234567"
EVAL = "2026-11-20-tiny-sft-eval-abc1234"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_log: Path) -> Path:
    """Runs and the gold cache under tmp_path, and a scratch dev log."""
    monkeypatch.setenv("FORGE_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "runs"


def _right() -> set[str]:
    """The train questions the stand-in model answers with their gold query: two in
    three, in the loader's order; it has no answer for the rest."""
    return {example.id for i, example in enumerate(load_train()) if i % 3 != 2}


def _evaluate(run_id: str, right: set[str], *, split: Split = "train", **options: Any) -> EvalRun:
    """A finished eval run of the stand-in model, as ``forge eval run`` stores one."""
    gold = {example.id: example.gold_sql for example in load_train()}

    def reply(prompt: Any, params: Any, sample_idx: int) -> str:
        if prompt.example_id in right:
            return f"```sql\n{gold[prompt.example_id]}\n```"
        return "I can't tell from this schema."

    config = EvalConfig(model=MODEL, split=split, run_id=run_id, **options)
    return run_eval(
        config, backend=FakeBackend(reply, model_ref=MODEL), allow_dirty=True, workers=1
    )


def _timed(gold: dict[str, ExecResult], ms: dict[str, float]) -> dict[str, ExecResult]:
    return {**gold, **{i: gold[i].model_copy(update={"ms": t}) for i, t in ms.items()}}


@pytest.mark.phase(5)
def test_filter_proportions(spider_mini: Path, runs: Path, tmp_path: Path) -> None:
    # The fixture's 40 train questions, the stand-in right on two in three of them.
    train = load_train()
    right = _right()
    _evaluate(EVAL, right)
    stored = read_run(runs / EVAL)
    assert len(stored.scores) == len(train) == 40
    # One right and one wrong question whose gold took 1.5 s, and one that took 1 s
    # exactly, which isn't slower than the limit.
    ordered = [example.id for example in train]
    slow = {ordered[0]: 1500.0, ordered[2]: 1500.0}
    gold = _timed(split_gold("train", cache_root=tmp_path / "cache").results, slow)
    gold = _timed(gold, {ordered[1]: MAX_GOLD_MS})
    assert ordered[0] in right and ordered[2] not in right

    found = select_rows(stored, train, gold, seed=0)

    rowless = {e.id for e in train if e.db_id in ROWLESS_DBS}
    usable = [e for e in train if e.id not in rowless and e.id not in slow]
    wrong = {e.id for e in usable if e.id not in right}
    solved = [e.id for e in usable if e.id in right]
    kept = {row.example_id for row in found.rows}
    kept_solved = kept & set(solved)
    # Every wrong question stays, 30% of the 19 right ones, 5.7 rounded to 6, and none on
    # a rowless database or with slow gold.
    assert (len(rowless), len(wrong), len(solved)) == (10, 9, 19)
    assert wrong <= kept and kept <= wrong | set(solved)
    assert len(kept_solved) == 6 == round(KEEP_SOLVED * len(solved))
    assert not kept & (rowless | set(slow))
    assert ordered[1] in solved  # 1 s exactly is not slower than the limit
    assert [row.example_id for row in found.rows] == [i for i in ordered if i in kept]
    for row in found.rows:
        assert row.solved == (row.example_id in right)
        assert row.gold_ms == gold[row.example_id].ms

    # The summary's tallies, by hardness and for all, are those same counts.
    summary = found.summary
    total = summary.total.model_dump()
    assert total == {
        "scored": 40,
        "rowless": 10,
        "slow_gold": 2,
        "wrong": 9,
        "solved": 19,
        "solved_kept": 6,
        "kept": 15,
    }
    by_level: dict[Hardness, Counter[str]] = {level: Counter() for level in LEVELS}
    for example in train:
        tally = by_level[example.hardness]
        tally["scored"] += 1
        tally["rowless"] += example.id in rowless
        tally["slow_gold"] += example.id in slow
        tally["wrong"] += example.id in wrong
        tally["solved"] += example.id in solved
        tally["solved_kept"] += example.id in kept_solved
        tally["kept"] += example.id in kept
    for level in LEVELS:
        counted = summary.by_hardness[level].model_dump()
        assert counted == {name: by_level[level][name] for name in counted}, level
    assert sorted(summary.slow_gold_ids) == sorted(slow)
    assert (summary.eval_run, summary.model_ref, summary.adapter_ref) == (EVAL, MODEL, None)
    assert (summary.keep_solved, summary.max_gold_ms, summary.seed) == (0.3, 1000.0, 0)
    assert summary.data_manifest_hash == stored.manifest.data_manifest_hash
    assert summary.hash == filter_hash(summary, found.rows)

    # The same seed keeps the same questions; another keeps as many, but others.
    assert select_rows(stored, train, gold, seed=0) == found
    other = select_rows(stored, train, gold, seed=1)
    other_solved = {row.example_id for row in other.rows if row.solved}
    assert len(other_solved) == 6 and other_solved != kept_solved
    assert {row.example_id for row in other.rows if not row.solved} == wrong
    assert other.summary.hash != summary.hash


def _stored_with(stored: StoredRun, **config: Any) -> StoredRun:
    changed = {**stored.manifest.config, **config}
    return dataclasses.replace(
        stored, manifest=stored.manifest.model_copy(update={"config": changed})
    )


@pytest.mark.phase(5)
def test_filter_reads_greedy_train_evals_only(spider_mini: Path, runs: Path) -> None:
    _evaluate(EVAL, _right(), limit=6)
    stored = read_run(runs / EVAL)
    data_hash = stored.manifest.data_manifest_hash
    check_filter_run(stored, data_hash)
    generation = stored.manifest.config["generation"]
    refused = [
        (_stored_with(stored, split="val"), "evaluated val, and the GRPO filter reads an eval"),
        (
            _stored_with(stored, generation={**generation, "n": 4, "temperature": 0.8}),
            "drew 4 replies a question at temperature 0.8",
        ),
        (_stored_with(stored, retry=1), "scores are of second replies"),
    ]
    for run, message in refused:
        with pytest.raises(ValueError, match=message):
            check_filter_run(run, data_hash)
    with pytest.raises(ValueError, match="its example ids may name other questions"):
        check_filter_run(stored, "f" * 64)

    # A question train doesn't hold is refused, before anything is kept.
    gold = split_gold("train").results
    with pytest.raises(ValueError, match="scored 6 questions train doesn't hold here"):
        select_rows(stored, [], gold)
    with pytest.raises(ValueError, match="has no result in the train gold cache"):
        select_rows(stored, load_train(), {})


@pytest.mark.phase(5)
def test_build_grpo_keeps_the_rows_and_their_summary(
    spider_mini: Path, runs: Path, tmp_path: Path
) -> None:
    _evaluate(EVAL, _right())
    out = tmp_path / "grpo"
    cli_runner = CliRunner()
    build = ["data", "build-grpo", "--out", str(out), "--filter-from", EVAL]

    built = cli_runner.invoke(cli.app, build)
    assert built.exit_code == 0, built.output
    folder = out / EVAL
    summary = json.loads((folder / SUMMARY_FILE).read_text(encoding="utf-8"))
    lines = (folder / ROWS_FILE).read_text(encoding="utf-8").splitlines()
    # The fixture's gold queries all run in milliseconds, so only the rowless 10 go, and
    # 30% of the 20 right ones on databases with rows stay beside the 10 wrong ones.
    assert summary["total"] == {
        "scored": 40,
        "rowless": 10,
        "slow_gold": 0,
        "wrong": 10,
        "solved": 20,
        "solved_kept": 6,
        "kept": 16,
    }
    assert sorted(summary["by_hardness"]) == sorted(LEVELS) and len(lines) == 16
    assert f"eval run {EVAL}: {MODEL}" in built.output
    assert "kept the wrong replies and 30% of the right ones (seed 0)" in built.output
    assert "gold over 1000 ms" in built.output
    assert f"grpo rows: {folder}\nhash: {summary['hash']}" in built.output
    table = built.output.split("hash: ", 1)[0]
    assert "hardness  scored  rowless  slow gold  wrong  solved  solved kept  kept" in table
    assert "all           40       10          0     10      20            6    16" in table
    # The help says what the constants are.
    help_text = cli_runner.invoke(cli.app, ["data", "build-grpo", "--help"]).output
    assert f"{KEEP_SOLVED:.0%} of those it got right" in help_text
    assert f"over {MAX_GOLD_MS / 1000:g} s" in help_text

    # Read back as training reads it; building it again keeps the folder as it is.
    found = read_filter(EVAL, root=out)
    assert len(found.rows) == 16 and found.summary.hash == summary["hash"]
    again = cli_runner.invoke(cli.app, build)
    assert again.exit_code == 0 and json.loads((folder / SUMMARY_FILE).read_text()) == summary
    # Other rows under the same name are refused, and so are rows edited by hand.
    reseeded = cli_runner.invoke(cli.app, [*build, "--seed", "1"])
    assert reseeded.exit_code == 1
    assert f"{folder} holds other rows of run {EVAL}" in reseeded.stderr
    (folder / ROWS_FILE).write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="don't hash to the summary's hash"):
        read_filter(EVAL, root=out)

    # Only a finished eval of train questions: val's is refused, and so is no run at all.
    _evaluate("val-eval", set(), split="val", limit=4)
    refused = cli_runner.invoke(cli.app, [*build[:-1], "val-eval"])
    assert refused.exit_code == 1 and "evaluated val" in refused.stderr
    missing = cli_runner.invoke(cli.app, [*build[:-1], "no-such-run"])
    assert missing.exit_code == 1 and "has no manifest.json" in missing.stderr
    assert sorted(path.name for path in out.iterdir()) == [EVAL]


@pytest.mark.phase(5)
def test_grpo_rows_are_the_filters(
    spider_mini: Path, runs: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _evaluate(EVAL, _right())
    stored = read_run(runs / EVAL)
    gold = split_gold("train").results
    found = select_rows(stored, load_train(), gold)
    out = tmp_path / "grpo"
    save_filter(found, out)
    config = parse_train_config(
        {"run": {"kind": "grpo", "model_ref": MODEL}, "grpo": {"filter_from": EVAL}}
    )

    rows, meta = grpo_rows(config, WordTokenizer(), data_dir=spider_mini, grpo_root=out)

    assert list(rows["example_id"]) == [row.example_id for row in found.rows]
    assert meta["filter"] == {
        "eval_run": EVAL,
        "hash": found.summary.hash,
        "kept": 16,
        "left_out": 14,
    }
    assert meta["rows"] == 16 and meta["rowless"] == 10
    unfiltered = parse_train_config({"run": {"kind": "grpo", "model_ref": MODEL}, "grpo": {}})
    _, all_rows = grpo_rows(unfiltered, WordTokenizer(), data_dir=spider_mini)
    assert all_rows["hash"] != meta["hash"]
    # Rows another model's replies picked train this one all the same, with a warning.
    other = config.model_copy(update={"run": config.run.model_copy(update={"model_ref": "org/b"})})
    with caplog.at_level(logging.WARNING, logger="forge.train.grpo"):
        grpo_rows(other, WordTokenizer(), data_dir=spider_mini, grpo_root=out)
    assert f"the rows of {EVAL} were picked by {MODEL}'s replies, and this run trains org/b" in (
        caplog.text
    )

    # Rows never built, rows of other data, and a run id that is no folder name.
    with pytest.raises(FileNotFoundError, match=f"forge data build-grpo --filter-from {EVAL}"):
        grpo_rows(config, WordTokenizer(), data_dir=spider_mini, grpo_root=tmp_path / "none")
    elsewhere = dataclasses.replace(
        stored,
        manifest=stored.manifest.model_copy(
            update={"run_id": "elsewhere", "data_manifest_hash": "0" * 64}
        ),
    )
    save_filter(select_rows(elsewhere, load_train(), gold), out)
    with pytest.raises(ValueError, match="were picked on data 000000000000"):
        read_filter("elsewhere", root=out)
    placeholder = parse_train_config(
        {"run": {"kind": "grpo", "model_ref": MODEL}, "grpo": {"filter_from": "<eval-run-id>"}}
    )
    with pytest.raises(ValueError, match="names a folder"):
        grpo_rows(placeholder, WordTokenizer(), data_dir=spider_mini, grpo_root=out)


def _rows(hardness: list[Hardness]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": f"question {i}"}],
                "example_id": f"spider-train-{i:05d}",
                "hardness": level,
            }
            for i, level in enumerate(hardness)
        ]
    )


def _steps(batches: list[list[dict[str, Any]]], per_step: int) -> list[list[dict[str, Any]]]:
    """Each optimizer step's rows, from the dataloader's micro-batches: ``per_step`` of
    them a step, every row as many times as its group has replies."""
    return [
        [row for batch in batches[s : s + per_step] for row in batch]
        for s in range(0, len(batches), per_step)
    ]


@pytest.mark.phase(5)
def test_curriculum_boundary(tmp_path: Path) -> None:
    # 3 rows of each level; 2 prompts a step, each with its 2 replies, over 2 micro-batches.
    dataset = _rows(["easy", "medium", "hard", "extra"] * 3)
    cfg = GrpoConfig(
        num_generations=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=10,
        curriculum=CurriculumConfig(),
    )
    assert cfg.prompts_per_step == 2 and warmup_steps(0.4, 10) == 4
    tokenizer = byte_tokenizer()
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )

    def reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        return [0.0] * len(completions)

    def trainer() -> Any:
        return TrlBackend("cpu").grpo_trainer(
            model, tokenizer, dataset, [reward], cfg, [], output_dir=tmp_path, seed=7
        )

    # TRL's own dataloader, as GRPOTrainer builds it for training.
    built = trainer()
    assert built.args.shuffle_dataset is False
    batches = list(built.get_train_dataloader())
    steps = _steps(batches, cfg.gradient_accumulation_steps)
    assert len(steps) == cfg.max_steps
    by_id = dict(zip(dataset["example_id"], dataset["hardness"], strict=True))
    prompts = []
    for step in steps:
        ids = [row["example_id"] for row in step]
        # Two prompts, each with its 2 replies in both of the step's micro-batches.
        assert sorted(Counter(ids).values()) == [4, 4]
        prompts.append(sorted(set(ids), key=ids.index))
    # The first 4 of the 10 steps, 40%, draw easy and medium rows alone, every one of the
    # 6 there before any comes again; the last 6 draw from all 12, each once.
    first, rest = prompts[:4], prompts[4:]
    assert all(by_id[i] in ("easy", "medium") for step in first for i in step)
    drawn = [i for step in first for i in step]
    assert len(set(drawn[:6])) == 6 and {by_id[i] for i in drawn} == {"easy", "medium"}
    assert sorted(i for step in rest for i in step) == sorted(dataset["example_id"])

    # A session that resumes after step 3 skips its batches, as the trainer does, and goes
    # on with the same steps: step 4 is still the first to draw from every row.
    resumed = skip_first_batches(trainer().get_train_dataloader(), 3 * 2)
    assert list(resumed) == batches[6:]

    # A part of the run with fewer rows than a step's is refused, and with no curriculum
    # the rows go to TRL as they are, for it to shuffle.
    with pytest.raises(ValueError, match="the curriculum has 1 rows of easy or medium"):
        curriculum_order(
            ["a", "b", "c"],
            ["easy", "hard", "extra"],
            steps=10,
            per_step=2,
            curriculum=CurriculumConfig(),
            seed=0,
        )
    plain = cfg.model_copy(update={"curriculum": None})
    assert grpo_settings(plain)["shuffle_dataset"] is True
    shuffled = TrlBackend("cpu").grpo_trainer(
        model, tokenizer, dataset, [reward], plain, [], output_dir=tmp_path, seed=7
    )
    assert list(shuffled.train_dataset["example_id"]) == list(dataset["example_id"])


@pytest.mark.phase(5)
def test_curriculum_order_is_fixed_by_its_inputs() -> None:
    ids = [f"spider-train-{i:05d}" for i in range(9)]
    levels = ["easy", "medium", "hard"] * 3
    curriculum = CurriculumConfig(share=0.5, hardness=("easy",))
    order = curriculum_order(ids, levels, steps=6, per_step=2, curriculum=curriculum, seed=3)
    # 3 of 6 steps from the 3 easy rows, two passes of them; then two thirds of a pass of
    # all 9, no row twice.
    assert len(order) == 12 and all(levels[i] == "easy" for i in order[:6])
    assert sorted(order[:3]) == sorted(order[3:6]) == [0, 3, 6]
    assert len(set(order[6:])) == 6
    assert order == curriculum_order(
        ids, levels, steps=6, per_step=2, curriculum=curriculum, seed=3
    )
    assert order != curriculum_order(
        ids, levels, steps=6, per_step=2, curriculum=curriculum, seed=4
    )
    # The ids, not the rows' places, set where each row goes.
    flipped = curriculum_order(
        ids[::-1], levels[::-1], steps=6, per_step=2, curriculum=curriculum, seed=3
    )
    assert [ids[::-1][i] for i in flipped] == [ids[i] for i in order]
    # The share is read as written: 0.7 of 10 steps is 7, and any part of a step counts.
    assert (warmup_steps(0.7, 10), warmup_steps(0.4, 600), warmup_steps(0.4, 2)) == (7, 240, 1)


@pytest.mark.phase(5)
def test_curriculum_config_is_checked() -> None:
    def grpo(**section: Any) -> dict[str, Any]:
        return {"run": {"kind": "grpo", "model_ref": MODEL}, "grpo": section}

    config = parse_train_config(grpo(curriculum={}))
    assert config.grpo is not None and config.grpo.curriculum == CurriculumConfig()
    assert (config.grpo.curriculum.share, config.grpo.curriculum.hardness) == (
        0.4,
        ("easy", "medium"),
    )
    assert parse_train_config(config.model_dump(mode="json")) == config
    plain = parse_train_config(grpo())
    assert plain.grpo is not None and (plain.grpo.curriculum, plain.grpo.filter_from) == (
        None,
        None,
    )
    refused = [
        ({"hardness": []}, "first steps need at least one hardness level"),
        ({"hardness": ["easy", "easy"]}, "hardness names easy more than once"),
        ({"hardness": ["easy", "medium", "hard", "extra"]}, "set curriculum: null for none"),
        ({"hardness": ["trivial"]}, "Input should be 'easy', 'medium', 'hard' or 'extra'"),
        ({"share": 1.0}, "Input should be less than 1"),
        ({"share": 0.0}, "Input should be greater than 0"),
        ({"steps": 0.4}, "Extra inputs are not permitted"),
    ]
    for section, message in refused:
        with pytest.raises(ValueError, match=message):
            parse_train_config(grpo(curriculum=section))
    with pytest.raises(ValueError, match="filter_from: String should have at least 1 character"):
        parse_train_config(grpo(filter_from=""))


@pytest.mark.phase(5)
def test_grpo_configs_take_the_curriculum() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "grpo"
    for path in sorted(root.glob("*.yaml")):
        grpo = load_train_config(path).grpo
        assert grpo is not None and grpo.curriculum == CurriculumConfig(), path.name
        # The main run filters by the eval FG-507 makes and names here, and the local pilot
        # by FG-591's, committed under docs/runs; the others train on every row.
        filters = {
            "qwen3-4b-main.yaml": "<filter-eval-run-id>",
            "pilot-qwen25coder05b-mps.yaml": "2026-11-17-qwen2-5-coder-0-5b-instruct-eval-a27afde",
        }
        assert grpo.filter_from == filters.get(path.name), path.name
