"""``forge data stats``, ``splits``, ``prompt-stats`` and ``cache-gold``, and the CLI's
spelled-out choices."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args, get_type_hints

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import Example
from forge.data import prompts
from forge.data.fetch import SOURCES, SourceName
from forge.data.manifest import data_manifest_hash
from forge.data.prompts import build_prompt, chat_token_counter
from forge.data.schema import PROMPT_VERSIONS
from forge.data.spider import load_raw, resolve_data_dir
from forge.data.splits import MANIFEST_DIR, MANIFEST_FILE, SPLITS, split_examples
from forge.data.stats import render_stats
from forge.eval import gold_cache
from forge.eval.gold_cache import GOLD_SPLITS
from tests.fakes import WordTokenizer


def _example(i: int, db_id: str, split: str, hardness: str) -> Example:
    return Example(
        id=f"x-{i}", db_id=db_id, question="q", gold_sql="SELECT 1", hardness=hardness, split=split
    )


def _cells(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines()]


@pytest.mark.phase(1)
def test_render_stats_counts_splits_databases_and_hardness() -> None:
    examples = [
        _example(0, "zoo", "dev", "extra"),
        _example(1, "bank", "train", "easy"),
        _example(2, "bank", "train", "hard"),
        _example(3, "atlas", "train", "easy"),
    ]

    assert _cells(render_stats(examples)) == [
        ["split", "questions", "databases", "easy", "medium", "hard", "extra"],
        ["train", "3", "2", "2", "0", "1", "0"],
        ["dev", "1", "1", "0", "0", "0", "1"],
        ["total", "4", "3", "2", "0", "1", "1"],
        [],
        ["database", "split", "questions", "easy", "medium", "hard", "extra"],
        ["atlas", "train", "1", "1", "0", "0", "0"],
        ["bank", "train", "2", "1", "0", "1", "0"],
        ["zoo", "dev", "1", "0", "0", "0", "1"],
    ]


@pytest.mark.phase(1)
def test_stats_cli_on_the_fixture(spider_mini: Path) -> None:
    result = CliRunner().invoke(cli.app, ["data", "stats"])

    assert result.exit_code == 0, result.output
    rows = _cells(result.stdout)
    assert rows[0] == ["data:", str(spider_mini)]
    assert ["train", "40", "4", "12", "12", "8", "8"] in rows
    assert ["val", "20", "1", "5", "5", "5", "5"] in rows
    assert ["total", "60", "5", "17", "17", "13", "13"] in rows
    assert ["flight_1", "val", "20", "5", "5", "5", "5"] in rows
    assert ["geo", "train", "10", "3", "3", "2", "2"] in rows
    assert rows[-1][:3] == ["data", "manifest", "hash:"]
    assert len(rows[-1][3]) == 64


@pytest.mark.phase(1)
def test_stats_cli_without_data_says_how_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_DATA_DIR", str(tmp_path / "empty"))
    result = CliRunner().invoke(cli.app, ["data", "stats"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "train_spider.json not found" in result.stderr
    assert "forge data fetch" in result.stderr


def _committed() -> dict[str, bytes]:
    names = [MANIFEST_FILE, *(f"{name}.ids.txt" for name in SPLITS)]
    return {name: (MANIFEST_DIR / name).read_bytes() for name in names}


@pytest.mark.phase(1)
def test_splits_cli_on_the_fixture(spider_mini: Path) -> None:
    result = CliRunner().invoke(cli.app, ["data", "splits"])

    # The fixture's split is not the full download's, and the command says so.
    assert result.exit_code == 1
    rows = _cells(result.stdout)
    assert rows[2:6] == [
        ["split", "questions", "databases"],
        ["train", "40", "4"],
        ["val", "20", "1"],
        ["dev", "0", "0"],
    ]
    assert "val databases (seed forge-val-v1): flight_1" in result.stdout
    assert "differ from this split" in result.stderr
    assert "point FORGE_DATA_DIR at it" in result.stderr


@pytest.mark.phase(1)
def test_splits_cli_write_refuses_a_subset(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Any write would land under this root, never in the repo.
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    before = _committed()
    result = CliRunner().invoke(cli.app, ["data", "splits", "--write"])

    assert result.exit_code == 1
    assert "--write needs the full Spider download" in result.stderr
    assert not (tmp_path / "src").exists()
    assert _committed() == before


@pytest.mark.phase(1)
@pytest.mark.requires_full_data
def test_splits_cli_rewrites_the_committed_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_DATA_DIR", str(resolve_data_dir()))
    check = CliRunner().invoke(cli.app, ["data", "splits"])
    assert check.exit_code == 0, check.output
    assert ["val", "872", "15"] in _cells(check.stdout)
    assert check.stdout.endswith("matches the committed split manifests\n")

    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    result = CliRunner().invoke(cli.app, ["data", "splits", "--write"])
    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "src" / "forge" / "data" / "splits"
    assert {path.name: path.read_bytes() for path in out_dir.iterdir()} == _committed()


@pytest.mark.phase(1)
def test_cli_offers_exactly_the_fetch_sources() -> None:
    hints = get_type_hints(cli.data_fetch, include_extras=False)
    assert get_args(hints["source"]) == get_args(SourceName) == tuple(SOURCES)


@pytest.mark.phase(1)
def test_cli_offers_exactly_the_splits_and_prompt_versions() -> None:
    hints = get_type_hints(cli.data_prompt_stats, include_extras=False)
    assert get_args(hints["split"]) == SPLITS
    assert get_args(hints["prompt_version"]) == PROMPT_VERSIONS


@pytest.mark.phase(1)
def test_prompt_stats_cli_on_the_fixture(
    spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded: list[str] = []

    def fake_load(name: str) -> WordTokenizer:
        loaded.append(name)
        return WordTokenizer()

    monkeypatch.setattr(prompts, "load_tokenizer", fake_load)
    args = ["data", "prompt-stats", "--tokenizer", "org/model", "--split", "val"]
    result = CliRunner().invoke(cli.app, [*args, "--prompt-version", "p2"])

    assert result.exit_code == 0, result.output
    assert loaded == ["org/model"]
    count = chat_token_counter(WordTokenizer())
    val = split_examples(load_raw())["val"]
    lengths = sorted(count(build_prompt(example, "p2")) for example in val)
    assert len(lengths) == 20
    rows = _cells(result.stdout)
    assert rows[:3] == [
        ["data:", str(spider_mini)],
        ["split", "val,", "prompt", "p2,", "tokenizer", "org/model"],
        [],
    ]
    # Nearest rank over 20 prompts: the 10th and the 19th shortest.
    assert rows[3:] == [
        ["prompts", "20"],
        ["tokens", "p50", str(lengths[9])],
        ["tokens", "p95", str(lengths[18])],
        ["tokens", "max", str(lengths[-1])],
        ["over", "1536", "tokens", "0"],
        ["sample", "rows", "dropped", "0"],
        ["columns", "dropped", "0"],
        ["excluded", "from", "training", "0"],
    ]


@pytest.mark.phase(1)
def test_prompt_stats_cli_on_dev_is_not_a_dev_use(
    with_dev: Path, dev_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_DATA_DIR", str(with_dev))
    monkeypatch.setattr(prompts, "load_tokenizer", lambda name: WordTokenizer())
    args = ["data", "prompt-stats", "--tokenizer", "org/model", "--split", "dev"]

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[:2] == [f"data: {with_dev}", cli.NOT_A_DEV_USE]
    assert ["prompts", "10"] in _cells(result.stdout)
    assert not dev_log.exists()

    # Other splits need no such note.
    val = CliRunner().invoke(cli.app, [*args[:-1], "val"])
    assert val.exit_code == 0, val.output
    assert cli.NOT_A_DEV_USE not in val.stdout


@pytest.mark.phase(1)
def test_prompt_stats_cli_errors(spider_mini: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(name: str) -> WordTokenizer:
        raise OSError(f"We couldn't connect to 'https://huggingface.co' to load {name}")

    monkeypatch.setattr(prompts, "load_tokenizer", offline)
    result = CliRunner().invoke(cli.app, ["data", "prompt-stats", "--tokenizer", "org/model"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "error: We couldn't connect" in result.stderr

    # The fixture has no dev questions; that is reported before any tokenizer loads.
    args = ["data", "prompt-stats", "--tokenizer", "org/model", "--split", "dev"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 1
    assert "the dev split has no questions" in result.stderr

    missing = CliRunner().invoke(cli.app, ["data", "prompt-stats"])
    assert missing.exit_code != 0
    assert "--tokenizer" in missing.output


@pytest.mark.phase(5)
def test_cache_gold_cli_builds_the_file_once(
    spider_mini: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_CACHE_DIR", str(tmp_path))
    path = tmp_path / "gold" / f"train-{data_manifest_hash(spider_mini)[:12]}.jsonl"
    built = CliRunner().invoke(cli.app, ["data", "cache-gold", "--split", "train"])
    assert built.exit_code == 0, built.output
    done = rf"train gold: 40 results in {re.escape(str(path))}, built in \d+\.\d s\n"
    assert re.fullmatch(done, built.stdout)
    assert path.is_file()

    def no_second_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("the gold queries ran again")

    monkeypatch.setattr(gold_cache, "run_gold", no_second_build)
    again = CliRunner().invoke(cli.app, ["data", "cache-gold"])
    assert again.exit_code == 0, again.output
    assert again.stdout == f"train gold: 40 results in {path}, already built for this data\n"

    # Dev's file is built by a dev evaluation alone.
    dev = CliRunner().invoke(cli.app, ["data", "cache-gold", "--split", "dev"])
    assert dev.exit_code == 2
    hints = get_type_hints(cli.data_cache_gold, include_extras=False)
    assert get_args(hints["split"]) == GOLD_SPLITS

    monkeypatch.setenv("FORGE_DATA_DIR", str(tmp_path / "nowhere"))
    missing = CliRunner().invoke(cli.app, ["data", "cache-gold", "--split", "val"])
    assert missing.exit_code == 1
    assert missing.stderr.startswith("error: ")
