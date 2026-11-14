"""``forge data stats``, ``forge data splits`` and the CLI's view of the fetch sources."""

from __future__ import annotations

from pathlib import Path
from typing import get_args, get_type_hints

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.core.types import Example
from forge.data.fetch import SOURCES, SourceName
from forge.data.spider import resolve_data_dir
from forge.data.splits import MANIFEST_DIR, MANIFEST_FILE, SPLITS
from forge.data.stats import render_stats


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
