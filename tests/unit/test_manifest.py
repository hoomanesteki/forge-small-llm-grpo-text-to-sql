from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.ops.manifest import DirtyTreeError, Platform, build_manifest, config_hash, git_sha


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture(autouse=True)
def _fixture_data(spider_mini: Path) -> None:
    """build_manifest hashes the data, so give it the small committed subset."""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


@pytest.mark.phase(0)
def test_dirty_tree_blocked(repo: Path) -> None:
    clean = git_sha(repo)
    assert len(clean) == 40
    (repo / "a.txt").write_text("two\n")
    with pytest.raises(DirtyTreeError):
        build_manifest("eval", {"x": 1}, model_ref="m", seed=0, repo=repo)
    manifest = build_manifest("eval", {"x": 1}, model_ref="m", seed=0, repo=repo, allow_dirty=True)
    assert manifest.git_sha == f"{clean}-dirty"
    assert manifest.run_id.endswith(f"-m-eval-{clean[:7]}")


@pytest.mark.phase(0)
def test_untracked_file_counts_as_dirty(repo: Path) -> None:
    (repo / "new_module.py").write_text("x = 1\n")
    with pytest.raises(DirtyTreeError):
        git_sha(repo)


@pytest.mark.phase(0)
def test_config_hash_stable() -> None:
    a = {"lr": 5e-6, "lora": {"r": 16, "alpha": 32}, "steps": 600}
    b = {"steps": 600, "lora": {"alpha": 32, "r": 16}, "lr": 5e-6}
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 64
    assert config_hash(a) != config_hash({**a, "steps": 601})


@pytest.mark.phase(0)
def test_manifest_records_hashes(repo: Path, spider_mini: Path) -> None:
    m = build_manifest(
        "sft", {"lr": 2e-4}, run_id="r1", model_ref="Qwen/Qwen3-1.7B", seed=42, repo=repo
    )
    assert isinstance(m, RunManifest)
    assert m.run_id == "r1"
    assert m.config_hash == config_hash({"lr": 2e-4})
    assert m.data_manifest_hash == data_manifest_hash(spider_mini)
    assert re.fullmatch(r"[0-9a-f]{64}", m.data_manifest_hash)
    assert m.status == "running"


@pytest.mark.phase(0)
def test_append_only_logs_do_not_make_tree_dirty(repo: Path) -> None:
    docs = repo / "docs"
    docs.mkdir()
    (docs / "compute-ledger.md").write_text("| header |\n")
    (docs / "dev-usage.md").write_text("| header |\n")
    _git(repo, "add", "docs")
    _git(repo, "commit", "-q", "-m", "logs")
    (docs / "compute-ledger.md").write_text("| header |\n| 2026-11-14 | row |\n")
    (docs / "dev-usage.md").write_text("| header |\n| 2026-11-14 | row |\n")
    assert not git_sha(repo).endswith("-dirty")


@pytest.mark.phase(0)
def test_untracked_counts_even_if_status_hides_them(repo: Path) -> None:
    _git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "new_module.py").write_text("x = 1\n")
    with pytest.raises(DirtyTreeError):
        git_sha(repo)


@pytest.mark.phase(1)
def test_run_id_generated_from_the_utc_start_date(repo: Path) -> None:
    # 01:00 at UTC+5 is still the previous day in UTC, which is what the id must use.
    started = datetime(2026, 11, 4, 1, 0, tzinfo=timezone(timedelta(hours=5)))
    m = build_manifest(
        "sft", {"lr": 2e-4}, model_ref="Qwen/Qwen3-1.7B", seed=42, repo=repo, now=started
    )
    assert m.run_id == f"2026-11-03-qwen3-1-7b-sft-{git_sha(repo)[:7]}"
    assert m.started_at == started
    assert m.started_at.astimezone(UTC).date().isoformat() == m.run_id[:10]


@pytest.mark.phase(1)
def test_naive_start_time_rejected(repo: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_manifest("eval", {}, model_ref="m", seed=0, repo=repo, now=datetime(2026, 11, 3))


@pytest.mark.phase(1)
def test_manifest_round_trips_through_json(repo: Path) -> None:
    m = build_manifest(
        "grpo",
        {"grpo": {"num_generations": 6}, "seed": 42},
        model_ref="Qwen/Qwen3-4B",
        seed=42,
        platform="kaggle",
        gpu="T4",
        parent_run="2026-11-01-qwen3-4b-sft-a1b2c3d",
        repo=repo,
    )
    assert RunManifest.model_validate_json(m.model_dump_json()) == m


@pytest.mark.phase(1)
def test_unknown_platform_rejected_at_runtime(repo: Path) -> None:
    # The CLI hands over plain strings, so the model has to catch what typing can't.
    with pytest.raises(ValidationError, match="platform"):
        build_manifest("eval", {}, model_ref="m", seed=0, repo=repo, platform="aws")


@pytest.mark.phase(1)
def test_platforms_match_the_manifest_contract() -> None:
    assert get_args(RunManifest.model_fields["platform"].annotation) == get_args(Platform)
