from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forge.ops.manifest import DirtyTreeError, build_manifest, config_hash, git_sha


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


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
        build_manifest("eval", {"x": 1}, run_id="r", model_ref="m", seed=0, repo=repo)
    manifest = build_manifest(
        "eval", {"x": 1}, run_id="r", model_ref="m", seed=0, repo=repo, allow_dirty=True
    )
    assert manifest["git_sha"] == f"{clean}-dirty"


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
def test_manifest_records_hashes(repo: Path) -> None:
    m = build_manifest(
        "sft", {"lr": 2e-4}, run_id="r1", model_ref="Qwen/Qwen3-1.7B", seed=42, repo=repo
    )
    assert m["config_hash"] == config_hash({"lr": 2e-4})
    assert m["data_manifest_hash"] == "unset"
    assert m["status"] == "running"
