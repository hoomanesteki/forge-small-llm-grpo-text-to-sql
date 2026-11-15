from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.ops.manifest import (
    DirtyTreeError,
    GitError,
    Platform,
    build_manifest,
    config_hash,
    git_sha,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture(autouse=True)
def _fixture_data(spider_mini: Path) -> None:
    """build_manifest hashes the data, so give it the small committed subset."""


def _forge_root(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``root`` the Forge root, so the append-only logs are under its docs folder."""
    monkeypatch.setenv("FORGE_ROOT", str(root))
    monkeypatch.delenv("FORGE_LEDGER", raising=False)
    monkeypatch.delenv("FORGE_DEV_USAGE", raising=False)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-commit checkout that is the Forge root."""
    _forge_root(tmp_path, monkeypatch)
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


class _NoOffset(tzinfo):
    """A tzinfo that knows no offset, which makes its datetimes naive."""

    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str | None:
        return None


@pytest.mark.phase(1)
def test_start_time_without_an_offset_rejected(repo: Path) -> None:
    now = datetime(2026, 11, 3, 23, 30, tzinfo=_NoOffset())
    assert now.tzinfo is not None
    with pytest.raises(ValueError, match="timezone-aware"):
        build_manifest("eval", {}, model_ref="m", seed=0, repo=repo, now=now)


@pytest.mark.phase(1)
def test_manifest_keeps_its_own_copy_of_the_config(repo: Path) -> None:
    lora: dict[str, Any] = {"r": 16, "target_modules": ["q_proj"]}
    config: dict[str, Any] = {"lora": lora, "data": Path("data/spider")}
    m = build_manifest("sft", config, model_ref="m", seed=0, repo=repo)

    lora["r"] = 32
    lora["target_modules"].append("v_proj")

    # What was hashed, as JSON: the caller's edits don't reach it.
    assert m.config == {"lora": {"r": 16, "target_modules": ["q_proj"]}, "data": "data/spider"}
    assert config_hash(m.config) == m.config_hash
    assert m.config_hash != config_hash(config)


@pytest.mark.phase(1)
def test_manifest_hash_matches_a_config_with_int_keys(repo: Path) -> None:
    config: dict[str, Any] = {"lr_schedule": {2: 1e-4, 10: 1e-5}}
    m = build_manifest("sft", config, model_ref="m", seed=0, repo=repo)

    # The copy has the keys as JSON writes them, and still hashes like the config.
    assert m.config == {"lr_schedule": {"2": 1e-4, "10": 1e-5}}
    assert m.config_hash == config_hash(config)
    assert config_hash(m.config) == m.config_hash


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


def _commit_layout(repo: Path) -> str:
    """Give ``repo`` a src module and both append-only logs, committed; return HEAD."""
    for rel in ("src/x.py", "docs/compute-ledger.md", "docs/dev-usage.md"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("one\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "layout")
    return git_sha(repo)


@pytest.mark.phase(2)
def test_dirty_src_is_seen_from_a_subdirectory(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _commit_layout(repo)
    (repo / "src" / "x.py").write_text("two\n")
    # The check covers the whole checkout, not the folder git happens to start in.
    monkeypatch.chdir(repo / "docs")
    with pytest.raises(DirtyTreeError):
        git_sha()
    with pytest.raises(DirtyTreeError):
        git_sha(repo / "docs")


@pytest.mark.phase(2)
def test_appended_dev_log_is_clean_from_a_subdirectory(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = _commit_layout(repo)
    with (repo / "docs" / "dev-usage.md").open("a") as log:
        log.write("| 2026-11-15 | row |\n")
    for where in (repo / "docs", repo / "src"):
        monkeypatch.chdir(where)
        assert git_sha() == head
        assert git_sha(where) == head


@pytest.mark.phase(2)
def test_appended_logs_are_clean_with_the_forge_root_below_the_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Forge root one level below the top of the checkout that holds it, as in a bigger
    # repository. The top has a docs/dev-usage.md of its own, which is no Forge log.
    top, forge = tmp_path / "mono", tmp_path / "mono" / "forge"
    for rel in ("docs/dev-usage.md", "forge/docs/PLAN.md"):
        (top / rel).parent.mkdir(parents=True, exist_ok=True)
        (top / rel).write_text("one\n")
    _git(top, "init", "-q")
    _git(top, "config", "user.email", "test@example.com")
    _git(top, "config", "user.name", "Test")
    _git(top, "add", ".")
    _forge_root(forge, monkeypatch)
    head = _commit_layout(forge)
    for log in ("compute-ledger.md", "dev-usage.md"):
        with (forge / "docs" / log).open("a") as appended:
            appended.write("| 2026-11-15 | row |\n")
    # The run appended only to the Forge logs, so the tree is clean, from wherever git
    # starts, and with the root found from the working directory as without FORGE_ROOT.
    assert git_sha(forge) == head
    monkeypatch.delenv("FORGE_ROOT")
    monkeypatch.chdir(forge / "src")
    assert git_sha() == head
    assert git_sha(top) == head
    # The rest of the checkout still counts, the top's own docs/dev-usage.md too.
    (top / "docs" / "dev-usage.md").write_text("two\n")
    with pytest.raises(DirtyTreeError):
        git_sha(forge)


@pytest.mark.phase(2)
def test_outside_a_checkout_is_a_git_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    # Keeps git from finding a checkout that happens to hold the temp folder.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    with pytest.raises(GitError, match=f"{re.escape(str(plain))} is not in a git checkout") as err:
        git_sha(plain, allow_dirty=True)
    assert "FORGE_ROOT" in str(err.value)
    # The checkout is required, so no build can name a commit it doesn't have.
    with pytest.raises(GitError):
        build_manifest("eval", {}, model_ref="m", seed=0, repo=plain, allow_dirty=True)
