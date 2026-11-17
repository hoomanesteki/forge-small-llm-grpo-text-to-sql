from __future__ import annotations

import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from forge.core.types import RunManifest
from forge.data.manifest import data_manifest_hash
from forge.ops.ledger import PLATFORMS
from forge.ops.manifest import (
    CODE_SHA_ENV,
    DirtyTreeError,
    GitError,
    NoCommitError,
    Platform,
    accelerator,
    build_manifest,
    close,
    code_sha,
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
@pytest.mark.parametrize("ignored", [False, True])
def test_a_forge_root_its_checkout_does_not_track_is_a_git_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignored: bool
) -> None:
    # A Forge root copied into an unrelated repository. Untracked, its files would make
    # the tree dirty; ignored, the other repository's commit would pass as clean.
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    _git(other, "config", "user.email", "test@example.com")
    _git(other, "config", "user.name", "Test")
    (other / ".gitignore").write_text("forge/\n" if ignored else "")
    _git(other, "add", ".gitignore")
    _git(other, "commit", "-q", "-m", "init")
    forge = other / "forge"
    (forge / "docs").mkdir(parents=True)
    (forge / "docs" / "PLAN.md").write_text("plan\n")
    _forge_root(forge, monkeypatch)
    with pytest.raises(GitError, match="doesn't track it"):
        git_sha(forge, allow_dirty=True)


def _ignores_case(folder: Path) -> bool:
    probe = folder / "Case-Probe"
    probe.write_text("")
    try:
        return (folder / "case-probe").exists()
    finally:
        probe.unlink()


@pytest.mark.phase(2)
def test_a_forge_root_in_other_letter_case_keeps_its_logs_out(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _ignores_case(repo):
        pytest.skip(f"{repo} is on a file system that tells letter case apart")
    head = _commit_layout(repo)
    # FORGE_ROOT typed in another case than the folder on disk, which such a file system
    # accepts; git's pathspecs still need the logs as git spells them.
    _forge_root(repo.with_name(repo.name.swapcase()), monkeypatch)
    with (repo / "docs" / "dev-usage.md").open("a") as log:
        log.write("| 2026-11-15 | row |\n")
    assert git_sha(repo) == head
    (repo / "src" / "x.py").write_text("two\n")
    with pytest.raises(DirtyTreeError):
        git_sha(repo)


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


# Built at run time, so no literal here looks like a real commit someone could mistake.
KERNEL_SHA = "".join("0123456789abcdef"[(i * 7) % 16] for i in range(40))


@pytest.mark.phase(3)
def test_a_checkout_names_its_own_commit_whatever_the_kernel_variable_says(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CODE_SHA_ENV, KERNEL_SHA)
    clean = git_sha(repo)
    assert code_sha(repo) == clean != KERNEL_SHA
    assert build_manifest("eval", {}, model_ref="m", seed=0, repo=repo).git_sha == clean
    # Nor does it excuse a dirty tree.
    (repo / "a.txt").write_text("two\n")
    with pytest.raises(DirtyTreeError, match="uncommitted changes"):
        code_sha(repo)
    assert code_sha(repo, allow_dirty=True) == f"{clean}-dirty"


@pytest.mark.phase(3)
def test_outside_a_checkout_the_commit_comes_from_the_kernel_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = tmp_path / "working"
    kernel.mkdir()
    monkeypatch.delenv(CODE_SHA_ENV, raising=False)
    # A kernel has no checkout: without the variable the code's commit is unknown.
    with pytest.raises(NoCommitError, match=f"not in a git checkout, and {CODE_SHA_ENV}"):
        build_manifest("eval", {}, model_ref="m", seed=0, repo=kernel)

    monkeypatch.setenv(CODE_SHA_ENV, KERNEL_SHA)
    started = datetime(2026, 11, 3, 9, tzinfo=UTC)
    manifest = build_manifest("eval", {}, model_ref="org/m", seed=0, repo=kernel, now=started)
    assert manifest.git_sha == KERNEL_SHA
    assert manifest.run_id == f"2026-11-03-m-eval-{KERNEL_SHA[:7]}"
    # The working folder of a kernel is where build_manifest looks by default.
    monkeypatch.chdir(kernel)
    assert code_sha() == KERNEL_SHA
    assert code_sha(env={CODE_SHA_ENV: KERNEL_SHA[::-1]}) == KERNEL_SHA[::-1]


@pytest.mark.phase(3)
@pytest.mark.parametrize("value", [f"{KERNEL_SHA}-dirty", KERNEL_SHA[:7], KERNEL_SHA.upper(), " "])
def test_kernel_variable_must_be_a_full_clean_commit(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match=f"{CODE_SHA_ENV} must be a full 40-character commit"):
        code_sha(tmp_path, env={CODE_SHA_ENV: value})


@pytest.mark.phase(3)
def test_accelerator_names_the_gpu_of_each_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    asked: list[tuple[str, ...]] = []

    def sysctl(stdout: str, code: int = 0) -> Any:
        def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            asked.append(tuple(command))
            return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")

        return run

    monkeypatch.setattr(subprocess, "run", sysctl("Apple M4\n"))
    assert accelerator("local-mps") == "Apple M4"
    assert asked == [("sysctl", "-n", "machdep.cpu.brand_string")]
    # A Mac that won't name its chip, and a machine with no sysctl, still have MPS.
    monkeypatch.setattr(subprocess, "run", sysctl("", code=1))
    assert accelerator("local-mps") == "mps"

    def no_sysctl(command: list[str], **kwargs: Any) -> None:
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(subprocess, "run", no_sysctl)
    assert accelerator("local-mps") == "mps"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Tesla T4")
    assert accelerator("kaggle") == "Tesla T4"
    assert accelerator("local-cpu") is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert accelerator("colab") is None
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch now raises ImportError
    assert accelerator("modal") is None


@pytest.mark.phase(3)
def test_close_counts_wall_clock_hours_on_gpu_platforms(repo: Path) -> None:
    started = datetime(2026, 11, 15, 9, 0, tzinfo=UTC)
    ended = started + timedelta(minutes=45)
    gpu = {"local-mps", "kaggle", "colab", "modal"}
    for platform in sorted(PLATFORMS):
        manifest = build_manifest(
            "eval", {}, model_ref="m", seed=0, repo=repo, platform=platform, now=started
        )
        closed = close(manifest, "stopped_time_guard", now=ended)
        assert (closed.status, closed.ended_at) == ("stopped_time_guard", ended)
        # Only the laptop's CPU holds no GPU while the run goes on.
        assert closed.gpu_hours == pytest.approx(0.75 if platform in gpu else 0.0), platform
        assert (manifest.status, manifest.ended_at, manifest.gpu_hours) == ("running", None, 0.0)

    kaggle = build_manifest("eval", {}, model_ref="m", seed=0, repo=repo, platform="kaggle")
    assert close(kaggle, "finished").ended_at is not None
    with pytest.raises(ValueError, match="timezone-aware"):
        close(kaggle, "failed", now=datetime(2026, 11, 15, 10))
    early = build_manifest(
        "eval", {}, model_ref="m", seed=0, repo=repo, platform="kaggle", now=ended
    )
    with pytest.raises(ValueError, match="can't end"):
        close(early, "finished", now=started)
