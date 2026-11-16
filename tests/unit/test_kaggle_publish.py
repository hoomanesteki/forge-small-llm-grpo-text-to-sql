"""``forge kaggle publish-code`` and ``publish-data``: what each bundle holds, and the
uv and kaggle commands that build and upload it, with both programs stood in for."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge import cli
from forge.data.spider import ChecksumMismatch, file_sha256, raw_files
from forge.ops import kaggle
from forge.ops.kaggle import (
    CODE_BUNDLE,
    DATA_BUNDLE,
    KaggleClient,
    KaggleError,
    Published,
    build_code_bundle,
    build_data_bundle,
    kaggle_owner,
    publish_code,
    publish_data,
    run_command,
    unpinned_requirements,
)
from forge.ops.manifest import DirtyTreeError

ROOT = Path(__file__).resolve().parents[2]
OWNER = "forge-owner"
WHEEL = "forge-0.1.0-py3-none-any.whl"
# What kaggle 2.2.5 prints for `datasets status` on a dataset that doesn't exist.
NOT_FOUND = (
    "404 Client Error: Not Found for url: "
    "https://api.kaggle.com/v1/datasets.DatasetApiService/GetDatasetStatus\n\n"
    "The resource was not found. Check the spelling of the reference you passed.\n"
)
CREATED = "Your private Dataset is being created. Please check progress at https://kaggle.test\n"


def _done(
    command: list[str], code: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)


class FakeCli:
    """Stands in for uv and kaggle: records each command and answers it as they would."""

    def __init__(self, *exists: str) -> None:
        self.exists = set(exists)
        self.status: dict[str, str] = {}  # what datasets status prints, if not ready
        self.commands: list[list[str]] = []
        self.uploaded: list[list[str]] = []  # the files of each uploaded folder

    def __call__(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = list(args)
        self.commands.append(command)
        if command[:2] == ["uv", "build"]:
            out = Path(command[command.index("--out-dir") + 1])
            (out / WHEEL).write_bytes(b"PK\x03\x04 a wheel")
            return _done(command, stdout=f"Successfully built {out / WHEEL}\n")
        if command[1:3] == ["datasets", "status"]:
            if command[3] in self.exists:
                return _done(command, stdout=f"{self.status.get(command[3], 'ready')}\n")
            return _done(command, 1, stderr=NOT_FOUND)
        if command[1:3] in (["datasets", "create"], ["datasets", "version"]):
            folder = Path(command[command.index("-p") + 1])
            files = [p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()]
            self.uploaded.append(sorted(files))
            return _done(command, stdout=CREATED)
        raise AssertionError(f"unexpected command {command}")


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed checkout with the real GPU requirements, ignoring build/ as Forge does."""
    root = tmp_path / "repo"
    (root / "kaggle").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "kaggle" / "requirements-gpu.txt", root / "kaggle" / "requirements-gpu.txt"
    )
    (root / ".gitignore").write_text("build/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def checksums(spider_mini: Path, tmp_path: Path) -> Path:
    """A checksum list for the fixture data, in spider.sha256's format."""
    path = tmp_path / "spider.sha256"
    lines = [f"{file_sha256(spider_mini / rel)}  {rel}" for rel in raw_files(spider_mini)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _client(fake: FakeCli) -> KaggleClient:
    return KaggleClient(run=fake, executable="kaggle")


@pytest.mark.phase(3)
def test_code_bundle_contains_wheel_and_sha(repo: Path) -> None:
    # A wheel left by an earlier bundle must not ride along with this one.
    stale = repo / CODE_BUNDLE / "forge-0.0.9-py3-none-any.whl"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    fake = FakeCli()

    bundle = build_code_bundle(repo, OWNER, run=fake)

    head = _git(repo, "rev-parse", "HEAD")
    assert bundle.path == repo / CODE_BUNDLE
    assert (bundle.wheel, bundle.git_sha) == (WHEEL, head)
    assert sorted(p.name for p in bundle.path.iterdir()) == [
        "GIT_SHA",
        "dataset-metadata.json",
        WHEEL,
        "requirements-gpu.txt",
    ]
    assert (bundle.path / "GIT_SHA").read_text(encoding="utf-8") == f"{head}\n"
    requirements = (ROOT / "kaggle" / "requirements-gpu.txt").read_bytes()
    assert (bundle.path / "requirements-gpu.txt").read_bytes() == requirements
    assert unpinned_requirements(requirements.decode()) == []
    meta = json.loads((bundle.path / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta == {
        "id": "forge-owner/forge-code",
        "title": "forge-code",
        "licenses": [{"name": "MIT"}],
        "isPrivate": True,
    }
    assert fake.commands == [
        [
            "uv",
            "build",
            str(repo),
            "--wheel",
            "--out-dir",
            str(bundle.path),
            "--no-create-gitignore",
        ]
    ]
    # Building leaves the tree clean, so the next publish isn't refused.
    assert _git(repo, "status", "--porcelain") == ""


@pytest.mark.phase(3)
def test_code_bundle_refuses_a_dirty_tree(repo: Path) -> None:
    (repo / "kaggle" / "requirements-gpu.txt").write_text("trl==9.9.9\n", encoding="utf-8")
    fake = FakeCli()
    with pytest.raises(DirtyTreeError, match="commit them first"):
        build_code_bundle(repo, OWNER, run=fake)
    assert fake.commands == []
    assert not (repo / CODE_BUNDLE).exists()


@pytest.mark.phase(3)
def test_code_bundle_refuses_unpinned_requirements(repo: Path) -> None:
    (repo / "kaggle" / "requirements-gpu.txt").write_text(
        "trl==1.18.0  # pinned\nvllm>=0.33\n--extra-index-url https://example.test\n",
        encoding="utf-8",
    )
    _git(repo, "commit", "-q", "-am", "loosen")
    fake = FakeCli()
    with pytest.raises(ValueError, match=r"must pin each package with ==, not \['vllm>=0.33'"):
        build_code_bundle(repo, OWNER, run=fake)
    assert fake.commands == []


@pytest.mark.phase(3)
def test_code_bundle_needs_exactly_one_wheel(repo: Path) -> None:
    def no_wheel(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _done(list(args))

    with pytest.raises(KaggleError, match=r"one wheel .* but left \[\]"):
        build_code_bundle(repo, OWNER, run=no_wheel)

    def build_fails(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _done(list(args), 2, stderr="  × Failed to build `forge`\n  hatchling: bad config\n")

    with pytest.raises(KaggleError, match=r"uv build .* exited 2: .*hatchling: bad config"):
        build_code_bundle(repo, OWNER, run=build_fails)


@pytest.mark.phase(3)
def test_publish_commands(repo: Path, spider_mini: Path, checksums: Path) -> None:
    fake = FakeCli()
    client = _client(fake)
    code = str(repo / CODE_BUNDLE)
    data = str(repo / DATA_BUNDLE)
    uv_build = ["uv", "build", str(repo), "--wheel", "--out-dir", code, "--no-create-gitignore"]

    # The first publish creates each dataset; zip mode keeps the data's database/ folder.
    bundle, first = publish_code(repo, OWNER, client)
    assert first == Published("forge-owner/forge-code", True, CREATED.strip())
    assert publish_data(repo, OWNER, client, spider_mini, checksums) == Published(
        "forge-owner/forge-spider", True, CREATED.strip()
    )
    # A whole copy of Spider, gone once it is uploaded.
    assert not (repo / DATA_BUNDLE).exists()
    assert fake.commands == [
        uv_build,
        ["kaggle", "datasets", "status", "forge-owner/forge-code"],
        ["kaggle", "datasets", "create", "-p", code, "-r", "zip"],
        ["kaggle", "datasets", "status", "forge-owner/forge-spider"],
        ["kaggle", "datasets", "create", "-p", data, "-r", "zip"],
    ]

    # After that, code goes up as a new version noted with its SHA, and data not at all.
    fake.exists |= {"forge-owner/forge-code", "forge-owner/forge-spider"}
    fake.commands.clear()
    _, second = publish_code(repo, OWNER, client)
    assert second == Published("forge-owner/forge-code", False, CREATED.strip())
    assert publish_data(repo, OWNER, client, spider_mini, checksums) is None
    assert fake.commands == [
        uv_build,
        ["kaggle", "datasets", "status", "forge-owner/forge-code"],
        ["kaggle", "datasets", "version", "-p", code, "-m", bundle.git_sha, "-r", "zip"],
        ["kaggle", "datasets", "status", "forge-owner/forge-spider"],
    ]
    assert (
        fake.uploaded[0]
        == fake.uploaded[2]
        == sorted(["GIT_SHA", "dataset-metadata.json", WHEEL, "requirements-gpu.txt"])
    )
    assert "spider.sha256" in fake.uploaded[1]


@pytest.mark.phase(3)
@pytest.mark.parametrize("status", ["failed", "deleted"])
def test_publish_data_counts_only_a_ready_dataset_as_there(
    repo: Path, spider_mini: Path, checksums: Path, status: str
) -> None:
    fake = FakeCli("forge-owner/forge-spider")
    fake.status["forge-owner/forge-spider"] = status

    published = publish_data(repo, OWNER, _client(fake), spider_mini, checksums)

    # Kaggle never made that version ready, so the data goes up again as a new one.
    assert published == Published("forge-owner/forge-spider", False, CREATED.strip())
    note = f"The verified Spider files again, after a {status} version"
    data = str(repo / DATA_BUNDLE)
    assert fake.commands == [
        ["kaggle", "datasets", "status", "forge-owner/forge-spider"],
        ["kaggle", "datasets", "version", "-p", data, "-m", note, "-r", "zip"],
    ]
    assert "spider.sha256" in fake.uploaded[0]
    assert not (repo / DATA_BUNDLE).exists()


@pytest.mark.phase(3)
def test_publish_data_never_uploads_over_a_version_being_processed(
    repo: Path, spider_mini: Path, checksums: Path
) -> None:
    fake = FakeCli("forge-owner/forge-spider")
    fake.status["forge-owner/forge-spider"] = "blobs_received"

    with pytest.raises(KaggleError, match=r"still being processed \(blobs_received\)"):
        publish_data(repo, OWNER, _client(fake), spider_mini, checksums)
    assert [c[2] for c in fake.commands] == ["status"]
    assert not (repo / DATA_BUNDLE).exists()

    # A status kagglesdk doesn't name is no answer at all.
    fake.status["forge-owner/forge-spider"] = "Something went wrong"
    with pytest.raises(KaggleError, match="printed 'Something went wrong'"):
        publish_data(repo, OWNER, _client(fake), spider_mini, checksums)


@pytest.mark.phase(3)
def test_data_bundle_holds_the_verified_files_and_attribution(
    repo: Path, spider_mini: Path, checksums: Path
) -> None:
    out = build_data_bundle(repo, OWNER, spider_mini, checksums)

    listed = raw_files(spider_mini)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == sorted([*listed, "spider.sha256", "dataset-metadata.json"])
    assert all((out / rel).read_bytes() == (spider_mini / rel).read_bytes() for rel in listed)
    assert (out / "spider.sha256").read_bytes() == checksums.read_bytes()
    meta = json.loads((out / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert (meta["id"], meta["title"]) == ("forge-owner/forge-spider", "forge-spider")
    assert meta["licenses"] == [{"name": "CC-BY-SA-4.0"}]
    assert meta["isPrivate"] is True
    for cited in ("CC BY-SA 4.0", "creativecommons.org/licenses/by-sa/4.0", "Tao Yu", "Yu et al."):
        assert cited in meta["description"], cited
    assert "doi:10.18653/v1/D18-1425" in meta["description"]


@pytest.mark.phase(3)
def test_data_that_fails_its_checksums_is_not_uploaded(
    repo: Path, spider_mini: Path, checksums: Path, tmp_path: Path
) -> None:
    data = Path(shutil.copytree(spider_mini, tmp_path / "data"))
    (data / "tables.json").write_text("[]", encoding="utf-8")
    fake = FakeCli()

    with pytest.raises(ChecksumMismatch, match=r"tables\.json: sha256 [0-9a-f]{64}, expected"):
        publish_data(repo, OWNER, _client(fake), data, checksums)
    assert fake.commands == [["kaggle", "datasets", "status", "forge-owner/forge-spider"]]
    assert not (repo / DATA_BUNDLE).exists()


@pytest.mark.phase(3)
def test_kaggle_failures_are_errors(tmp_path: Path) -> None:
    def answer(code: int, stdout: str = "", stderr: str = "") -> KaggleClient:
        return KaggleClient(
            run=lambda args: _done(list(args), code, stdout, stderr), executable="kaggle"
        )

    assert answer(1, stderr=NOT_FOUND).dataset_exists("forge-owner/forge-code") is False
    # Anything but not found, such as missing credentials, stops the publish.
    with pytest.raises(
        KaggleError, match=r"kaggle datasets status forge-owner/forge-code exited 1"
    ):
        answer(1, stdout="Authentication required to call the Kaggle API.\n").dataset_exists(
            "forge-owner/forge-code"
        )
    # create and version print a refused upload and exit 0.
    refused = 'Dataset creation error: The requested slug "forge-code" is already in use.\n'
    with pytest.raises(KaggleError, match="was refused: Dataset creation error"):
        answer(0, stdout=refused).create_dataset(tmp_path)
    with pytest.raises(KaggleError, match="was refused: Dataset version creation error"):
        answer(0, stdout="Dataset version creation error: bad request\n").version_dataset(
            tmp_path, "abc"
        )

    def missing(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", args[0])

    with pytest.raises(KaggleError, match="could not run kaggle"):
        KaggleClient(run=missing, executable="/venv/bin/kaggle").dataset_exists("a/forge-code")


@pytest.mark.phase(3)
def test_run_command_passes_arguments_without_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    monkeypatch.delenv("FORGE_KAGGLE_USER", raising=False)
    (tmp_path / ".env").write_text("FORGE_KAGGLE_USER=from-dotenv\n", encoding="utf-8")
    arg = "a; echo injected $HOME `id`"
    script = "import os, sys; print(sys.argv[1]); print(os.environ['FORGE_KAGGLE_USER'])"

    result = run_command([sys.executable, "-c", script, arg])

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [arg, "from-dotenv"]


@pytest.mark.phase(3)
@pytest.mark.parametrize(
    ("env", "kaggle_json", "owner"),
    [
        ({"FORGE_KAGGLE_USER": "forge-owner", "KAGGLE_USERNAME": "other"}, None, "forge-owner"),
        ({"KAGGLE_USERNAME": "forge-owner"}, '{"username": "other", "key": "k"}', "forge-owner"),
        ({}, '{"username": "forge-owner", "key": "k"}', "forge-owner"),
        ({"KAGGLE_CONFIG_DIR": "{tmp}/config"}, '{"username": "forge-owner"}', "forge-owner"),
    ],
)
def test_owner_comes_from_the_override_or_the_credentials(
    tmp_path: Path, env: dict[str, str], kaggle_json: str | None, owner: str
) -> None:
    env = {name: value.replace("{tmp}", str(tmp_path)) for name, value in env.items()}
    if kaggle_json is not None:
        config_dir = Path(env.get("KAGGLE_CONFIG_DIR", tmp_path / ".kaggle"))
        config_dir.mkdir(parents=True)
        (config_dir / "kaggle.json").write_text(kaggle_json, encoding="utf-8")
    assert kaggle_owner(env, tmp_path) == owner


@pytest.mark.phase(3)
def test_owner_errors(tmp_path: Path) -> None:
    # An API token alone names no user.
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "access_token").write_text("x", encoding="utf-8")
    with pytest.raises(KaggleError, match="set FORGE_KAGGLE_USER"):
        kaggle_owner({}, tmp_path)
    with pytest.raises(KaggleError, match="is not a Kaggle username"):
        kaggle_owner({"FORGE_KAGGLE_USER": "a/b"}, tmp_path)
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(KaggleError, match=r"kaggle\.json is not valid JSON"):
        kaggle_owner({}, tmp_path)


@pytest.mark.phase(3)
def test_unpinned_requirements() -> None:
    text = "# header\n\ntrl==1.18.0  # note\npeft[extra]==0.21.3\nvllm>=0.33\ntorch\n-r more.txt\n"
    assert unpinned_requirements(text) == ["vllm>=0.33", "torch", "-r more.txt"]


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No credentials, .env or home of the developer's: an empty repo root and home."""
    monkeypatch.setenv("FORGE_ROOT", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("FORGE_KAGGLE_USER", "KAGGLE_USERNAME", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.phase(3)
def test_publish_cli_needs_a_username(cli_env: Path) -> None:
    for command in ("publish-code", "publish-data"):
        result = CliRunner().invoke(cli.app, ["kaggle", command])
        assert result.exit_code == 1, result.output
        assert "error: no Kaggle username: set FORGE_KAGGLE_USER" in result.stderr


@pytest.mark.phase(3)
def test_publish_code_cli(repo: Path, cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_ROOT", str(repo))
    monkeypatch.setenv("FORGE_KAGGLE_USER", OWNER)
    fake = FakeCli("forge-owner/forge-code")
    monkeypatch.setattr(kaggle, "KaggleClient", lambda: _client(fake))

    result = CliRunner().invoke(cli.app, ["kaggle", "publish-code"])

    assert result.exit_code == 0, result.output
    sha = _git(repo, "rev-parse", "HEAD")
    assert result.stdout.splitlines() == [
        f"bundle {repo / CODE_BUNDLE}: {WHEEL}, requirements-gpu.txt, GIT_SHA",
        f"uploaded a new version of forge-owner/forge-code at {sha}",
        CREATED.strip(),
    ]

    # An untracked file is enough to refuse: the wheel would carry it under this SHA.
    (repo / "scratch.py").write_text("print('not committed')\n", encoding="utf-8")
    fake.commands.clear()
    result = CliRunner().invoke(cli.app, ["kaggle", "publish-code"])
    assert result.exit_code == 1, result.output
    assert "error: the working tree has uncommitted changes" in result.stderr
    assert fake.commands == []


@pytest.mark.phase(3)
def test_publish_data_cli_leaves_a_published_dataset(
    cli_env: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_KAGGLE_USER", OWNER)
    fake = FakeCli("forge-owner/forge-spider")
    monkeypatch.setattr(kaggle, "KaggleClient", lambda: _client(fake))

    result = CliRunner().invoke(cli.app, ["kaggle", "publish-data"])

    assert result.exit_code == 0, result.output
    assert "forge-owner/forge-spider is ready already" in result.stdout
    assert fake.commands == [["kaggle", "datasets", "status", "forge-owner/forge-spider"]]


@pytest.mark.phase(3)
def test_publish_data_cli_removes_its_bundle_unless_kept(
    repo: Path,
    cli_env: Path,
    spider_mini: Path,
    checksums: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_KAGGLE_USER", OWNER)
    monkeypatch.setenv("FORGE_ROOT", str(repo))
    # The fixture is a subset of Spider, so its own checksum list stands in for the pinned.
    monkeypatch.setattr(
        kaggle, "publish_data", functools.partial(publish_data, checksums=checksums)
    )
    fake = FakeCli()
    monkeypatch.setattr(kaggle, "KaggleClient", lambda: _client(fake))

    kept = CliRunner().invoke(cli.app, ["kaggle", "publish-data", "--keep-build"])

    assert kept.exit_code == 0, kept.output
    assert f"bundle {repo / DATA_BUNDLE} (kept)" in kept.stdout
    assert "created forge-owner/forge-spider" in kept.stdout
    assert (repo / DATA_BUNDLE / "spider.sha256").is_file()

    fake.exists.add("forge-owner/forge-spider")
    fake.status["forge-owner/forge-spider"] = "failed"
    again = CliRunner().invoke(cli.app, ["kaggle", "publish-data"])

    assert again.exit_code == 0, again.output
    assert f"bundle {repo / DATA_BUNDLE} (removed after the upload)" in again.stdout
    assert "uploaded a new version of forge-owner/forge-spider" in again.stdout
    assert not (repo / DATA_BUNDLE).exists()


@pytest.mark.phase(3)
def test_publish_data_cli_refuses_data_that_is_not_the_full_download(
    cli_env: Path, spider_mini: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fixture is a subset, so most of spider.sha256's files are missing from it.
    monkeypatch.setenv("FORGE_KAGGLE_USER", OWNER)
    fake = FakeCli()
    monkeypatch.setattr(kaggle, "KaggleClient", lambda: _client(fake))

    result = CliRunner().invoke(cli.app, ["kaggle", "publish-data"])

    assert result.exit_code == 1, result.output
    assert "differ from spider.sha256" in result.stderr
    assert "forge data fetch" in result.stderr
    assert [c[2] for c in fake.commands] == ["status"]


@pytest.mark.phase(3)
def test_publish_code_outside_a_checkout_is_one_error_line(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keeps git from finding a checkout that happens to hold the temp folder.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(cli_env.parent))
    monkeypatch.setenv("FORGE_KAGGLE_USER", OWNER)
    result = CliRunner().invoke(cli.app, ["kaggle", "publish-code"])
    assert result.exit_code == 1, result.output
    assert f"error: {cli_env} is not in a git checkout" in result.stderr
    assert isinstance(result.exception, SystemExit)
