"""The kaggle CLI wrapper that publishes Forge's datasets, with kaggle stood in for: how
its answers are read, and whose datasets they are."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from forge.ops.kaggle import (
    KaggleClient,
    KaggleError,
    kaggle_owner,
    run_command,
)

# What kaggle 2.2.5 prints for `datasets status` on a dataset that doesn't exist.
NOT_FOUND = (
    "404 Client Error: Not Found for url: "
    "https://api.kaggle.com/v1/datasets.DatasetApiService/GetDatasetStatus\n\n"
    "The resource was not found. Check the spelling of the reference you passed.\n"
)


def _done(
    command: list[str], code: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)


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
