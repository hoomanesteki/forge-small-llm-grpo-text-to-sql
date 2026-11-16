"""The folders and arguments of a Kaggle job kernel, built on the laptop for tests."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from forge.data import spider
from forge.data.spider import file_sha256, raw_files

WHEEL = "forge-0.1.0-py3-none-any.whl"
# Built at run time, so no literal here looks like a real commit.
SHA = "".join("0123456789abcdef"[(i * 3) % 16] for i in range(40))
# What kaggle/entry.py's main sets, and tests put back.
KERNEL_ENV = (
    "FORGE_RUNS_DIR",
    "FORGE_CACHE_DIR",
    "FORGE_DATA_DIR",
    "FORGE_CODE_SHA",
    "FORGE_DEV_USAGE",
    "HF_TOKEN",
    "WANDB_API_KEY",
)
# Where build_inputs puts the checksum list the installed wheel would ship, relative to
# the kernel's folder: the full data's list can't check the fixture.
PINNED = Path("wheel") / "spider.sha256"


def kernel_folder(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``root`` as a kernel's folder: main's variables cleared and put back after, the
    working directory there, since main moves into its output folder, and the wheel's
    pinned checksums the list ``build_inputs`` writes there."""
    for name in KERNEL_ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(spider, "CHECKSUMS", root / PINNED)
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(root)
    return root


def mount_output(previous: Path, root: Path, slug: str = "forge-run-1") -> Path:
    """The output of the kernel that ran in ``previous``, mounted under ``root``'s input
    folder as a resume's kernel source is."""
    return Path(shutil.copytree(previous / "working", root / "input" / slug))


def write_checksums(data: Path) -> None:
    lines = [f"{file_sha256(data / rel)}  {rel}" for rel in raw_files(data)]
    (data / "spider.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_inputs(root: Path, data_dir: Path, *, sha: str = SHA, zipped: bool = False) -> Path:
    """The input folder as Kaggle mounts the code and data datasets, and the checksum list
    of ``data_dir`` as the wheel's pinned one, in ``root / PINNED``."""
    code = root / "input" / "forge-code"
    code.mkdir(parents=True)
    (code / "GIT_SHA").write_text(f"{sha}\n", encoding="utf-8")
    (code / WHEEL).write_bytes(b"PK\x03\x04 a wheel")
    (code / "requirements-gpu.txt").write_text("trl==1.18.0\n", encoding="utf-8")
    data = Path(shutil.copytree(data_dir, root / "input" / "forge-spider"))
    write_checksums(data)
    (root / PINNED).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data / "spider.sha256", root / PINNED)
    if zipped:
        # What `datasets create -r zip` uploads, if Kaggle doesn't unpack it.
        with zipfile.ZipFile(data / "database.zip", "w") as archive:
            for path in sorted((data / "database").rglob("*.sqlite")):
                archive.write(path, path.relative_to(data / "database").as_posix())
        shutil.rmtree(data / "database")
    return root / "input"


def argv(root: Path, *extra: str, options: dict[str, Any] | None = None) -> list[str]:
    """kaggle/entry.py's arguments for run ``run-1`` with its folders under ``root``."""
    return [
        *("--job", "eval", "--run-id", "run-1", "--code-sha", SHA),
        *("--options", json.dumps(options or {"model": "gold", "backend": "fake"})),
        *("--run-store", "forge-owner/forge-runs", "--input-dir", str(root / "input")),
        *("--work-dir", str(root / "working"), "--temp-dir", str(root / "temp")),
        *extra,
    ]
