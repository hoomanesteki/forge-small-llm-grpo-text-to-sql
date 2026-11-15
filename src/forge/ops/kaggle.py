"""Kaggle: the metadata a kernel push sends, and the datasets a kernel mounts.

``kaggle kernels push -p DIR`` reads ``DIR/kernel-metadata.json`` and uploads the one
script it names. The CLI (kaggle 2.2.5) lets some mistakes through quietly: it ignores a
misspelt key, a missing ``enable_internet`` means internet on although its docs say off,
and a retired accelerator such as the P100 is swapped for a T4 with only a warning.
``check_kernel_metadata`` makes each of those an error here, so a bad push fails on the
laptop instead of in a queued GPU session.

Every Forge kernel is a private Python script. The template in
``kaggle/kernel-metadata.template.json`` asks for a T4 with internet; a CPU kernel drops
the GPU and the machine shape.

A GPU job mounts two private datasets: ``<owner>/forge-code``, the wheel built at one
commit with ``requirements-gpu.txt`` and ``GIT_SHA``, and ``<owner>/forge-spider``, the
Spider files ``spider.sha256`` lists, with that file. ``KaggleClient`` runs the kaggle CLI
as an argument list, never through a shell, and reads what it prints as well as its exit
code: ``datasets create`` and ``version`` report a refused upload and still exit 0.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.ops.dotenv import forge_env

TEMPLATE = Path("kaggle") / "kernel-metadata.template.json"  # relative to the repo root
METADATA_FILE = "kernel-metadata.json"
GPU_MACHINE_SHAPE = "NvidiaTeslaT4"

# Every key kaggle 2.2.5's kernels_push reads. Any other key is a typo the CLI would drop.
KNOWN_KEYS = frozenset(
    {
        "id",
        "id_no",
        "title",
        "code_file",
        "language",
        "kernel_type",
        "is_private",
        "enable_gpu",
        "enable_tpu",
        "enable_internet",
        "machine_shape",
        "dataset_sources",
        "kernel_sources",
        "competition_sources",
        "model_sources",
        "docker_image",
        "docker_image_pinning_type",
        "keywords",
    }
)
# How many "/"-separated parts each kind of source has: owner/slug for datasets and
# kernels, a bare slug for competitions, owner/model/framework/variation/version for models.
SOURCE_PARTS = {
    "dataset_sources": 2,
    "kernel_sources": 2,
    "competition_sources": 1,
    "model_sources": 5,
}
# The line of a kernel script that bundling replaces with the script's arguments: Kaggle
# runs a script kernel with none, so they have to be written into the script itself.
ARGV_LINE = "KERNEL_ARGV: list[str] = []"

_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def title_slug(title: str) -> str:
    """The slug Kaggle derives from a kernel title: lowercase words joined by hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def load_template(root: Path) -> dict[str, Any]:
    data = json.loads((root / TEMPLATE).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{TEMPLATE} must hold a JSON object")
    return data


def check_kernel_metadata(meta: Mapping[str, Any]) -> list[str]:
    """Every way ``meta`` breaks Forge's rules for a kernel push; empty when it is fine."""
    problems = [f"unknown key {key!r}" for key in sorted(set(meta) - KNOWN_KEYS)]
    problems += [
        f"{key} still holds a template placeholder: {value!r}"
        for key, value in _strings(meta)
        if _PLACEHOLDER.search(value)
    ]
    if problems:
        return problems

    kernel_id = meta.get("id")
    parts = kernel_id.split("/") if isinstance(kernel_id, str) else []
    if len(parts) != 2:
        # A third part would be a version, which push refuses.
        problems.append(f"id must be 'owner/slug', got {kernel_id!r}")
    else:
        owner, slug = parts
        if not _OWNER.fullmatch(owner):
            problems.append(f"id has a bad owner {owner!r}")
        if not _SLUG.fullmatch(slug):
            problems.append(f"id has a bad slug {slug!r}: lowercase letters, digits, hyphens")
        title = meta.get("title")
        if not isinstance(title, str) or len(title) < 5:
            problems.append(f"title must be at least 5 characters, got {title!r}")
        elif title_slug(title) != slug:
            problems.append(f"title {title!r} slugifies to {title_slug(title)!r}, not {slug!r}")

    code_file = meta.get("code_file")
    if not isinstance(code_file, str) or not code_file.endswith(".py"):
        problems.append(f"code_file must name a .py script, got {code_file!r}")
    elif Path(code_file).is_absolute() or ".." in Path(code_file).parts:
        problems.append(f"code_file must be relative to the kernel folder, got {code_file!r}")

    for key, wanted in (("language", "python"), ("kernel_type", "script")):
        if meta.get(key) != wanted:
            problems.append(f"{key} must be {wanted!r}, got {meta.get(key)!r}")
    # The CLI also takes "true" and "false" strings; real booleans leave no doubt.
    for key in ("is_private", "enable_gpu", "enable_internet"):
        if not isinstance(meta.get(key), bool):
            problems.append(f"{key} must be set to true or false, got {meta.get(key)!r}")
    if meta.get("is_private") is False:
        problems.append("is_private must be true: Forge kernels are never public")
    if meta.get("enable_tpu"):
        problems.append("enable_tpu must be false: Forge runs on T4s")

    shape = meta.get("machine_shape")
    if meta.get("enable_gpu") is True and shape != GPU_MACHINE_SHAPE:
        # The P100 is retired: the server would run a T4 anyway, after a warning.
        problems.append(f"a GPU kernel needs machine_shape {GPU_MACHINE_SHAPE!r}, got {shape!r}")
    if meta.get("enable_gpu") is False and shape is not None:
        problems.append(f"a CPU kernel takes no machine_shape, got {shape!r}")

    for key, count in SOURCE_PARTS.items():
        sources = meta.get(key)
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            problems.append(f"{key} must be a list of strings, got {sources!r}")
            continue
        problems += [
            f"{key} entry {source!r} must have {count} '/'-separated parts"
            for source in sources
            if len(source.split("/")) != count or not all(source.split("/"))
        ]
    return problems


def render_kernel_metadata(
    template: Mapping[str, Any],
    values: Mapping[str, str],
    *,
    gpu: bool = True,
    dataset_sources: Sequence[str] = (),
) -> dict[str, Any]:
    """The template with its ``{name}`` placeholders filled from ``values``, checked.

    ``gpu=False`` makes a CPU kernel, which spends no GPU quota. Raises ``ValueError``
    listing every problem when the result would not pass ``check_kernel_metadata``.
    """

    def fill(value: Any) -> Any:
        if isinstance(value, str):
            return _PLACEHOLDER.sub(lambda m: _value(values, m.group(0)[1:-1]), value)
        if isinstance(value, list):
            return [fill(item) for item in value]
        return value

    meta = {key: fill(value) for key, value in template.items()}
    if dataset_sources:
        current = meta.get("dataset_sources")
        if not isinstance(current, list):
            raise ValueError(f"the template's dataset_sources must be a list, got {current!r}")
        meta["dataset_sources"] = [*current, *dataset_sources]
    if not gpu:
        meta["enable_gpu"] = False
        meta.pop("machine_shape", None)
    problems = check_kernel_metadata(meta)
    if problems:
        raise ValueError("kernel metadata is not valid:\n- " + "\n- ".join(problems))
    return meta


def bake_argv(source: str, argv: Sequence[str]) -> str:
    """``source`` with its ``ARGV_LINE`` set to ``argv``, so the kernel runs with them."""
    lines = source.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == ARGV_LINE]
    if len(hits) != 1:
        raise ValueError(f"the script must have the line {ARGV_LINE!r} exactly once")
    if not all(isinstance(arg, str) for arg in argv):
        raise TypeError("kernel arguments must be strings")
    # A JSON list of strings is also a Python list literal.
    lines[hits[0]] = f"KERNEL_ARGV: list[str] = {json.dumps(list(argv))}\n"
    return "".join(lines)


def write_kernel_dir(directory: Path, meta: Mapping[str, Any], code: str) -> Path:
    """Write ``kernel-metadata.json`` and the script it names, ready for a push."""
    problems = check_kernel_metadata(meta)
    if problems:
        raise ValueError("kernel metadata is not valid:\n- " + "\n- ".join(problems))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / str(meta["code_file"])).write_text(code, encoding="utf-8")
    (directory / METADATA_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return directory


def _value(values: Mapping[str, str], name: str) -> str:
    if name not in values:
        raise ValueError(f"the kernel template needs a value for {{{name}}}")
    return values[name]


def _strings(meta: Mapping[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for key, value in meta.items():
        items = value if isinstance(value, list) else [value]
        found += [(key, item) for item in items if isinstance(item, str)]
    return found


OWNER_ENV = "FORGE_KAGGLE_USER"

# kaggle 2.2.5 prints "Dataset creation error: ..." or "Dataset version creation error: ..."
# on stdout and exits 0 when the server refuses an upload.
_UPLOAD_REFUSED = re.compile(r"creation error", re.IGNORECASE)

# Gets a command as an argument list; returns its exit code and captured output.
Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class KaggleError(RuntimeError):
    """A kaggle command failed, or printed a failure and exited 0."""


def run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run ``args`` without a shell, capturing output, with the repo's ``.env`` merged in.

    The merged environment matters: Kaggle credentials may live only in ``.env``.
    """
    return subprocess.run(list(args), capture_output=True, text=True, check=False, env=forge_env())


def kaggle_bin() -> str:
    """The kaggle script next to this interpreter, so an unactivated venv still works."""
    sibling = Path(sys.executable).parent / "kaggle"
    return str(sibling) if sibling.is_file() else "kaggle"


@dataclass(frozen=True)
class KaggleClient:
    """The kaggle CLI, run through ``run`` so tests can stand in for it."""

    run: Runner = run_command
    executable: str = field(default_factory=kaggle_bin)

    def dataset_exists(self, ref: str) -> bool:
        """Whether the dataset ``owner/slug`` exists; raises on any other failure."""
        command = [self.executable, "datasets", "status", ref]
        result = _run(self.run, command)
        if result.returncode == 0:
            return True
        # A missing dataset is a 404, which kaggle 2.2.5 reports as "... was not found."
        if "not found" in _output(result).lower():
            return False
        raise _failed(command, result)

    def create_dataset(self, folder: Path) -> str:
        """Create the dataset ``folder``'s metadata names; returns what the CLI printed."""
        # Never -u: without it a new dataset is private. zip mode uploads each subfolder
        # as an archive, where the default (skip) would leave it out without a word.
        return self._upload([self.executable, "datasets", "create", "-p", str(folder), "-r", "zip"])

    def version_dataset(self, folder: Path, message: str) -> str:
        """Upload ``folder`` as a new version, noted ``message``; returns what the CLI printed."""
        command = [self.executable, "datasets", "version", "-p", str(folder), "-m", message]
        return self._upload([*command, "-r", "zip"])

    def _upload(self, command: list[str]) -> str:
        output = _check(self.run, command)
        if _UPLOAD_REFUSED.search(output):
            raise KaggleError(f"{_display(command)} was refused: {_tail(output)}")
        return output.strip()


def kaggle_owner(env: Mapping[str, str], home: Path) -> str:
    """The Kaggle user who owns Forge's datasets.

    ``FORGE_KAGGLE_USER`` if set, else the username in the Kaggle credentials:
    ``KAGGLE_USERNAME``, or ``kaggle.json`` in ``KAGGLE_CONFIG_DIR`` or ``~/.kaggle``. An
    API token doesn't name its user, so with one ``FORGE_KAGGLE_USER`` has to be set.
    """
    owner = env.get(OWNER_ENV) or env.get("KAGGLE_USERNAME") or _kaggle_json_username(env, home)
    if not owner:
        raise KaggleError(
            f"no Kaggle username: set {OWNER_ENV} in .env or the environment, since an API "
            "token doesn't name its user"
        )
    if not _OWNER.fullmatch(owner):
        raise KaggleError(f"{owner!r} is not a Kaggle username")
    return owner


def _kaggle_json_username(env: Mapping[str, str], home: Path) -> str | None:
    config_dir = (
        Path(env["KAGGLE_CONFIG_DIR"]) if env.get("KAGGLE_CONFIG_DIR") else home / ".kaggle"
    )
    path = config_dir / "kaggle.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # The decoder's message gives a position, never the file's text, so no key leaks.
        raise KaggleError(f"{path} is not valid JSON: {exc}") from exc
    username = data.get("username") if isinstance(data, dict) else None
    return username if isinstance(username, str) and username else None


def _run(run: Runner, command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return run(command)
    except OSError as exc:  # FileNotFoundError when the program isn't installed
        raise KaggleError(f"could not run {Path(command[0]).name}: {exc}") from exc


def _check(run: Runner, command: list[str]) -> str:
    result = _run(run, command)
    if result.returncode != 0:
        raise _failed(command, result)
    return result.stdout


def _failed(command: list[str], result: subprocess.CompletedProcess[str]) -> KaggleError:
    return KaggleError(f"{_display(command)} exited {result.returncode}: {_tail(_output(result))}")


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stdout or ''}\n{result.stderr or ''}"


def _tail(text: str, lines: int = 3) -> str:
    # The last lines say why; earlier ones are upload progress and usage text.
    kept = [line.strip() for line in text.splitlines() if line.strip()]
    return " / ".join(kept[-lines:]) or "no output"


def _display(command: Sequence[str]) -> str:
    return shlex.join([Path(command[0]).name, *command[1:]])
