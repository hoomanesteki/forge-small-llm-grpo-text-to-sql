"""Kaggle: the metadata a kernel push sends, and the datasets a kernel mounts.

``kaggle kernels push -p DIR`` reads ``DIR/kernel-metadata.json`` and uploads the one
script it names. The CLI (kaggle 2.2.5) lets some mistakes through quietly: it ignores a
misspelt key, a missing ``enable_internet`` means internet on although its docs say off,
and a retired accelerator such as the P100 is swapped for a T4 with only a warning. Kaggle
itself refuses a title over 50 characters with a bare ``400 Client Error: Bad Request``,
which the CLI passes on only after the code dataset is checked. ``check_kernel_metadata``
makes each of those an error here, so a bad push fails on the laptop instead of in a
queued GPU session.

Every Forge kernel is a private Python script. The template in
``kaggle/kernel-metadata.template.json`` asks for a T4 with internet; a CPU kernel drops
the GPU and the machine shape.

A GPU job mounts two private datasets: ``<owner>/forge-code``, the wheel built at one
commit with ``requirements-gpu.txt`` and ``GIT_SHA``, and ``<owner>/forge-spider``, the
Spider files ``spider.sha256`` lists, with that file. ``KaggleClient`` runs the kaggle CLI
as an argument list, never through a shell, and reads what it prints as well as its exit
code: ``datasets create`` and ``version`` report a refused upload and still exit 0, and so
does ``kernels push`` for a push error. A push is refused only when Kaggle says so, as
"Kernel push error": any other line means Kaggle saved and started the kernel, even one
that names no version (kaggle 2.2.5 prints one when the server sends none). When
``kernels push`` says a source could not be added, Kaggle has still saved and started
that version, without the source, so the push is returned with those lines in
``PushedKernel.dropped`` rather than refused. A CLI that exits with an error before it
says the kernel was pushed may only have lost Kaggle's answer, a timeout among them, after
Kaggle saved the kernel, so ``push_kernel`` asks Kaggle whether it has a kernel no earlier
push made before it calls the push failed.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from forge.data.spider import CHECKSUMS, load_checksums, verify_files
from forge.ops.dotenv import forge_env
from forge.ops.manifest import DirtyTreeError, git_sha

TEMPLATE = Path("kaggle") / "kernel-metadata.template.json"  # relative to the repo root
METADATA_FILE = "kernel-metadata.json"
GPU_MACHINE_SHAPE = "NvidiaTeslaT4"
# The longest kernel title Kaggle saves: FG-301 saved 50 characters and got the 400 at 51,
# with nothing else in the metadata changed. A slug is its title's, so it has the same bound.
MAX_KERNEL_NAME = 50

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
# Datasets and kernels may add /<version number>: kaggle 2.2.5 then mounts that version, not
# the latest, which is how a job runs the code dataset version that was checked.
VERSIONED_SOURCES = frozenset({"dataset_sources", "kernel_sources"})
# The line of a kernel script that bundling replaces with the script's arguments: Kaggle
# runs a script kernel with none, so they have to be written into the script itself.
ARGV_LINE = "KERNEL_ARGV: list[str] = []"

_SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
_VERSION = re.compile(r"[1-9][0-9]*")


def valid_slug(slug: str) -> bool:
    """Whether Kaggle takes ``slug``: lowercase letters and digits, single hyphens between."""
    return _SLUG.fullmatch(slug) is not None


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
        if not valid_slug(slug):
            problems.append(f"id has a bad slug {slug!r}: lowercase letters, digits, hyphens")
        if len(slug) > MAX_KERNEL_NAME:
            problems.append(_too_long("slug", slug))
        title = meta.get("title")
        if not isinstance(title, str) or len(title) < 5:
            problems.append(f"title must be at least 5 characters, got {title!r}")
        elif len(title) > MAX_KERNEL_NAME:
            problems.append(_too_long("title", title))
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
            problem for source in sources if (problem := _source_problem(key, count, source))
        ]
    return problems


def _too_long(what: str, name: str) -> str:
    return (
        f"{what} {name!r} has {len(name)} characters, and Kaggle refuses a kernel's over "
        f"{MAX_KERNEL_NAME}"
    )


def _source_problem(key: str, count: int, source: str) -> str | None:
    """Why ``source`` can't be an entry of ``key``, or ``None`` when it can."""
    parts = source.split("/")
    if all(parts) and len(parts) == count:
        return None
    if all(parts) and key in VERSIONED_SOURCES and len(parts) == count + 1:
        if _VERSION.fullmatch(parts[-1]):
            return None
        return f"{key} entry {source!r} must end in a version number, not {parts[-1]!r}"
    pinned = f", or {count + 1} to pin a version" if key in VERSIONED_SOURCES else ""
    return f"{key} entry {source!r} must have {count} '/'-separated parts{pinned}"


def render_kernel_metadata(
    template: Mapping[str, Any],
    values: Mapping[str, str],
    *,
    gpu: bool = True,
    dataset_sources: Sequence[str] = (),
    kernel_sources: Sequence[str] = (),
) -> dict[str, Any]:
    """The template with its ``{name}`` placeholders filled from ``values``, checked.

    ``gpu=False`` makes a CPU kernel, which spends no GPU quota. The sources are added to
    the template's. Raises ``ValueError`` listing every problem when the result would not
    pass ``check_kernel_metadata``.
    """

    def fill(value: Any) -> Any:
        if isinstance(value, str):
            return _PLACEHOLDER.sub(lambda m: _value(values, m.group(0)[1:-1]), value)
        if isinstance(value, list):
            return [fill(item) for item in value]
        return value

    meta = {key: fill(value) for key, value in template.items()}
    for key, extra in (("dataset_sources", dataset_sources), ("kernel_sources", kernel_sources)):
        if extra:
            current = meta.get(key)
            if not isinstance(current, list):
                raise ValueError(f"the template's {key} must be a list, got {current!r}")
            meta[key] = [*current, *extra]
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


CODE_SLUG = "forge-code"
DATA_SLUG = "forge-spider"
OWNER_ENV = "FORGE_KAGGLE_USER"
DATASET_METADATA_FILE = "dataset-metadata.json"
GIT_SHA_FILE = "GIT_SHA"
CHECKSUM_FILE = "spider.sha256"
# Relative to the repo root; git ignores build/.
REQUIREMENTS = Path("kaggle") / "requirements-gpu.txt"
CODE_BUNDLE = Path("build") / "kaggle-code"
DATA_BUNDLE = Path("build") / "kaggle-data"
# kagglesdk's DatabundleVersionStatus names, lowercased as `kaggle datasets status` prints
# them. Only a ready dataset can be mounted, a failed or deleted version never will be,
# and every other one is a step on the way to ready.
DATASET_READY = "ready"
DATASET_BROKEN = frozenset({"failed", "deleted"})
DATASET_STATUSES = DATASET_BROKEN | {
    DATASET_READY,
    "not_yet_persisted",
    "blobs_received",
    "blobs_decompressed",
    "blobs_copied_to_sds",
    "individual_blobs_compressed",
    "reprocessing",
}

# CC BY-SA 4.0 asks for the authors, the license and its link wherever the data goes.
SPIDER_DESCRIPTION = (
    "Spider 1.0 for Forge's text-to-SQL jobs: the question and schema JSON files and every "
    "`database/<db>/<db>.sqlite`, unchanged, each listed with its sha256 in `spider.sha256`."
    "\n\n"
    "Spider 1.0 by Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, "
    "James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang and Dragomir Radev "
    "(Yale LILY), licensed under CC BY-SA 4.0 "
    "(https://creativecommons.org/licenses/by-sa/4.0/)."
    "\n\n"
    'Yu et al., "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain '
    'Semantic Parsing and Text-to-SQL Task", EMNLP 2018, pages 3911-3921, '
    "doi:10.18653/v1/D18-1425.\n"
)

# One package at one exact version: ``name==1.2.3``, with extras allowed.
_PINNED = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9._+!-]+")
# kaggle 2.2.5 prints "Dataset creation error: ..." or "Dataset version creation error: ..."
# on stdout and exits 0 when the server refuses an upload.
_UPLOAD_REFUSED = re.compile(r"creation error", re.IGNORECASE)
# kernels push exits 0 after "Kernel push error: ...". It prints "The following are not
# valid dataset sources and could not be added to the kernel: [...]" (or kernel, tag or
# competition sources) before "Kernel version N successfully pushed": that version runs.
# With no version from the server it prints "Kernel version successfully pushed".
_PUSH_REFUSED = re.compile(r"Kernel push error", re.IGNORECASE)
_DROPPED = re.compile(r"could not be added", re.IGNORECASE)
_DROPPED_KERNEL = re.compile(r"kernel sources", re.IGNORECASE)
_PUSHED = re.compile(r"successfully pushed", re.IGNORECASE)
_PUSHED_VERSION = re.compile(r"Kernel version (\d+)")
_KERNEL_URL = re.compile(r"kaggle\.com/code/([^/\s]+)/([^/\s?#]+)")
# kernels status prints 'owner/slug has status "KernelWorkerStatus.COMPLETE"'.
_KERNEL_STATUS = re.compile(r'has status "(?:KernelWorkerStatus\.)?([A-Z_]+)"')
_FAILURE = re.compile(r'Failure message: "(.*)"', re.DOTALL)
# The worker states of kagglesdk's KernelWorkerStatus, and whether each is the end.
KERNEL_STATES = {
    "QUEUED": False,
    "RUNNING": False,
    "NEW_SCRIPT": False,
    "COMPLETE": True,
    "ERROR": True,
    "CANCEL_REQUESTED": True,
    "CANCEL_ACKNOWLEDGED": True,
}

# Gets a command as an argument list; returns its exit code and captured output.
Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class KaggleError(RuntimeError):
    """A kaggle or uv command failed, or printed a failure and exited 0."""


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
        """Whether the dataset ``owner/slug`` exists, in any status; raises on any other
        failure."""
        return self.dataset_status(ref) is not None

    def dataset_status(self, ref: str) -> str | None:
        """The status of the dataset ``owner/slug``, one of ``DATASET_STATUSES``, or None
        when there is no such dataset; raises on any other failure."""
        command = [self.executable, "datasets", "status", ref]
        result = _run(self.run, command)
        if result.returncode == 0:
            # kaggle 2.2.5 prints the status alone, last, and exits 0 whatever it is.
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            status = lines[-1] if lines else ""
            if status not in DATASET_STATUSES:
                raise KaggleError(f"{_display(command)} printed {_tail(result.stdout)!r}")
            return status
        # A missing dataset is a 404, "... was not found.", or, as the live API answers for
        # a slug in the owner's own namespace that holds none, a 403, "You don't have access
        # to this resource.": Kaggle tells no one which datasets exist. Either way the first
        # publish creates it, and a create that still can't is refused with its reason.
        text = _output(result).lower()
        if "not found" in text or "don't have access to this resource" in text:
            return None
        raise _failed(command, result)

    def create_dataset(self, folder: Path) -> str:
        """Create the dataset ``folder``'s metadata names; returns what the CLI printed."""
        # Never -u: without it a new dataset is private. zip mode uploads each subfolder
        # as an archive, where the default (skip) would leave it out with only a printed
        # "Skipping folder" note, and exit 0.
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

    def dataset_version(self, ref: str) -> DatasetVersion:
        """Whether the dataset ``owner/slug`` is processed, and its current version number."""
        fields = "json(status,current_version_number)"
        command = [self.executable, "datasets", "status", ref, "--format", fields]
        output = _check(self.run, command)
        try:
            data = json.loads(output)
            return DatasetVersion(str(data["status"]), int(data["current_version_number"]))
        except (ValueError, TypeError, KeyError) as exc:
            raise KaggleError(f"{_display(command)} printed {_tail(output)!r}") from exc

    def dataset_file(self, ref: str, name: str, folder: Path) -> Path:
        """Download the file ``name`` of the dataset ``ref`` (``owner/slug[/version]``)."""
        command = [self.executable, "datasets", "download", ref, "-f", name, "-p", str(folder)]
        _check(self.run, [*command, "-o", "-q", "--unzip"])
        path = folder / name
        if not path.is_file():
            raise KaggleError(f"kaggle downloaded no {name} from {ref} into {folder}")
        return path

    def push_kernel(
        self, folder: Path, *, timeout_s: int, gpu: bool, new_kernel: bool = False
    ) -> PushedKernel:
        """Push and start the kernel ``folder`` holds; returns its id, its version and
        the sources Kaggle dropped from it, if any.

        A CLI that fails before it says the kernel was pushed may have lost the answer to
        a push Kaggle saved. With ``new_kernel``, for a kernel no earlier push made, Kaggle
        is then asked whether it has the kernel (``kaggle kernels status``): if it does,
        the push was made, and is returned with no version, so the kernel's latest is
        followed, and with ``lost`` set to the CLI's error. A push to a kernel an earlier
        push made is not checked, since the status names no version and so can't tell
        this push's from the one before. Raises ``KaggleError`` for a push Kaggle refused,
        one whose kernel it doesn't have, or one it couldn't be asked about, which says so.
        """
        accelerator = ["--accelerator", GPU_MACHINE_SHAPE] if gpu else []
        command = [self.executable, "kernels", "push", "-p", str(folder), *accelerator]
        result = _run(self.run, [*command, "-t", str(timeout_s)])
        output = result.stdout or ""
        if _PUSH_REFUSED.search(output):
            raise KaggleError(f"{_display(command)} was refused: {_tail(output)}")
        if result.returncode != 0 and not _PUSHED.search(output):
            failure = _failed(command, result)
            if not new_kernel:
                raise failure
            sent = _pushed_id(folder)
            try:
                saved = self.find_kernel(sent)
            except KaggleError as exc:
                raise KaggleError(
                    f"{failure}; whether Kaggle saved {sent} all the same is unknown, since "
                    f"asking failed too ({exc}): check kaggle kernels status {sent} before "
                    "pushing again"
                ) from exc
            if saved is None:
                raise failure
            return PushedKernel(sent, None, output.strip(), lost=str(failure))
        url = _KERNEL_URL.search(output)
        kernel = "/".join(url.groups()) if url else None
        version = _PUSHED_VERSION.search(output)
        dropped = tuple(line.strip() for line in output.splitlines() if _DROPPED.search(line))
        number = int(version.group(1)) if version else None
        return PushedKernel(kernel, number, output.strip(), dropped)

    def kernel_status(self, kernel: str) -> KernelStatus:
        """The worker state of ``kernel`` (``owner/slug[/version]``), as kagglesdk names it."""
        command = [self.executable, "kernels", "status", kernel]
        return _kernel_state(command, _check(self.run, command))

    def find_kernel(self, kernel: str) -> KernelStatus | None:
        """The worker state of ``kernel``, or None when Kaggle has no such kernel; raises
        on any other failure."""
        command = [self.executable, "kernels", "status", kernel]
        result = _run(self.run, command)
        if result.returncode == 0:
            return _kernel_state(command, result.stdout)
        # kaggle 2.2.5 prints a 404 as "404 Client Error: Not Found ...", and turns a 403,
        # which Kaggle may answer for a private slug it holds nothing at, into "Cannot
        # access kernel ... (Permission 'kernels.get' was denied)".
        text = _output(result).lower()
        if "404 client error" in text or "permission 'kernels.get' was denied" in text:
            return None
        raise _failed(command, result)

    def kernel_logs(self, kernel: str) -> str:
        return _check(self.run, [self.executable, "kernels", "logs", kernel])

    def kernel_output(self, kernel: str, folder: Path, pattern: str | None = None) -> None:
        """Download the files ``kernel`` wrote, and its log, into ``folder``."""
        command = [self.executable, "kernels", "output", kernel, "-p", str(folder), "-o", "-q"]
        _check(self.run, [*command, *(["--file-pattern", pattern] if pattern else [])])

    def gpu_hours_left(self) -> float:
        """The GPU hours left this week by ``kaggle quota -v``; raises when it can't say."""
        command = [self.executable, "quota", "-v"]
        output = _check(self.run, command)
        for row in csv.DictReader(io.StringIO(output)):
            if row.get("resource") == "GPU":
                try:
                    return float(row["remaining"].removesuffix("h"))
                except (AttributeError, ValueError) as exc:
                    raise KaggleError(f"{_display(command)} printed {_tail(output)!r}") from exc
        raise KaggleError(f"{_display(command)} shows no GPU quota: {_tail(output)}")


@dataclass(frozen=True)
class DatasetVersion:
    status: str  # "ready" once Kaggle has processed the upload
    number: int


@dataclass(frozen=True)
class PushedKernel:
    kernel: str | None  # owner/slug from the URL the CLI printed, when it printed one
    version: int | None  # None when the CLI named none: the kernel's latest is this push
    output: str
    dropped: tuple[str, ...] = ()  # the CLI's lines naming sources it could not add
    lost: str | None = None  # the CLI's error, when Kaggle had the kernel all the same

    @property
    def kernel_source_dropped(self) -> bool:
        """Whether Kaggle dropped a kernel source, which only a resume mounts: without its
        last attempt's output, this one can't restore the run."""
        return any(_DROPPED_KERNEL.search(line) for line in self.dropped)


@dataclass(frozen=True)
class KernelStatus:
    state: str  # a KERNEL_STATES key
    failure: str | None = None

    @property
    def done(self) -> bool:
        return KERNEL_STATES[self.state]


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


def dataset_metadata(ref: str, license_name: str, description: str | None = None) -> dict[str, Any]:
    """``dataset-metadata.json`` for the private dataset ``ref``, titled with its slug."""
    meta: dict[str, Any] = {"id": ref, "title": ref.split("/")[1]}
    if description is not None:
        meta["description"] = description
    meta["licenses"] = [{"name": license_name}]
    # create keeps a dataset private without -u, but `kaggle datasets metadata --update`
    # makes it public unless the metadata says otherwise.
    meta["isPrivate"] = True
    return meta


def unpinned_requirements(text: str) -> list[str]:
    """The lines of a requirements file that don't pin one exact version with ``==``."""
    lines = (raw.split("#", 1)[0].strip() for raw in text.splitlines())
    return [line for line in lines if line and not _PINNED.fullmatch(line)]


@dataclass(frozen=True)
class CodeBundle:
    path: Path
    wheel: str  # the wheel's file name
    git_sha: str


@dataclass(frozen=True)
class Published:
    ref: str
    created: bool  # False for a new version of a dataset that was there
    output: str  # what the kaggle CLI printed


def build_code_bundle(root: Path, owner: str, *, run: Runner = run_command) -> CodeBundle:
    """Write ``build/kaggle-code``: the wheel built at HEAD, the pinned GPU requirements,
    ``GIT_SHA`` and the dataset metadata, and nothing else.

    A tree with uncommitted changes is refused before anything is built: the wheel would
    hold code that ``GIT_SHA`` doesn't name.
    """
    try:
        sha = git_sha(root)
    except DirtyTreeError as exc:
        raise DirtyTreeError(
            "the working tree has uncommitted changes; commit them first, so the code "
            "dataset holds a commit CI can test"
        ) from exc
    unpinned = unpinned_requirements((root / REQUIREMENTS).read_text(encoding="utf-8"))
    if unpinned:
        raise ValueError(f"{REQUIREMENTS} must pin each package with ==, not {unpinned}")
    out = _fresh_dir(root / CODE_BUNDLE)
    _check(
        run, ["uv", "build", str(root), "--wheel", "--out-dir", str(out), "--no-create-gitignore"]
    )
    built = sorted(path.name for path in out.iterdir())
    if len(built) != 1 or not built[0].endswith(".whl"):
        raise KaggleError(f"uv build should leave one wheel in {out}, but left {built}")
    shutil.copyfile(root / REQUIREMENTS, out / REQUIREMENTS.name)
    (out / GIT_SHA_FILE).write_text(f"{sha}\n", encoding="utf-8")
    _write_json(out / DATASET_METADATA_FILE, dataset_metadata(f"{owner}/{CODE_SLUG}", "MIT"))
    return CodeBundle(out, built[0], sha)


def build_data_bundle(
    root: Path, owner: str, data_dir: Path, checksums: Traversable = CHECKSUMS
) -> Path:
    """Write ``build/kaggle-data``: each file ``checksums`` lists, the list itself as
    ``spider.sha256``, and the dataset metadata with Spider's attribution.

    Every file is checked against the list first, so only verified data is uploaded;
    ``ChecksumMismatch`` names each one that is missing or differs.
    """
    expected = load_checksums(checksums)
    verify_files(data_dir, expected)
    out = _fresh_dir(root / DATA_BUNDLE)
    for rel in sorted(expected):
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_dir / rel, target)
    (out / CHECKSUM_FILE).write_bytes(checksums.read_bytes())
    meta = dataset_metadata(f"{owner}/{DATA_SLUG}", "CC-BY-SA-4.0", SPIDER_DESCRIPTION)
    _write_json(out / DATASET_METADATA_FILE, meta)
    return out


def publish_code(root: Path, owner: str, client: KaggleClient) -> tuple[CodeBundle, Published]:
    """Build the code bundle and upload it: a new version noted with the SHA, or the
    dataset's first version when it doesn't exist yet. uv runs through the client's runner."""
    bundle = build_code_bundle(root, owner, run=client.run)
    ref = f"{owner}/{CODE_SLUG}"
    if client.dataset_exists(ref):
        return bundle, Published(ref, False, client.version_dataset(bundle.path, bundle.git_sha))
    return bundle, Published(ref, True, client.create_dataset(bundle.path))


def publish_data(
    root: Path,
    owner: str,
    client: KaggleClient,
    data_dir: Path,
    checksums: Traversable = CHECKSUMS,
    *,
    keep_build: bool = False,
) -> Published | None:
    """Upload the verified data as ``<owner>/forge-spider``; ``None`` if it is there.

    The checksum list pins Spider, so the data never changes and one upload serves every
    job: a dataset whose status says ready is left as it is. The check comes first, so
    that costs no copy of the data. A dataset whose last version failed or was deleted is
    not there, so the data goes up again as a new version, and one Kaggle is still
    processing raises ``KaggleError`` rather than go up twice. The bundle is a whole copy
    of Spider, so it is removed once the upload is over, whether or not it worked, unless
    ``keep_build``.
    """
    ref = f"{owner}/{DATA_SLUG}"
    status = client.dataset_status(ref)
    if status == DATASET_READY:
        return None
    if status is not None and status not in DATASET_BROKEN:
        raise KaggleError(
            f"{ref} is still being processed ({status}); run forge kaggle publish-data "
            f"again once kaggle datasets status {ref} says ready"
        )
    bundle = root / DATA_BUNDLE
    try:
        build_data_bundle(root, owner, data_dir, checksums)
        if status is None:
            return Published(ref, True, client.create_dataset(bundle))
        note = f"The verified Spider files again, after a {status} version"
        return Published(ref, False, client.version_dataset(bundle, note))
    finally:
        # A failed upload's copy serves nothing either: a rerun builds its own, or, when
        # Kaggle took the data before the CLI failed, finds the dataset ready and builds
        # none, which would leave this copy for good.
        if not keep_build and bundle.exists():
            shutil.rmtree(bundle)


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


def _fresh_dir(path: Path) -> Path:
    # Files left from an earlier bundle would be uploaded with this one.
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _kernel_state(command: list[str], output: str) -> KernelStatus:
    state = _KERNEL_STATUS.search(output)
    if state is None or state.group(1) not in KERNEL_STATES:
        raise KaggleError(f"{_display(command)} printed {_tail(output)!r}")
    failure = _FAILURE.search(output)
    return KernelStatus(state.group(1), failure.group(1).strip() if failure else None)


def _pushed_id(folder: Path) -> str:
    """The kernel ``folder``'s metadata names: the one ``kernels push`` sent."""
    meta = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    kernel = meta.get("id") if isinstance(meta, dict) else None
    if not isinstance(kernel, str):
        raise ValueError(f"{folder / METADATA_FILE} names no kernel id")
    return kernel


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
