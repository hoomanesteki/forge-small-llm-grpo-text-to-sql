"""What an eval run evaluates: the options of ``forge eval run``, checked before it starts.

``EvalConfig`` holds one field per option. ``check_config`` refuses options that no run
can take, and those that a later story builds, before any data is loaded or any file
written. ``manifest_config`` is what a run's manifest records and hashes: every setting
that can change a score, the sandbox and official-EX bounds among them (ADR-0008), so a
resumed run has to match it exactly. That includes the device, since greedy decoding on
another device can pick another token where two are nearly tied; the manifest's platform
follows it (ADR-0016). ``read_config_file`` reads the options a file in ``configs/eval/``
sets, which ``forge eval run --config`` takes as its defaults.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from forge.core.types import GenParams, Split
from forge.data.prompts import MAX_PROMPT_TOKENS
from forge.data.schema import check_version
from forge.data.splits import SPLITS
from forge.eval.agreement import GOLD_TIMEOUT_MS
from forge.eval.gold_cache import MAX_ROWS
from forge.infer.base import check_params
from forge.ops.manifest import Platform
from forge.sql.executor import DEFAULT_TIMEOUT_MS

Backend = Literal["hf", "vllm", "llamacpp", "fake"]
# Where the hf backend runs the model: the CPU, or Apple silicon's GPU through MPS.
Device = Literal["cpu", "mps"]
Store = Literal["local", "hub"]
BACKENDS: tuple[Backend, ...] = get_args(Backend)
DEVICES: tuple[Device, ...] = get_args(Device)
STORES: tuple[Store, ...] = get_args(Store)
# The fake backend's models: one answers with each question's gold query, the other
# always with SELECT 1. Tests build other fakes with FakeBackend(fn).
FAKE_MODELS = ("gold", "select-1")
# Backends that later stories build, with the story that brings each.
LATER_BACKENDS: Mapping[str, str] = {"vllm": "FG-306", "llamacpp": "FG-702"}

PRED_TIMEOUT_MS = DEFAULT_TIMEOUT_MS
# Spider's gold queries run in milliseconds, so 10 s is ample for an item, where the
# evaluator's own 60 s would let one adversarial prediction hold it for a minute while
# its memory grows (ADR-0008).
OFFICIAL_ITEM_TIMEOUT_S = 10.0

# What a config file may set, with the kind of value each takes: the options that can
# change a score. The run id, the store and a dev reason belong to one invocation, so
# they stay on the command line.
FILE_OPTIONS: Mapping[str, str] = {
    "model": "text",
    "adapter": "text",
    "backend": "text",
    "device": "text",
    "split": "text",
    "limit": "a whole number",
    "prompt_version": "text",
    "n": "a whole number",
    "temperature": "a number",
    "retry": "a whole number",
}
_KINDS: Mapping[str, tuple[type, ...]] = {
    "text": (str,),
    "a whole number": (int,),
    "a number": (int, float),
}
_NULLABLE = frozenset({"adapter", "limit"})

# Run ids name folders here and in the Hub run store, so they keep to make_run_id's
# alphabet, and nothing like "../x" can leave the runs folder.
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]*")


class NotAvailable(RuntimeError):
    """An option that a later story builds."""


@dataclass(frozen=True)
class EvalConfig:
    """The options of ``forge eval run``, one field each."""

    model: str
    adapter: str | None = None
    backend: Backend = "hf"
    device: Device = "cpu"
    split: Split = "val"
    reason: str | None = None
    limit: int | None = None
    prompt_version: str = "p1"
    n: int = 1
    temperature: float = 0.0
    retry: int = 0
    run_id: str | None = None
    store: Store = "local"

    @property
    def params(self) -> GenParams:
        return GenParams(temperature=self.temperature, n=self.n)

    @property
    def platform(self) -> Platform:
        """Where the run's model runs, as its manifest names it: a run on MPS is a
        ``local-mps`` pilot (ADR-0016)."""
        return "local-mps" if self.device == "mps" else "local-cpu"

    def manifest_config(self) -> dict[str, Any]:
        """What the manifest records and hashes: everything that can change a score.

        The reason is in the dev usage log, and the store says only where files go.
        """
        return {
            "backend": self.backend,
            "device": self.device,
            "model": self.model,
            "adapter": self.adapter,
            "split": self.split,
            "limit": self.limit,
            "prompt_version": self.prompt_version,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "generation": self.params.model_dump(),
            "retry": self.retry,
            "execution": {
                "timeout_ms": PRED_TIMEOUT_MS,
                "gold_timeout_ms": GOLD_TIMEOUT_MS,
                "max_rows": MAX_ROWS,
                "official_item_timeout_s": OFFICIAL_ITEM_TIMEOUT_S,
            },
        }


def check_config(config: EvalConfig) -> None:
    """Refuse options no run can take, before anything is loaded or written.

    Raises ``ValueError`` for a wrong option and ``NotAvailable`` for one that a later
    story builds.
    """
    if config.split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}, got {config.split!r}")
    if config.split == "dev" and not (config.reason or "").strip():
        raise ValueError(
            "--split dev needs --reason: every dev evaluation is logged in "
            "docs/dev-usage.md, and there are six at most (ADR-0004)"
        )
    if config.split != "dev" and config.reason is not None:
        raise ValueError("--reason goes with --split dev, the only split whose uses are logged")
    if config.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {config.backend!r}")
    if config.device not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}, got {config.device!r}")
    if config.device == "mps" and config.split == "dev":
        raise ValueError(
            "--device mps makes the run a local-mps pilot, and pilots never evaluate dev, "
            "so none spends one of the six dev looks (ADR-0016)"
        )
    if config.store not in STORES:
        raise ValueError(f"store must be one of {', '.join(STORES)}, got {config.store!r}")
    check_version(config.prompt_version)
    if config.limit is not None and config.limit < 1:
        raise ValueError(f"limit must be at least 1, got {config.limit}")
    if config.retry not in (0, 1):
        raise ValueError(f"retry must be 0 or 1, got {config.retry}")
    if config.run_id is not None:
        check_run_id(config.run_id)
    if config.backend == "fake":
        if config.model not in FAKE_MODELS:
            raise ValueError(
                f"--backend fake takes --model {' or '.join(FAKE_MODELS)}, got {config.model!r}"
            )
        if config.adapter is not None:
            raise ValueError("--backend fake loads no adapter")
        if config.device != "cpu":
            # Its manifest would call a run with no model a pilot on the laptop's GPU.
            raise ValueError(
                f"--backend fake runs no model, so it takes no --device {config.device}"
            )
    check_params(config.params)
    if config.backend in LATER_BACKENDS:
        raise NotAvailable(
            f"the {config.backend} backend is not available yet: it comes with "
            f"{LATER_BACKENDS[config.backend]}"
        )
    if config.store == "hub":
        raise NotAvailable("--store hub is available after FG-304 (Hub run store)")


def check_run_id(run_id: str) -> None:
    """Raise ``ValueError`` unless ``run_id`` is a plain folder name in the run id alphabet."""
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(
            f"run id {run_id!r} names a folder, so it takes lowercase letters, digits "
            "and '-', and starts with a letter or digit"
        )


def read_config_file(path: Path) -> dict[str, Any]:
    """The options the eval config file at ``path`` sets, by option name.

    Raises ``OSError`` when the file can't be read, and ``ValueError`` unless it is a YAML
    mapping of ``FILE_OPTIONS`` to values of their kinds. Whether a value is one its option
    takes, such as a known backend, is checked as for the command line's.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping of options, such as 'model: org/name'")
    unknown = sorted(str(name) for name in loaded if name not in FILE_OPTIONS)
    if unknown:
        raise ValueError(
            f"{path} sets {', '.join(unknown)}, which a config file can't: it takes "
            f"{', '.join(FILE_OPTIONS)}, and the run id, store and dev reason go on the "
            "command line"
        )
    for name, value in loaded.items():
        kind = FILE_OPTIONS[name]
        if value is None and name in _NULLABLE:
            continue
        # YAML reads true and false as booleans, which Python also counts as whole numbers.
        if isinstance(value, bool) or not isinstance(value, _KINDS[kind]):
            raise ValueError(f"{path}: {name} must be {kind}, got {value!r}")
    return dict(loaded)
