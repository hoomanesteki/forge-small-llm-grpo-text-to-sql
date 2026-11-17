"""Load the kernel scripts under kaggle/: standalone scripts rather than part of the package."""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

THROUGHPUT = Path(__file__).resolve().parents[1] / "kaggle" / "spike" / "throughput.py"
ENTRY = Path(__file__).resolve().parents[1] / "kaggle" / "entry.py"
NAME = "forge_spike_throughput"


def load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up while the class is built.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def throughput() -> ModuleType:
    return load_script(THROUGHPUT, NAME)


@functools.cache
def entry() -> ModuleType:
    return load_script(ENTRY, "forge_kaggle_entry")
