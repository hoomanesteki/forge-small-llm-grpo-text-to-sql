"""Load kaggle/spike/throughput.py, a standalone script rather than part of the package."""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

THROUGHPUT = Path(__file__).resolve().parents[1] / "kaggle" / "spike" / "throughput.py"
NAME = "forge_spike_throughput"


@functools.cache
def throughput() -> ModuleType:
    spec = importlib.util.spec_from_file_location(NAME, THROUGHPUT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up while the class is built.
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
    return module
