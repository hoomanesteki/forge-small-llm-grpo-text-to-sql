"""Check configs/models.yaml against the Hugging Face Hub, reading metadata only.

Usage: uv run python scripts/check_models.py [configs/models.yaml]

Each model the file gives a license for must still be on the Hub at its pinned commit,
with that license tag alone, not gated, and with the stated count of safetensors
parameters. It calls huggingface_hub's ``model_info`` without a token, since every model
Forge loads is public, and downloads no weights. It prints one line per model and exits 1
if any differs from the file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import yaml
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]


class ModelInfoApi(Protocol):
    def model_info(self, repo_id: str, *, revision: str | None = None) -> Any: ...


def licensed_models(models: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The file's models that state a license, by name: a top-level entry, such as
    ``pilot``, or one of a group's, such as ``candidates.qwen3-4b``."""
    found: dict[str, dict[str, Any]] = {}
    for name, value in models.items():
        entries = {name: value} if "repo" in value else {f"{name}.{k}": v for k, v in value.items()}
        found.update((key, entry) for key, entry in entries.items() if "license" in entry)
    return found


def problems(entry: Mapping[str, Any], api: ModelInfoApi) -> list[str]:
    """How the Hub's metadata for ``entry`` at its commit differs from what it states."""
    try:
        info = api.model_info(entry["repo"], revision=entry["revision"])
    except HfHubHTTPError as exc:
        return [f"the Hub has no {entry['repo']} at {entry['revision']}: {exc}"]
    found: list[str] = []
    if info.sha != entry["revision"]:
        found.append(f"the Hub's commit is {info.sha}, not {entry['revision']}")
    licenses = sorted(
        tag.removeprefix("license:") for tag in info.tags if tag.startswith("license:")
    )
    if licenses != [entry["license"]]:
        found.append(f"its license tags are {licenses}, not [{entry['license']!r}]")
    if bool(info.gated) != entry["gated"]:
        found.append(f"it is gated={info.gated!r}, not {entry['gated']}")
    params = info.safetensors.total if info.safetensors else None
    if params != entry["params"]:
        found.append(f"it has {params} safetensors parameters, not {entry['params']}")
    return found


def main(argv: list[str] | None = None, api: ModelInfoApi | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="?", type=Path, default=ROOT / "configs" / "models.yaml")
    args = parser.parse_args(argv)
    models = yaml.safe_load(args.models.read_text(encoding="utf-8"))
    hub = api or HfApi(token=False)
    failed = False
    for name, entry in licensed_models(models).items():
        found = problems(entry, hub)
        failed = failed or bool(found)
        gated = "gated" if entry["gated"] else "not gated"
        verdict = "; ".join(found) if found else f"ok, {entry['license']}, {gated}"
        print(f"{name}: {entry['repo']}@{entry['revision'][:7]}: {verdict}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
