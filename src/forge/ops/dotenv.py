"""Read ``KEY=value`` lines from the repo's ``.env`` without overriding the real environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        # An unterminated quote keeps the rest of the line, like most dotenv readers.
        return value[1:end] if end != -1 else value[1:]
    # Unquoted values end at an inline comment, which needs whitespace before the '#'.
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1] in " \t":
            return value[:i].rstrip()
    return value


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = _parse_value(value)
    return values


def merged_env(env: Mapping[str, str], dotenv: Path) -> dict[str, str]:
    """The process environment wins over ``.env`` so a shell export always takes effect."""
    merged = parse_dotenv(dotenv.read_text(encoding="utf-8")) if dotenv.is_file() else {}
    merged.update(env)
    return merged


def forge_env() -> dict[str, str]:
    """``os.environ`` plus the ``.env`` at the repo root (or cwd outside the repo)."""
    from forge.ops.paths import RepoRootNotFound, repo_root

    try:
        root = repo_root()
    except RepoRootNotFound:
        root = Path.cwd()
    return merged_env(os.environ, root / ".env")
