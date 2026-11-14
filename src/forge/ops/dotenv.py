"""Read ``KEY=value`` lines from a local ``.env`` file without overriding the real environment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def merged_env(env: Mapping[str, str], dotenv: Path) -> dict[str, str]:
    """The process environment wins over ``.env`` so a shell export always takes effect."""
    merged = parse_dotenv(dotenv.read_text(encoding="utf-8")) if dotenv.is_file() else {}
    merged.update(env)
    return merged
