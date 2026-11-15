"""Stable hashing for configs and manifests."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from pathlib import PurePath
from typing import Any


def _encode(obj: Any) -> Any:
    """Encode the few non-JSON types configs actually hold; refuse anything else.

    A catch-all like ``default=str`` would hash sets or arbitrary objects by their
    repr, which can differ between processes and silently change a config hash.
    """
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set | frozenset):
        try:
            return sorted(obj)
        except TypeError as exc:
            raise TypeError(f"cannot hash a set of mixed types: {obj!r}") from exc
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        # Python mode, so the fields come back through these same branches: JSON mode
        # writes a UTC datetime with "Z" where isoformat gives "+00:00", and a model
        # would hash unlike its own model_dump().
        return model_dump()
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable for hashing")


def _distinct_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object read back, refusing keys such as 1 and "1" that JSON writes alike."""
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise TypeError(f"two keys are both written as {key!r} in JSON")
        obj[key] = value
    return obj


def canonical_json(obj: Any) -> bytes:
    """JSON with sorted keys and no whitespace, so equal configs hash equally.

    Keys sort as the strings JSON writes, so the JSON read back, with every key a
    string, gives the same bytes again.
    """
    # sort_keys compares keys as they are, putting 500 before 2000 but "2000" before
    # "500". Written once and read back, the keys are the strings any reader sees.
    plain = json.loads(
        json.dumps(obj, ensure_ascii=False, default=_encode), object_pairs_hook=_distinct_keys
    )
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
