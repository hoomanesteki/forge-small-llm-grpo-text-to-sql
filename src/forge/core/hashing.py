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
        return model_dump(mode="json")
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable for hashing")


def canonical_json(obj: Any) -> bytes:
    """JSON with sorted keys and no whitespace, so equal configs hash equally."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_encode
    ).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
