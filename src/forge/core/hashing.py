"""Stable hashing for configs and manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """JSON with sorted keys and no whitespace, so equal configs hash equally."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
