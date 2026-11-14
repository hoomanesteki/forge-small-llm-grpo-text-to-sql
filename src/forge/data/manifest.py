"""Hash of the exact data a run used, recorded in every run manifest."""

from __future__ import annotations

from pathlib import Path

from forge.core.hashing import canonical_json, sha256_hex
from forge.data.spider import file_sha256, load_raw, raw_files, resolve_data_dir


def data_manifest_hash(data_dir: Path | None = None) -> str:
    """sha256 over the example ids of each split and the sha256 of every raw file.

    The ids pin which questions each split holds and in what order; the file hashes pin
    their text and the databases' contents. The directory's own path is left out, so a
    laptop and a Kaggle kernel with the same files get the same hash.
    """
    root = resolve_data_dir(data_dir)
    split_ids: dict[str, list[str]] = {}
    for example in load_raw(root):
        split_ids.setdefault(example.split, []).append(example.id)
    files = {rel: file_sha256(root / rel) for rel in raw_files(root)}
    return sha256_hex(canonical_json({"split_ids": split_ids, "files": files}))
