from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from forge.data.manifest import data_manifest_hash

# data_manifest_hash of tests/fixtures/spider_mini. CI on Linux and a Mac must agree on
# it; it changes only with the fixture files, the ids or the hash recipe.
SPIDER_MINI_HASH = "2549a64c6001fdb900f1156dbb6f1359d57692111910bc1ab98609f007516dd9"


@pytest.mark.phase(1)
def test_data_manifest_hash_is_pinned_for_the_fixture(spider_mini: Path) -> None:
    assert data_manifest_hash() == SPIDER_MINI_HASH
    assert data_manifest_hash(spider_mini) == SPIDER_MINI_HASH


@pytest.mark.phase(1)
def test_data_manifest_hash_tracks_ids_and_file_bytes(spider_mini: Path, tmp_path: Path) -> None:
    copy = Path(shutil.copytree(spider_mini, tmp_path / "copy"))
    # The directory's path is not part of the hash.
    assert data_manifest_hash(copy) == SPIDER_MINI_HASH

    db = copy / "database" / "geo" / "geo.sqlite"
    original = db.read_bytes()
    db.write_bytes(original + b"\x00")
    assert re.fullmatch(r"[0-9a-f]{64}", changed := data_manifest_hash(copy))
    assert changed != SPIDER_MINI_HASH
    db.write_bytes(original)

    # Same files except one question fewer: the ids and the JSON bytes both change.
    others = copy / "train_others.json"
    items = json.loads(others.read_text(encoding="utf-8"))
    others.write_bytes(json.dumps(items[:-1], indent=4).encode())
    assert data_manifest_hash(copy) not in {SPIDER_MINI_HASH, changed}
