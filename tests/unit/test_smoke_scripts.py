"""The integration checkpoint scripts under ``scripts/smoke`` stay runnable."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SMOKE = Path(__file__).resolve().parents[2] / "scripts" / "smoke"


@pytest.mark.phase(1)
def test_phase_1_smoke_runs_the_checkpoint_steps() -> None:
    script = SMOKE / "phase_1.sh"

    assert os.access(script, os.X_OK), f"{script} is not executable"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    # Integration checkpoint 1 in docs/PLAN.md, in order.
    steps = [
        'uv run forge data fetch --from-zip "$zip"',
        "uv run forge data stats",
        "uv run forge eval gold --split val",
        'uv run forge sql run --db concert_singer "DELETE FROM singer"',
        "uv run forge eval agreement --mutations 500 --out",
    ]
    positions = [text.find(step) for step in steps]
    assert -1 not in positions, [s for s, p in zip(steps, positions, strict=True) if p == -1]
    assert positions == sorted(positions)
