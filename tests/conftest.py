"""Phase marker plugin and shared Spider data fixtures.

Every test declares the plan phase that introduced it with ``@pytest.mark.phase(n)``.
``pytest --phase N`` runs phases 0..N and skips the rest, which is what a phase gate
runs. A test without the marker stops collection, so nothing slips past a gate
unlabeled.

Tests marked ``requires_full_data`` need the full Spider download, which CI doesn't
have; they are skipped, with the reason, wherever it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SPIDER_MINI = Path(__file__).resolve().parent / "fixtures" / "spider_mini"


@pytest.fixture
def spider_mini(monkeypatch: pytest.MonkeyPatch) -> Path:
    """The committed Spider subset, with FORGE_DATA_DIR pointing at it."""
    monkeypatch.setenv("FORGE_DATA_DIR", str(SPIDER_MINI))
    return SPIDER_MINI


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--phase",
        action="store",
        type=int,
        default=None,
        help="Run tests from phases 0..N only; later phases are skipped.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "phase(n): plan phase that introduced the test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    max_phase = config.getoption("--phase")
    unmarked: list[str] = []
    for item in items:
        marker = item.get_closest_marker("phase")
        if marker is None or len(marker.args) != 1 or not isinstance(marker.args[0], int):
            unmarked.append(item.nodeid)
            continue
        test_phase = marker.args[0]
        if max_phase is not None and test_phase > max_phase:
            reason = f"phase {test_phase} is after --phase {max_phase}"
            item.add_marker(pytest.mark.skip(reason=reason))
    if unmarked:
        listing = "\n  ".join(unmarked)
        raise pytest.UsageError(f"tests without @pytest.mark.phase(n):\n  {listing}")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("requires_full_data") is None:
        return
    # Only full-data tests need the loader, so collecting the others never imports it.
    from forge.data.spider import load_checksums, resolve_data_dir

    root = resolve_data_dir()
    missing = [rel for rel in load_checksums() if not (root / rel).is_file()]
    if missing:
        pytest.skip(
            f"needs the full Spider data, but {len(missing)} of its files are missing from "
            f"{root}; run `uv run forge data fetch` or set FORGE_DATA_DIR"
        )
