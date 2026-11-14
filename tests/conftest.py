"""Phase marker plugin.

Every test declares the plan phase that introduced it with ``@pytest.mark.phase(n)``.
``pytest --phase N`` runs phases 0..N and skips the rest, which is what a phase gate
runs. A test without the marker stops collection, so nothing slips past a gate
unlabeled.
"""

from __future__ import annotations

import pytest


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
