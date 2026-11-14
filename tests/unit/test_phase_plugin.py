"""The phase plugin is what a gate relies on, so it is tested against a real pytest run."""

from __future__ import annotations

from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


@pytest.fixture
def phased(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(CONFTEST.read_text())
    return pytester


@pytest.mark.phase(0)
def test_missing_marker_is_error(phased: pytest.Pytester) -> None:
    phased.makepyfile(
        """
        import pytest

        @pytest.mark.phase(0)
        def test_marked():
            pass

        def test_unmarked():
            pass
        """
    )
    result = phased.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*without @pytest.mark.phase*", "*test_unmarked*"])


@pytest.mark.phase(0)
def test_later_phase_skipped(phased: pytest.Pytester) -> None:
    phased.makepyfile(
        """
        import pytest

        @pytest.mark.phase(0)
        def test_now():
            pass

        @pytest.mark.phase(1)
        def test_later():
            pass
        """
    )
    result = phased.runpytest("--phase", "0", "-rs")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*phase 1 is after --phase 0*"])


@pytest.mark.phase(0)
def test_no_phase_flag_runs_everything(phased: pytest.Pytester) -> None:
    phased.makepyfile(
        """
        import pytest

        @pytest.mark.phase(3)
        def test_any():
            pass
        """
    )
    phased.runpytest().assert_outcomes(passed=1)
