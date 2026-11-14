from __future__ import annotations

import pytest
from typer.testing import CliRunner

from forge import __version__
from forge.cli import app


@pytest.mark.phase(0)
def test_version_cli() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"forge {__version__}"
