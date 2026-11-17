"""scripts/check_models.py, with the Hub's model metadata faked."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from check_models import licensed_models, main
from huggingface_hub.errors import RevisionNotFoundError

from tests.fakes import hub_error

ROOT = Path(__file__).resolve().parents[2]
REVISION = "".join("0123456789abcdef"[(i * 5) % 16] for i in range(40))
MODELS = {
    "tiny": {"repo": "org/tiny", "revision": REVISION},
    "candidates": {
        "small": {
            "repo": "org/small",
            "revision": REVISION,
            "license": "apache-2.0",
            "gated": False,
            "params": 1_500_000_000,
        },
    },
}


class FakeApi:
    """``HfApi.model_info`` answering from ``infos``, by repo, and recording each call."""

    def __init__(self, **infos: Any) -> None:
        self.infos = infos
        self.calls: list[tuple[str, str | None]] = []

    def model_info(self, repo_id: str, *, revision: str | None = None) -> Any:
        self.calls.append((repo_id, revision))
        info = self.infos[repo_id]
        if isinstance(info, Exception):
            raise info
        return info


def _info(
    sha: str = REVISION, tags: tuple[str, ...] = ("license:apache-2.0",), **changes: Any
) -> SimpleNamespace:
    fields: dict[str, Any] = {"gated": False, "safetensors": SimpleNamespace(total=1_500_000_000)}
    return SimpleNamespace(sha=sha, tags=["text-generation", *tags], **{**fields, **changes})


@pytest.mark.phase(3)
def test_models_are_checked_against_the_hub_at_their_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every model the committed file gives a license for is checked; the tiny one has none.
    committed = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    names = set(licensed_models(committed))
    assert {"pilot", "reference.qwen3-8b", "candidates.qwen3-4b"} <= names
    assert "tiny" not in names

    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(MODELS), encoding="utf-8")
    api = FakeApi(**{"org/small": _info()})
    assert main([str(path)], api=api) == 0
    assert api.calls == [("org/small", REVISION)]
    assert capsys.readouterr().out == (
        f"candidates.small: org/small@{REVISION[:7]}: ok, apache-2.0, not gated\n"
    )

    for info, problem in (
        (_info(sha=REVISION[::-1]), f"the Hub's commit is {REVISION[::-1]}, not {REVISION}"),
        (_info(tags=("license:other",)), "its license tags are ['other'], not ['apache-2.0']"),
        (_info(tags=()), "its license tags are [], not ['apache-2.0']"),
        (_info(gated="manual"), "it is gated='manual', not False"),
        (_info(safetensors=None), "it has None safetensors parameters, not 1500000000"),
        (
            hub_error(404, RevisionNotFoundError),
            f"the Hub has no org/small at {REVISION}: 404 from the Hub",
        ),
    ):
        assert main([str(path)], api=FakeApi(**{"org/small": info})) == 1
        assert problem in capsys.readouterr().out
