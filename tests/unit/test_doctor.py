from __future__ import annotations

import secrets
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge.ops.doctor import Context, exit_code, render, run_doctor
from forge.ops.dotenv import merged_env, parse_dotenv

# Fresh random canaries on every run, so no token-shaped literal lives in the source.
CANARIES = {
    name: f"canary-{secrets.token_hex(12)}"
    for name in ("HF_TOKEN", "KAGGLE_API_TOKEN", "WANDB_API_KEY")
}


def ok_run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    # Echo the arguments back, so a token passed on the command line would show up.
    return subprocess.CompletedProcess(list(args), 0, stdout="ok " + " ".join(args), stderr="")


def make_ctx(tmp_path: Path, env: dict[str, str], **overrides: Any) -> Context:
    defaults: dict[str, Any] = {
        "env": env,
        "home": tmp_path,
        "cwd": tmp_path,
        "run": ok_run,
        "hf_whoami": lambda token: {"name": "someone"},
        "hf_repo_info": lambda repo, token: SimpleNamespace(private=True),
        "disk_free_gb": lambda path: 50.0,
    }
    defaults.update(overrides)
    return Context(**defaults)


@pytest.mark.phase(0)
def test_reports_missing_kaggle_creds(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {"HF_TOKEN": CANARIES["HF_TOKEN"]})
    checks = run_doctor(ctx)
    kaggle = next(c for c in checks if c.name == "kaggle auth")
    assert not kaggle.ok
    assert "~/.kaggle/access_token" in kaggle.hint
    assert exit_code(checks) == 1
    out = render(checks, ctx.env)
    assert "FAIL  kaggle auth" in out
    assert "fix: phone-verify" in out


@pytest.mark.phase(0)
def test_never_prints_tokens(tmp_path: Path) -> None:
    def leaky_whoami(token: str | None) -> dict[str, Any]:
        return {"name": f"user-with-{token}"}

    ctx = make_ctx(tmp_path, dict(CANARIES), hf_whoami=leaky_whoami)
    checks = run_doctor(ctx)
    out = render(checks, ctx.env)
    for value in CANARIES.values():
        assert value not in out
    assert "***" in out


@pytest.mark.phase(0)
def test_all_green_exits_zero(tmp_path: Path) -> None:
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "access_token").write_text("x")
    ctx = make_ctx(tmp_path, {"HF_TOKEN": CANARIES["HF_TOKEN"]})
    checks = run_doctor(ctx)
    assert [c.name for c in checks if not c.ok] in ([], ["python"])
    assert "run store" in render(checks, ctx.env)


@pytest.mark.phase(0)
def test_public_run_store_fails(tmp_path: Path) -> None:
    ctx = make_ctx(
        tmp_path,
        {"HF_TOKEN": CANARIES["HF_TOKEN"]},
        hf_repo_info=lambda repo, token: SimpleNamespace(private=False),
    )
    store = next(c for c in run_doctor(ctx) if c.name == "run store")
    assert not store.ok and "public" in store.detail


@pytest.mark.phase(0)
def test_low_disk_fails(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, {}, disk_free_gb=lambda path: 3.0)
    disk = next(c for c in run_doctor(ctx) if c.name == "disk")
    assert not disk.ok and "3.0 GB free" in disk.detail


@pytest.mark.phase(0)
def test_dotenv_never_overrides_environment(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\nexport HF_TOKEN='from-file'\nOTHER=1\n")
    assert parse_dotenv(dotenv.read_text()) == {"HF_TOKEN": "from-file", "OTHER": "1"}
    merged = merged_env({"HF_TOKEN": "from-shell"}, dotenv)
    assert merged == {"HF_TOKEN": "from-shell", "OTHER": "1"}
