"""``forge doctor``: confirm the free accounts and the machine are ready for GPU stories.

Every check is a small function with its side effects injected, so tests can run the
whole doctor without network access. Output goes through :func:`redact`, which masks
the value of any environment variable that looks like a secret, plus any token the
checks read from a file.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.ops.kaggle import kaggle_bin

DEFAULT_RUN_STORE = "hoomanesteki/forge-runs"
SECRET_NAME = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD)", re.IGNORECASE)

# A runner gets the command and the environment to run it with.
Runner = Callable[[Sequence[str], Mapping[str, str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""
    required: bool = True


def _default_env() -> dict[str, str]:
    from forge.ops.dotenv import forge_env

    return forge_env()


@dataclass
class Context:
    """Everything the checks read from the outside world."""

    env: Mapping[str, str] = field(default_factory=_default_env)
    home: Path = field(default_factory=Path.home)
    cwd: Path = field(default_factory=Path.cwd)
    run: Runner | None = None
    hf_whoami: Callable[[str | None], dict[str, Any]] | None = None
    hf_repo_info: Callable[[str, str | None], Any] | None = None
    disk_free_gb: Callable[[Path], float] | None = None
    # Secret values found along the way that are not in env (e.g. read from a token file).
    secrets: set[str] = field(default_factory=set)


def _run(args: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), capture_output=True, text=True, timeout=60, check=False, env=dict(env)
    )


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _hf_whoami(token: str | None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    return dict(HfApi(token=token).whoami())


def _hf_repo_info(repo_id: str, token: str | None) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token).model_info(repo_id)


def redact(text: str, env: Mapping[str, str], extra: Sequence[str] = ()) -> str:
    values = [v for name, v in env.items() if SECRET_NAME.search(name)] + list(extra)
    # Longest first, so a secret that contains another is masked whole.
    for value in sorted({v for v in values if v and len(v) >= 6}, key=len, reverse=True):
        text = text.replace(value, "***")
    return text


def check_python() -> Check:
    version = platform.python_version()
    # The range pyproject.toml allows: 3.13 is the Kaggle image's, 3.12 the laptop's.
    ok = (3, 11) <= sys.version_info[:2] < (3, 14)
    return Check(
        "python", ok, version, "" if ok else "use Python 3.11 to 3.13 (uv python install 3.12)"
    )


def check_uv(ctx: Context) -> Check:
    run = ctx.run or _run
    try:
        out = run(["uv", "--version"], ctx.env)
    except FileNotFoundError:
        return Check("uv", False, "not installed", "install uv: https://docs.astral.sh/uv/")
    except (subprocess.TimeoutExpired, OSError) as exc:
        return Check("uv", False, f"could not run: {type(exc).__name__}", "check the uv install")
    ok = out.returncode == 0
    return Check("uv", ok, out.stdout.strip() or out.stderr.strip(), "" if ok else "reinstall uv")


def kaggle_credential_source(ctx: Context) -> str | None:
    """Where the Kaggle CLI would find credentials, without reading their values."""
    env = ctx.env
    if env.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN"
    for name in ("access_token", "access_token.txt"):
        if (ctx.home / ".kaggle" / name).is_file():
            return f"~/.kaggle/{name}"
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME/KAGGLE_KEY"
    config_dir = (
        Path(env["KAGGLE_CONFIG_DIR"]) if env.get("KAGGLE_CONFIG_DIR") else ctx.home / ".kaggle"
    )
    if (config_dir / "kaggle.json").is_file():
        return str(config_dir / "kaggle.json").replace(str(ctx.home), "~")
    return None


KAGGLE_HINT = (
    "phone-verify your Kaggle account, create an API token at kaggle.com/settings/api, "
    "and save it to ~/.kaggle/access_token (chmod 600); see docs/runbook.md#accounts"
)


def check_kaggle(ctx: Context) -> Check:
    source = kaggle_credential_source(ctx)
    if source is None:
        return Check("kaggle auth", False, "no credentials found", KAGGLE_HINT)
    run = ctx.run or _run
    try:
        # The cheapest authenticated call: list one of your own kernels. The merged env
        # matters: credentials that live only in .env must reach the CLI too.
        out = run([kaggle_bin(), "kernels", "list", "--mine", "--page-size", "1"], ctx.env)
    except FileNotFoundError:
        return Check(
            "kaggle auth", False, f"credentials in {source}, CLI missing", "run make setup"
        )
    except subprocess.TimeoutExpired:
        return Check("kaggle auth", False, f"{source}: kaggle CLI timed out", "check the network")
    except OSError as exc:
        return Check("kaggle auth", False, f"{source}: {type(exc).__name__}", "run make setup")
    if out.returncode != 0:
        reason = (out.stderr or out.stdout).strip().splitlines()[-1:] or ["unknown error"]
        return Check("kaggle auth", False, f"{source}: {reason[0][:120]}", KAGGLE_HINT)
    return Check("kaggle auth", True, f"authenticated via {source}")


def _hf_token(ctx: Context) -> str | None:
    if ctx.env.get("HF_TOKEN"):
        return ctx.env["HF_TOKEN"]
    token_file = ctx.home / ".cache" / "huggingface" / "token"
    if not token_file.is_file():
        return None
    token = token_file.read_text().strip()
    ctx.secrets.add(token)
    return token or None


HF_HINT = "create a write token at huggingface.co/settings/tokens and put HF_TOKEN in .env"


def check_hf(ctx: Context) -> tuple[Check, str | None]:
    token = _hf_token(ctx)
    if not token:
        return Check("hugging face auth", False, "no HF_TOKEN", HF_HINT), None
    try:
        user = (ctx.hf_whoami or _hf_whoami)(token)
    except Exception as exc:  # network or auth failure; the message never holds the token
        return Check(
            "hugging face auth", False, f"whoami failed: {type(exc).__name__}", HF_HINT
        ), None
    name = str(user.get("name", "?"))
    return Check("hugging face auth", True, f"logged in as {name}"), token


def check_run_store(ctx: Context, token: str | None) -> Check:
    repo_id = ctx.env.get("FORGE_RUN_STORE", DEFAULT_RUN_STORE)
    hint = f"create the private model repo {repo_id} on huggingface.co/new"
    if token is None:
        return Check("run store", False, f"{repo_id}: needs Hugging Face auth first", hint)
    try:
        info = (ctx.hf_repo_info or _hf_repo_info)(repo_id, token)
    except Exception as exc:
        return Check("run store", False, f"{repo_id}: {type(exc).__name__}", hint)
    if not getattr(info, "private", False):
        return Check("run store", False, f"{repo_id} is public", "make the run store repo private")
    return Check("run store", True, f"{repo_id} exists and is private")


def check_disk(ctx: Context) -> Check:
    need = float(ctx.env.get("FORGE_MIN_FREE_GB", "10"))
    free = (ctx.disk_free_gb or _disk_free_gb)(ctx.cwd)
    ok = free >= need
    hint = (
        ""
        if ok
        else f"free up {need - free:.1f} GB (Spider, models and caches need about {need:.0f} GB)"
    )
    return Check("disk", ok, f"{free:.1f} GB free (need {need:.0f})", hint)


def run_doctor(ctx: Context | None = None) -> list[Check]:
    ctx = ctx or Context()
    hf, token = check_hf(ctx)
    return [
        check_python(),
        check_uv(ctx),
        check_kaggle(ctx),
        hf,
        check_run_store(ctx, token),
        check_disk(ctx),
    ]


def render(checks: list[Check], env: Mapping[str, str], secrets: Sequence[str] = ()) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.required else "warn")
        lines.append(f"{mark}  {c.name:<{width}}  {c.detail}")
        if c.hint and not c.ok:
            lines.append(f"      {'':<{width}}  fix: {c.hint}")
    failed = sum(not c.ok and c.required for c in checks)
    lines.append("")
    lines.append("all checks passed" if not failed else f"{failed} required check(s) failed")
    return redact("\n".join(lines), env, secrets)


def exit_code(checks: list[Check]) -> int:
    return 1 if any(not c.ok and c.required for c in checks) else 0
