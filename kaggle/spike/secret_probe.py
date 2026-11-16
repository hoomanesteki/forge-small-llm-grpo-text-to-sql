"""Why a Kaggle Secret can't be read in a pushed run. Prints names and error text, never a value.

FG-301 ran this once, on 2026-11-16 (UTC), after the throughput spike's ``secrets`` stage
found ``HF_TOKEN`` unreadable with ``ConnectionError``. It went up by hand as version 3 of
the spike's own kernel, ``hoomanesteki/forge-throughput-spike``, so that the ``HF_TOKEN``
binding attached in the Kaggle editor would apply if it had survived the CLI push: ``kaggle
kernels push -p <folder>``, the folder holding this script as ``probe.py`` and a
``kernel-metadata.json`` with that id, ``"code_file": "probe.py"`` and ``"enable_gpu":
false``, so the session spent no GPU quota. ``kaggle kernels output`` brought back
``secret-probe.json``, committed as ``docs/spikes/kaggle-gpu/secret-probe.json``.

It asks for ``HF_TOKEN`` and for a label no secret has, so the two answers can be compared,
and records for each only whether a value came back and its length, or the error: its
type, message, cause and the end of its traceback, with anything shaped like a Hugging Face
token cut out. The code is the script that ran, with type hints and ``main`` added so a
test can run it against a fake client; nothing imports it, and ``scripts/kaggle_spike.py``
bundles ``throughput.py`` alone.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

OUT = Path("/kaggle/working/secret-probe.json")
LABELS = ("HF_TOKEN", "FORGE_NO_SUCH_LABEL")


def scrub(text: str) -> str:
    return re.sub(r"hf_[A-Za-z0-9]{8,}", "hf_<redacted>", text)


def probe() -> dict[str, Any]:
    out: dict[str, Any] = {"run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")}
    out["kaggle_env_names"] = sorted(k for k in os.environ if k.startswith(("KAGGLE", "KERNEL")))
    out["has_user_secrets_token"] = bool(os.environ.get("KAGGLE_USER_SECRETS_TOKEN"))
    out["url_base"] = os.environ.get("KAGGLE_URL_BASE")
    try:
        import kaggle_secrets

        out["client_file"] = kaggle_secrets.__file__
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        for label in LABELS:
            try:
                value = client.get_secret(label)
                out[label] = {"readable": bool(value), "length": len(value or "")}
                del value
            # Every failure is the finding here, recorded with its cause.
            except Exception as exc:
                out[label] = {
                    "error": type(exc).__name__,
                    "module": type(exc).__module__,
                    "message": scrub(str(exc))[:600],
                    "cause": scrub(repr(exc.__cause__ or exc.__context__))[:600],
                    "trace_tail": scrub("".join(traceback.format_exception(exc))[-1200:]),
                }
    except Exception as exc:
        out["client"] = f"{type(exc).__name__}: {scrub(str(exc))[:300]}"
    return out


def main(out_file: Path = OUT) -> None:
    out = probe()
    print(json.dumps(out, indent=1))
    with open(out_file, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
