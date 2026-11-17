"""Bundle the FG-301 spike as two Kaggle kernels, ready for `kaggle kernels push`.

Usage: uv run python scripts/kaggle_spike.py --owner <kaggle-username> [--out build/kaggle-spike]

Each kernel folder gets a checked kernel-metadata.json and kaggle/spike/throughput.py
with its arguments written in, since Kaggle starts a script kernel with none:

- gpu: every measuring stage on a T4 with internet, bounded by the push timeout.
- limit: the session-limit probe as a CPU kernel, so waiting for the kill spends no GPU
  quota.

It pushes nothing. It prints the commands that push each kernel, check on it and fetch
its output; docs/spikes/kaggle-gpu.md says what to do with the results.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path

from forge.ops.kaggle import (
    GPU_MACHINE_SHAPE,
    bake_argv,
    load_template,
    render_kernel_metadata,
    write_kernel_dir,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kaggle" / "spike" / "throughput.py"


@dataclass(frozen=True)
class Kernel:
    slug: str
    title: str
    gpu: bool
    argv: tuple[str, ...]
    timeout_s: int  # passed to `kaggle kernels push -t`


KERNELS = {
    # 90 minutes: the story's 1.5 GPU hours. The script starts no stage after 80, so the
    # report is written before the session is cut.
    "gpu": Kernel(
        "forge-throughput-spike", "Forge throughput spike", True, ("run",), timeout_s=5400
    ),
    # A beat every 30 s against a 15-minute timeout: the last beat kept says how much of
    # the output survives a session that is killed at its limit.
    "limit": Kernel(
        "forge-session-limit-probe",
        "Forge session limit probe",
        False,
        ("limit-probe", "--interval-seconds", "30"),
        timeout_s=900,
    ),
}


def bundle(owner: str, out: Path) -> list[str]:
    """Write a folder per kernel under ``out``; return the commands to run them."""
    template = load_template(ROOT)
    source = SCRIPT.read_text(encoding="utf-8")
    commands = []
    for name, kernel in KERNELS.items():
        values = {"owner": owner, "slug": kernel.slug, "title": kernel.title}
        meta = render_kernel_metadata(
            template, {**values, "code_file": SCRIPT.name}, gpu=kernel.gpu
        )
        folder = write_kernel_dir(out / name, meta, bake_argv(source, kernel.argv))
        accelerator = ["--accelerator", GPU_MACHINE_SHAPE] if kernel.gpu else []
        push = ["kaggle", "kernels", "push", "-p", str(folder), *accelerator]
        commands += [
            shlex.join([*push, "-t", str(kernel.timeout_s)]),
            shlex.join(["kaggle", "kernels", "status", meta["id"]]),
            shlex.join(["kaggle", "kernels", "output", meta["id"], "-p", f"{folder}-output", "-o"]),
        ]
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--owner", required=True, help="the Kaggle username that owns the kernels")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "kaggle-spike")
    args = parser.parse_args()
    commands = bundle(args.owner, args.out)
    print("Check the GPU quota first: kaggle quota -v")
    print("\n".join(commands))


if __name__ == "__main__":
    main()
