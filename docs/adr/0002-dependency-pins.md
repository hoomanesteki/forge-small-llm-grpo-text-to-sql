# ADR-0002: Pin dependencies with uv and match the Kaggle image

- Status: accepted
- Date: 2026-11-14

## Context

The pipeline has to behave the same on a laptop, in CI and on a Kaggle GPU session.
The ML stack moves fast; an unpinned install months from now would not reproduce
today's numbers.

## Decision

- `uv.lock` pins every dependency. At the start of the project it resolved to the
  latest stable releases, among them torch 2.15.0, transformers 5.23.0, trl 1.18.0,
  peft 0.21.3, datasets 5.2.0, accelerate 1.15.0, pydantic 2.14.3, sqlglot 30.31.0
  and huggingface-hub 2.4.0.
- Local development uses Python 3.12 (`.python-version`). CI runs Python 3.11, 3.12
  and 3.13. Kaggle's GPU image moved to Python 3.13 (v173, October 2026), and the code
  dataset's wheel installs there only if `requires-python` allows it, so since FG-303
  it is `>=3.11,<3.14`. Every package CI installs has a 3.13 wheel; re-locking changed
  no version and added only `audioop-lts`, which the demo extra's gradio needs on 3.13.
- On Linux, torch comes from the PyTorch CPU index so CI never downloads CUDA wheels.
- GPU packages (unsloth, vllm, bitsandbytes) are not in the lock. They must match
  the CUDA build of torch preinstalled on Kaggle, so FG-301 pins them in
  `kaggle/requirements-gpu.txt`, tested on that image.
- vLLM brings its own torch. Since FG-306 that file pins vllm 0.33.0, which pins torch
  2.14.0 (built for CUDA 13.0, the image's CUDA) and needs transformers below 5.21, so GPU
  jobs run torch 2.14.0 and transformers 5.20.2 where CI runs 2.15.0 and 5.23.0. Every
  other pin there is the lock's version. The transformers calls Forge makes in a kernel
  were checked against the 5.20.2 source: they behave the same for Forge's arguments.
- The llama.cpp commit used for GGUF export is pinned in FG-702 and added here.

## Consequences

- `uv sync --locked` fails if `pyproject.toml` and `uv.lock` drift apart, in CI too.
- Upgrades are deliberate: bump, re-lock, run `make gate`, and note it here.

## Revisit if

The Kaggle image changes its Python or CUDA version.
