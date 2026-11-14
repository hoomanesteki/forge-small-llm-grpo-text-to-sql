"""Forge: post-train a small open LLM with GRPO on verifiable text-to-SQL rewards."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("forge")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = ["__version__"]
