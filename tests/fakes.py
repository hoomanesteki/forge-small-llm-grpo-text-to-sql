"""Stand-ins for what tests can't use for real: libraries that would reach the network,
and worker processes that a test double in this process never reaches."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge.sql.executor import SqliteExecutor


class WordTokenizer:
    """A Hugging Face tokenizer, as far as Forge uses one: one token per word of a chat
    template that tags each message with its role."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, list[int]]:
        self.calls.append((conversation, kwargs))
        text = " ".join(f"<|{m['role']}|> {m['content']}" for m in conversation)
        if kwargs.get("add_generation_prompt"):
            text += " <|assistant|>"
        return {"input_ids": list(range(len(text.split())))}


class InlineSandbox:
    """A ``SandboxPool`` that runs each job in this process, so monkeypatched code runs.

    It has no deadline: tests that need the kill use a real pool.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.executor = SqliteExecutor(self.data_dir)

    def call(self, fn: Callable[..., Any], *args: Any, timeout_s: float) -> Any:
        return fn(self.executor, *args)
