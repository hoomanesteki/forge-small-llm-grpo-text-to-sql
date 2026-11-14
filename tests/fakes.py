"""Offline stand-ins for the libraries tests must not reach over the network."""

from __future__ import annotations

from typing import Any


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
