"""A byte-level chat tokenizer, built on the spot, so tests run real transformers code
without downloading anything."""

from __future__ import annotations

from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer

# The turns of Qwen's chat templates. CHATML opens the assistant's turn as Qwen2.5 does,
# with no thinking flag; QWEN3 ends as Qwen3's template does, which reads one.
_TURNS = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
)
CHATML = _TURNS + "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
QWEN3 = (
    _TURNS + "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
)


def byte_tokenizer(chat_template: str = CHATML) -> Qwen2Tokenizer:
    """A Qwen2 tokenizer with one token per byte and no merges, so any text encodes."""
    vocab = {char: i for i, char in enumerate(sorted(ByteLevel.alphabet()))}
    for token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        vocab[token] = len(vocab)
    tokenizer = Qwen2Tokenizer(
        vocab=vocab,
        merges=[],
        unk_token=None,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        extra_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = chat_template
    return tokenizer
