"""A byte-level chat tokenizer and a random-weight Qwen2 model, built on the spot, so tests
run real transformers code without downloading anything."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tokenizers.pre_tokenizers import ByteLevel
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

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
    """A Qwen2 tokenizer with one token per byte and no merges, so any text encodes.

    Qwen2's class, because transformers loads that class for any qwen2 checkpoint, and it
    rebuilds a tokenizer of any other kind wrongly.
    """
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


def save_random_model(directory: Path, *, generation: dict[str, Any] | None = None) -> Path:
    """Save a two-layer Qwen2 model with random weights, and ``byte_tokenizer``, to
    ``directory``; ``generation`` is the generation config the checkpoint ships.

    The weights are drawn wide, so at every step the likeliest token leads the next by
    far more than float error, and a greedy reply doesn't depend on how a batch pads it.
    """
    tokenizer = byte_tokenizer()
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        initializer_range=0.3,
        tie_word_embeddings=False,
        bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(config)
    model.generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **(generation or {}),
    )
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory
