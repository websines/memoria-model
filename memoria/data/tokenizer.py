"""Tokenizer setup.

Default: Llama 3 tokenizer (128K vocab, good at code + multilingual).
GPT-2 tokenizer is from 2019 — bad at code, small vocab, wastes tokens.

Fallback chain: Llama 3 → Mistral → GPT-2 (if nothing else available).
"""

from transformers import AutoTokenizer


# Preferred tokenizers in order. First available wins.
# Qwen3 is the best current tokenizer for code + multilingual (2026):
# 151,936 vocab, byte-level BPE, trained on 36T tokens across 119 languages.
TOKENIZER_PREFERENCES = [
    "Qwen/Qwen3-0.6B",                   # 151K vocab, byte-level BPE, best for code+multilingual (2026)
    "meta-llama/Meta-Llama-3-8B",         # 128K vocab, excellent code/multilingual
    "mistralai/Mistral-7B-v0.1",          # 32K vocab, good code support
    "gpt2",                                # 50K vocab, fallback only
]


def get_tokenizer(name: str | None = None, vocab_size: int | None = None):
    """Load a tokenizer.

    Args:
        name: explicit HuggingFace tokenizer name. If None, tries TOKENIZER_PREFERENCES in order.
        vocab_size: if set, verify tokenizer vocab fits config (warning, not assert)

    Returns:
        tokenizer object
    """
    if name is not None:
        return _load_tokenizer(name, vocab_size)

    # Try preferred tokenizers in order
    for candidate in TOKENIZER_PREFERENCES:
        try:
            return _load_tokenizer(candidate, vocab_size)
        except Exception:
            continue

    raise RuntimeError("No tokenizer available. Install transformers and authenticate with HF if needed.")


def _load_tokenizer(name: str, vocab_size: int | None):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if vocab_size is not None and len(tokenizer) > vocab_size:
        print(f"WARNING: Tokenizer {name} vocab ({len(tokenizer)}) exceeds config vocab_size ({vocab_size}). "
              f"Update TransformerConfig.vocab_size to match.")

    return tokenizer
