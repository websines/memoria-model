"""Interleave multiple data streams with configurable weights.

Mixes FineWeb-Edu (70%), Stack v2 (20%), and synthetic tasks (10%)
into a single training stream.
"""

import random
import torch
from torch import Tensor
from typing import Iterator
from transformers import PreTrainedTokenizer

from .streaming import stream_fineweb_edu, stream_code
from .synthetic import generate_all_synthetic


def interleaved_stream(
    tokenizer: PreTrainedTokenizer,
    seq_len: int = 2048,
    weights: tuple[float, float, float] = (0.7, 0.2, 0.1),
    synthetic_data: list[str] | None = None,
    stack_languages: list[str] | None = None,
    skip_documents: int = 0,
) -> Iterator[dict[str, Tensor]]:
    """Interleave multiple data sources by weight.

    Args:
        tokenizer: tokenizer for encoding
        seq_len: sequence length
        weights: (fineweb_weight, stack_weight, synthetic_weight)
        synthetic_data: pre-generated synthetic sequences (if None, generates on the fly)
        stack_languages: language filter for Stack v2
        skip_documents: number of raw documents to skip in the primary FineWeb
            stream for fast resume. Uses HF native skip() — O(1) not O(N).

    Yields:
        dict with 'input_ids' [seq_len] and 'labels' [seq_len] and 'source' str
    """
    w_fineweb, w_stack, w_synthetic = weights
    total = w_fineweb + w_stack + w_synthetic

    # Normalize
    w_fineweb /= total
    w_stack /= total
    w_synthetic /= total

    # Initialize streams
    fineweb_iter = stream_fineweb_edu(tokenizer, seq_len, skip_documents=skip_documents) if w_fineweb > 0 else iter([])

    # Code dataset: starcoderdata (ungated) → fallback to Stack v2
    code_iter = iter([])
    if w_stack > 0:
        try:
            code_iter = stream_code(tokenizer, seq_len, languages=stack_languages)
        except Exception as e:
            print(f"WARNING: Code dataset unavailable ({e}). Redistributing {w_stack:.0%} weight to FineWeb.")
            print(f"  Data mix changed: FineWeb {w_fineweb:.0%} → {w_fineweb + w_stack:.0%}")
            w_fineweb += w_stack
            w_stack = 0

    # Synthetic: tokenize from pre-generated list
    if synthetic_data is None:
        synthetic_data = generate_all_synthetic()
    synthetic_iter = _synthetic_stream(synthetic_data, tokenizer, seq_len) if w_synthetic > 0 else iter([])

    while True:
        r = random.random()

        if r < w_fineweb:
            try:
                batch = next(fineweb_iter)
                batch['source'] = 'fineweb'
                yield batch
            except StopIteration:
                fineweb_iter = stream_fineweb_edu(tokenizer, seq_len)
                continue

        elif r < w_fineweb + w_stack:
            try:
                batch = next(code_iter)
                batch['source'] = 'code'
                yield batch
            except StopIteration:
                try:
                    code_iter = stream_code(tokenizer, seq_len, languages=stack_languages)
                except Exception as e:
                    print(f"WARNING: Code stream restart failed ({e}). Redistributing weight to FineWeb.")
                    w_fineweb += w_stack
                    w_stack = 0
                continue
            except Exception as e:
                print(f"WARNING: Code stream error ({e}). Redistributing weight to FineWeb.")
                w_fineweb += w_stack
                w_stack = 0
                continue

        else:
            try:
                batch = next(synthetic_iter)
                batch['source'] = 'synthetic'
                yield batch
            except StopIteration:
                # Reshuffle and restart synthetic
                random.shuffle(synthetic_data)
                synthetic_iter = _synthetic_stream(synthetic_data, tokenizer, seq_len)
                continue


def _synthetic_stream(
    data: list[str],
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
) -> Iterator[dict[str, Tensor]]:
    """Tokenize synthetic text data into fixed-length sequences."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.vocab_size - 1
    buffer = []

    for text in data:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_id)
        buffer.extend(tokens)

        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len:]

            yield {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }
