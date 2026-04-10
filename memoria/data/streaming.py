"""Streaming data loaders for HuggingFace datasets.

All large datasets streamed — no local disk footprint.
FineWeb-Edu (10BT) + The Stack v2 dedup streamed from HF servers.

Supports two encoding modes:
- Token mode: BPE tokenizer → token IDs (for standard LM head)
- Byte mode: UTF-8 encoding → byte IDs 0-259 (for BLT byte-level I/O)

Reference: huggingface.co/docs/datasets/stream
"""

import torch
from torch import Tensor
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Iterator


# ── Byte encoding constants (matching BLT's 260-class byte vocab) ──
# Bytes 0-255: raw byte values
# 256-259: special tokens
BOS_BYTE = 256
EOS_BYTE = 257
PAD_BYTE = 258
SEP_BYTE = 259


def text_to_bytes(text: str) -> list[int]:
    """Encode text as UTF-8 byte values (0-255).

    Invalid/surrogate characters are replaced with U+FFFD (3 bytes: EF BF BD).
    """
    return list(text.encode('utf-8', errors='replace'))


def stream_fineweb_edu(
    tokenizer: PreTrainedTokenizer | None,
    seq_len: int = 2048,
    split: str = "train",
    skip_documents: int = 0,
    byte_mode: bool = False,
) -> Iterator[dict[str, Tensor]]:
    """Stream sequences from FineWeb-Edu 10BT sample.

    Args:
        tokenizer: tokenizer for encoding (ignored in byte_mode)
        seq_len: sequence length (tokens or bytes depending on mode)
        split: dataset split
        skip_documents: number of raw documents to skip (for fast resume).
            Uses HF datasets' native skip() which seeks without iterating.
        byte_mode: if True, encode as UTF-8 bytes instead of BPE tokens

    Yields:
        dict with 'input_ids' [seq_len] and 'labels' [seq_len] tensors
    """
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split=split,
        streaming=True,
    )
    if skip_documents > 0:
        ds = ds.skip(skip_documents)

    if byte_mode:
        yield from _byte_stream(ds, seq_len, text_key="text")
    else:
        yield from _tokenize_stream(ds, tokenizer, seq_len, text_key="text")


def stream_code(
    tokenizer: PreTrainedTokenizer | None,
    seq_len: int = 2048,
    languages: list[str] | None = None,
    split: str = "train",
    byte_mode: bool = False,
) -> Iterator[dict[str, Tensor]]:
    """Stream code from starcoderdata (ungated, 250B tokens, 86 languages).

    Falls back to The Stack v2 dedup if starcoderdata fails.

    Args:
        tokenizer: tokenizer to use (ignored in byte_mode)
        seq_len: sequence length
        languages: filter to specific languages (e.g., ["python", "javascript"])
        split: dataset split
        byte_mode: if True, encode as UTF-8 bytes instead of BPE tokens

    Yields:
        dict with 'input_ids' and 'labels' tensors
    """
    encode_fn = _byte_stream if byte_mode else _tokenize_stream

    # Try starcoderdata first (ungated, always accessible)
    for dataset_name, text_key in [
        ("bigcode/starcoderdata", "content"),
        ("bigcode/the-stack-v2-dedup", "content"),
    ]:
        try:
            if languages and dataset_name == "bigcode/starcoderdata":
                # starcoderdata uses data_dir for language filtering
                for lang in languages:
                    try:
                        ds = load_dataset(
                            dataset_name,
                            data_dir=lang,
                            split=split,
                            streaming=True,
                        )
                        if byte_mode:
                            yield from _byte_stream(ds, seq_len, text_key=text_key)
                        else:
                            yield from _tokenize_stream(ds, tokenizer, seq_len, text_key=text_key)
                    except Exception:
                        continue
                return
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True)
                if languages:
                    lang_set = set(l.lower() for l in languages)
                    ds = ds.filter(lambda x: x.get("lang", "").lower() in lang_set)
                if byte_mode:
                    yield from _byte_stream(ds, seq_len, text_key=text_key)
                else:
                    yield from _tokenize_stream(ds, tokenizer, seq_len, text_key=text_key)
                return
        except Exception as e:
            print(f"WARNING: {dataset_name} failed ({e}), trying next...")
            continue

    raise RuntimeError("No code dataset available")


def _byte_stream(
    dataset,
    seq_len: int,
    text_key: str = "text",
) -> Iterator[dict[str, Tensor]]:
    """Encode a streaming dataset as packed byte sequences.

    Text → UTF-8 bytes (0-255). Documents separated by EOS_BYTE (257).
    Packed into fixed-length sequences with no padding waste.

    Yields:
        dict with 'input_ids' [seq_len] and 'labels' [seq_len],
        values in range 0-259 (byte vocab).
    """
    buffer: list[int] = []

    for example in dataset:
        text = example.get(text_key, "")
        if not text:
            continue

        byte_ids = text_to_bytes(text)
        byte_ids.append(EOS_BYTE)
        buffer.extend(byte_ids)

        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len:]

            yield {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }


def _tokenize_stream(
    dataset,
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
    text_key: str = "text",
) -> Iterator[dict[str, Tensor]]:
    """Tokenize a streaming dataset into fixed-length sequences.

    Packs multiple documents into sequences (no padding waste).
    Documents separated by EOS token.

    Yields:
        dict with 'input_ids' [seq_len] and 'labels' [seq_len]
    """
    buffer = []
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.vocab_size - 1

    for example in dataset:
        text = example.get(text_key, "")
        if not text:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(eos_id)  # document boundary
        buffer.extend(tokens)

        # Yield full sequences from buffer
        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len:]  # overlap by 1 for labels

            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)

            yield {"input_ids": input_ids, "labels": labels}
