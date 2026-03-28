"""Streaming data loaders for HuggingFace datasets.

All large datasets streamed — no local disk footprint.
FineWeb-Edu (10BT) + The Stack v2 dedup streamed from HF servers.

Reference: huggingface.co/docs/datasets/stream
"""

import torch
from torch import Tensor
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Iterator


def stream_fineweb_edu(
    tokenizer: PreTrainedTokenizer,
    seq_len: int = 2048,
    split: str = "train",
) -> Iterator[dict[str, Tensor]]:
    """Stream tokenized sequences from FineWeb-Edu 10BT sample.

    Yields:
        dict with 'input_ids' [seq_len] and 'labels' [seq_len] tensors
    """
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split=split,
        streaming=True,
    )

    yield from _tokenize_stream(ds, tokenizer, seq_len, text_key="text")


def stream_code(
    tokenizer: PreTrainedTokenizer,
    seq_len: int = 2048,
    languages: list[str] | None = None,
    split: str = "train",
) -> Iterator[dict[str, Tensor]]:
    """Stream tokenized code from starcoderdata (ungated, 250B tokens, 86 languages).

    Falls back to The Stack v2 dedup if starcoderdata fails.

    Args:
        tokenizer: tokenizer to use
        seq_len: sequence length
        languages: filter to specific languages (e.g., ["python", "javascript"])
        split: dataset split

    Yields:
        dict with 'input_ids' and 'labels' tensors
    """
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
                        yield from _tokenize_stream(ds, tokenizer, seq_len, text_key=text_key)
                    except Exception:
                        continue
                return
            else:
                ds = load_dataset(dataset_name, split=split, streaming=True)
                if languages:
                    lang_set = set(l.lower() for l in languages)
                    ds = ds.filter(lambda x: x.get("lang", "").lower() in lang_set)
                yield from _tokenize_stream(ds, tokenizer, seq_len, text_key=text_key)
                return
        except Exception as e:
            print(f"WARNING: {dataset_name} failed ({e}), trying next...")
            continue

    raise RuntimeError("No code dataset available")


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
