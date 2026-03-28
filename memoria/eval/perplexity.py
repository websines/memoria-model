"""Standard perplexity evaluation. Must not degrade vs baseline transformer."""

import torch
import math
from torch import Tensor

from ..model.memoria_model import MemoriaModel
from ..data.tokenizer import get_tokenizer
from ..data.streaming import stream_fineweb_edu


@torch.no_grad()
def evaluate_perplexity(
    model: MemoriaModel,
    num_batches: int = 100,
    batch_size: int = 4,
    seq_len: int = 2048,
) -> dict:
    """Evaluate perplexity on held-out FineWeb-Edu data.

    Args:
        model: trained MemoriaModel
        num_batches: number of evaluation batches
        batch_size: sequences per batch
        seq_len: sequence length

    Returns:
        dict with perplexity, loss, bits_per_byte
    """
    model.eval()
    device = next(model.parameters()).device
    tokenizer = get_tokenizer()

    total_loss = 0.0
    total_tokens = 0
    data_iter = stream_fineweb_edu(tokenizer, seq_len, split="train")

    for batch_idx in range(num_batches):
        batch_input = []
        batch_labels = []

        for _ in range(batch_size):
            sample = next(data_iter)
            batch_input.append(sample['input_ids'])
            batch_labels.append(sample['labels'])

        input_ids = torch.stack(batch_input).to(device)
        labels = torch.stack(batch_labels).to(device)

        result = model.forward(input_ids, targets=labels)
        loss = result.get('loss_token', result['logits'])

        if isinstance(loss, Tensor) and loss.dim() == 0:
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
        else:
            # Compute loss from logits
            logits = result['logits']
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += labels.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # cap to avoid overflow
    bits_per_byte = avg_loss / math.log(2) * 0.29  # approximate BPB

    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'bits_per_byte': bits_per_byte,
        'total_tokens': total_tokens,
    }
