"""Main training loop for Memoria.

Three phases:
1. Language foundation (α=0): L_token only, model learns language
2. Cognitive awakening (α ramps): L_fe introduced, state begins organizing
3. Full training (α=α_max): both losses active, all cognitive systems enabled

Each step:
- Forward pass (pass 1): tokens → transformer + state interfaces → logits + candidates
- Compute loss: L_token + α·L_fe
- Backward + optimizer step
- Pass 2: surprise → belief update → Hebbian → Telos → consolidation → meta

Reference: autoresearch/train.py (training loop structure)
Reference: modded-nanogpt (training speed tricks)
"""

import gc
import math
import os
import time
import threading
import queue
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from pathlib import Path

from ..model.config import MemoriaConfig
from ..model.memoria_model import MemoriaModel
from ..cognition.pass2 import run_pass2
from ..data.tokenizer import get_tokenizer
from ..data.interleave import interleaved_stream
from ..data.synthetic import generate_all_synthetic
from .optimizer import setup_optimizer
from .schedule import get_lr_multiplier, get_alpha
from .distributed import (
    setup_distributed, cleanup_distributed, wrap_ddp,
    broadcast_state, gather_candidates, sync_ranks, setup_device, get_batch_to_device,
)
from ..interface.write_path import pack_candidates, unpack_candidates


class DataPrefetcher:
    """Background thread that prefetches batches from a data stream.

    Keeps a queue of ready-to-go batches so the training loop never waits
    on data loading. Batches are stacked to [B, T] and moved to the target device.
    """

    def __init__(self, data_stream, batch_size: int, device: torch.device, prefetch_count: int = 2):
        self.stream = data_stream
        self.batch_size = batch_size
        self.device = device
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            while not self.stop_event.is_set():
                seqs = [next(self.stream) for _ in range(self.batch_size)]
                input_ids = torch.stack([s['input_ids'] for s in seqs])
                labels = torch.stack([s['labels'] for s in seqs])
                self.queue.put((input_ids, labels))
        except StopIteration:
            pass

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.queue.get()
        return input_ids.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

    def stop(self):
        self.stop_event.set()


def _load_env():
    """Load .env file if it exists."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def train(
    config: MemoriaConfig,
    max_steps: int = 10000,
    time_budget: float | None = None,
    checkpoint_dir: str = "checkpoints",
    log_to_wandb: bool = True,
    push_to_hub: bool = True,
    hub_push_every: int = 2000,
    resume_from: str | None = None,
):
    """Main training entry point.

    Args:
        config: full Memoria config
        max_steps: maximum training steps (overridden by time_budget if set)
        time_budget: training time budget in seconds (None = use max_steps)
        checkpoint_dir: directory for checkpoints
        log_to_wandb: whether to log to wandb
        resume_from: checkpoint path to resume from
    """
    _load_env()
    rank, world_size, device = setup_distributed()
    tc = config.training
    is_main = (rank == 0)

    # Weights & Biases (rank 0 only)
    if log_to_wandb and not is_main:
        log_to_wandb = False
    if log_to_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "memoria"),
            config={
                'transformer': config.transformer.__dict__,
                'state': config.state.__dict__,
                'training': tc.__dict__,
                'world_size': world_size,
            },
            save_code=True,
        )

    # HuggingFace Hub (rank 0 only)
    hf_repo = os.environ.get("HF_REPO", None)
    hf_token = os.environ.get("HF_TOKEN", None)
    if push_to_hub and (not hf_repo or not hf_token or not is_main):
        if is_main and (not hf_repo or not hf_token):
            print("WARNING: push_to_hub=True but HF_REPO or HF_TOKEN not set. Disabling hub push.")
        push_to_hub = False

    # Tokenizer
    tokenizer = get_tokenizer(vocab_size=config.transformer.vocab_size)

    # Model
    if is_main:
        print("Building model...")
    model = MemoriaModel(config)
    model.init_weights()
    model, base_model = wrap_ddp(model, device, rank, world_size)
    if is_main:
        print(base_model.summary())
        if world_size > 1:
            print(f"  DDP: {world_size} GPUs, state on rank 0")

    # Optimizer (always on the base model, not the DataParallel wrapper)
    optimizer = setup_optimizer(base_model, config)

    # Data
    if is_main:
        print("Generating synthetic data...")
    synthetic_data = generate_all_synthetic()
    if is_main:
        print(f"Generated {len(synthetic_data)} synthetic sequences")
        print("Starting data stream...")
    data_stream = interleaved_stream(
        tokenizer,
        seq_len=config.transformer.sequence_len,
        weights=(0.7, 0.2, 0.1),  # FineWeb 70% + code 20% (starcoderdata, ungated) + synthetic 10%
        synthetic_data=synthetic_data,
        stack_languages=["python", "javascript", "rust", "go"],
    )

    # Resume
    start_step = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        base_model.load_state_dict(checkpoint['model_state'], strict=False)
        base_model.state.load_state_cognitive(checkpoint['cognitive_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step']
        if is_main:
            print(f"Resumed from step {start_step}")

    # Checkpoint dir
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Training state
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    smooth_loss = 0.0
    smooth_loss_token = 0.0
    smooth_loss_fe = 0.0
    t_start = time.time()
    total_training_time = 0.0

    # Gradient accumulation
    tokens_per_micro = tc.device_batch_size * config.transformer.sequence_len
    grad_accum = max(1, tc.total_batch_size // tokens_per_micro)

    if is_main:
        print(f"Training: {max_steps} steps, grad_accum={grad_accum}, "
              f"batch_size={tc.device_batch_size}, seq_len={config.transformer.sequence_len}")
        print(f"Phase 1 (α=0): steps 0-{tc.phase1_steps}")
        print(f"Phase 2 (α ramps): steps {tc.phase1_steps}-{tc.phase1_steps + tc.alpha_warmup_steps}")
        print(f"Phase 3 (α={tc.alpha_max}): steps {tc.phase1_steps + tc.alpha_warmup_steps}+")

    # Background data prefetcher (loads + stacks batches in a separate thread)
    if is_main:
        print("Starting data prefetcher (downloading initial data shard)...", flush=True)
    prefetcher = DataPrefetcher(data_stream, tc.device_batch_size, device, prefetch_count=3)
    if is_main:
        print(f"Prefetcher ready. Batch size: {tc.device_batch_size}")

    # torch.compile for fused kernels (skip if not available or if DDP)
    # Note: torch.compile with DDP requires careful handling; compile the base model
    use_compile = hasattr(torch, 'compile') and torch.cuda.is_available()
    if use_compile:
        try:
            if world_size > 1:
                # With DDP, compile the inner model before wrapping
                # Since we already wrapped, compile the forward of the base model
                base_model.transformer = torch.compile(base_model.transformer)
            else:
                model = torch.compile(model)
            if is_main:
                print("torch.compile enabled")
        except Exception as e:
            if is_main:
                print(f"torch.compile skipped: {e}")

    # DDP no_sync context for gradient accumulation
    has_no_sync = hasattr(model, 'no_sync')

    # GC management (from autoresearch — prevents Python GC stalls)
    gc.collect()
    gc.freeze()
    gc.disable()

    belief_dim = config.state.belief_dim

    for step in range(start_step, max_steps):
        t0 = time.time()

        # Broadcast cognitive state to all ranks before forward pass
        broadcast_state(base_model.state, rank, world_size)

        # Schedule
        alpha = get_alpha(step, tc)
        lr_mult = get_lr_multiplier(step, max_steps, tc)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * lr_mult

        # Accumulate gradients
        all_candidates = []
        total_loss = 0.0
        total_loss_token = 0.0
        total_loss_fe = 0.0

        for micro_step in range(grad_accum):
            if step == 0 and is_main:
                print(f"\r  micro-batch {micro_step+1}/{grad_accum}...", end="", flush=True)

            # Get pre-stacked, pre-transferred batch from prefetcher
            input_ids, labels = prefetcher.next_batch()

            # DDP no_sync for all but last micro-step (avoids redundant allreduce)
            is_last_micro = (micro_step == grad_accum - 1)
            sync_ctx = nullcontext() if (is_last_micro or not has_no_sync) else model.no_sync()

            with sync_ctx:
                with autocast_ctx:
                    fwd = model(input_ids, targets=labels)
                    result = base_model.compute_loss_from_forward(fwd, input_ids, labels, alpha=alpha)

                loss = result['loss'] / grad_accum
                loss.backward()

            total_loss += loss.item()
            total_loss_token += result['loss_token'].item() / grad_accum
            total_loss_fe += result['loss_fe'].item() / grad_accum
            all_candidates.extend(result['candidates'])

        if step == 0 and is_main:
            print(f"\r  Step 0 complete.{' ' * 30}", flush=True)

        # Optimizer step (DDP syncs gradients automatically)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        # Fast fail
        if math.isnan(total_loss) or total_loss > 100:
            if is_main:
                print(f"\nFAIL: loss exploded at step {step}: {total_loss}")
            break

        # Gather candidates from all ranks to rank 0 for Pass 2
        packed = pack_candidates(all_candidates)
        packed = gather_candidates(packed, rank, world_size, belief_dim)

        # Pass 2: cognitive update on rank 0 only (state is sequential)
        base_model.detach_state()

        if is_main:
            gathered_candidates = unpack_candidates(packed, belief_dim)
            read_indices = list(set(
                c.matched_slot for c in gathered_candidates if c.matched_slot >= 0
            ))

            pass2_stats = run_pass2(
                state=base_model.state,
                candidates=gathered_candidates,
                read_belief_indices=read_indices,
                current_step=step,
                is_sequence_boundary=True,
                temperature=tc.fe_temperature,
            )
        else:
            pass2_stats = {
                'beta': 1.0, 'active_beliefs': 0, 'active_edges': 0,
                'active_goals': 0, 'total_surprise': 0.0,
            }

        # Sync ranks after Pass 2 (rank 0 does extra work, rank 1 must wait)
        sync_ranks(world_size)

        # Timing
        t1 = time.time()
        dt = t1 - t0
        if step > 5:
            total_training_time += dt

        # Logging
        ema = 0.9
        smooth_loss = ema * smooth_loss + (1 - ema) * total_loss
        smooth_loss_token = ema * smooth_loss_token + (1 - ema) * total_loss_token
        smooth_loss_fe = ema * smooth_loss_fe + (1 - ema) * total_loss_fe
        debiased = smooth_loss / (1 - ema ** (step - start_step + 1))

        if step % tc.log_interval == 0 and is_main:
            phase = "P1" if alpha == 0 else ("P2" if alpha < tc.alpha_max else "P3")
            print(
                f"\rstep {step:05d} [{phase}] | "
                f"loss: {debiased:.4f} (tok: {smooth_loss_token / (1 - ema ** (step - start_step + 1)):.4f}, "
                f"fe: {smooth_loss_fe / (1 - ema ** (step - start_step + 1)):.4f}) | "
                f"α: {alpha:.4f} | β: {pass2_stats['beta']:.3f} | "
                f"beliefs: {pass2_stats['active_beliefs']} | "
                f"edges: {pass2_stats['active_edges']} | "
                f"goals: {pass2_stats['active_goals']} | "
                f"dt: {dt * 1000:.0f}ms",
                end="", flush=True,
            )

        if log_to_wandb:
            import wandb
            wandb.log({
                'step': step,
                'loss': total_loss,
                'loss_token': total_loss_token,
                'loss_fe': total_loss_fe,
                'alpha': alpha,
                'beta': pass2_stats['beta'],
                'lr_mult': lr_mult,
                'active_beliefs': pass2_stats['active_beliefs'],
                'active_edges': pass2_stats['active_edges'],
                'active_goals': pass2_stats['active_goals'],
                'total_surprise': pass2_stats['total_surprise'],
                'new_beliefs': pass2_stats.get('belief_new_allocations', 0),
                'reconsolidations': pass2_stats.get('belief_reconsolidations', 0),
                'goals_generated': pass2_stats.get('goals_generated', 0),
                'dt_ms': dt * 1000,
            })

        # Checkpoint (rank 0 only)
        if is_main and step > 0 and step % tc.checkpoint_interval == 0:
            save_checkpoint(base_model, optimizer, step, ckpt_path / f"step_{step}.pt")

        # Push to HuggingFace Hub (rank 0 only)
        if push_to_hub and step > 0 and step % hub_push_every == 0:
            _push_to_hub(base_model, step, ckpt_path, hf_repo, hf_token)

        # Time budget
        if time_budget and total_training_time >= time_budget:
            if is_main:
                print(f"\nTime budget reached ({time_budget}s) at step {step}")
            break

        # Periodic GC
        if (step + 1) % 5000 == 0:
            gc.collect()

    # Cleanup
    prefetcher.stop()
    gc.enable()

    if is_main:
        print(f"\nTraining complete. {step + 1} steps, {total_training_time:.1f}s training time.")

        # Final checkpoint
        save_checkpoint(base_model, optimizer, step, ckpt_path / "final.pt")

        # Final push to hub
        if push_to_hub:
            _push_to_hub(base_model, step, ckpt_path, hf_repo, hf_token, is_final=True)

    if log_to_wandb:
        import wandb
        wandb.finish()

    cleanup_distributed()
    return base_model


def save_checkpoint(model, optimizer, step: int, path: Path):
    """Save model + cognitive state + optimizer checkpoint."""
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'cognitive_state': model.state.state_dict_cognitive(),
        'optimizer_state': optimizer.state_dict(),
    }, path)
    print(f"\n  Checkpoint saved: {path}")


def _push_to_hub(model, step: int, ckpt_path: Path, repo_id: str, token: str, is_final: bool = False):
    """Push checkpoint to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        # Upload the latest checkpoint
        tag = "final" if is_final else f"step-{step}"
        ckpt_file = ckpt_path / ("final.pt" if is_final else f"step_{step}.pt")

        if ckpt_file.exists():
            api.upload_file(
                path_or_fileobj=str(ckpt_file),
                path_in_repo=f"checkpoints/{ckpt_file.name}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Checkpoint at step {step}",
            )
            print(f"\n  Pushed to hub: {repo_id}/checkpoints/{ckpt_file.name}")

        # Also upload training state summary
        summary = {
            'step': step,
            'active_beliefs': model.state.num_active_beliefs(),
            'active_edges': model.state.num_active_edges(),
            'active_goals': model.state.num_active_goals(),
            'beta': model.state.beta,
        }
        import json, tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(summary, f, indent=2)
            f.flush()
            api.upload_file(
                path_or_fileobj=f.name,
                path_in_repo=f"checkpoints/{tag}_summary.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Training summary at step {step}",
            )
        os.unlink(f.name)

    except Exception as e:
        print(f"\n  WARNING: Hub push failed: {e}")
