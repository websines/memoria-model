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
import torch
import torch.nn.functional as F
from pathlib import Path

from ..model.config import MemoriaConfig
from ..model.memoria_model import MemoriaModel
from ..cognition.pass2 import run_pass2
from ..data.tokenizer import get_tokenizer
from ..data.interleave import interleaved_stream
from ..data.synthetic import generate_all_synthetic
from .optimizer import setup_optimizer
from .schedule import get_lr_multiplier, get_alpha
from .distributed import setup_device, get_batch_to_device


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
    device = setup_device()
    tc = config.training

    # Weights & Biases
    if log_to_wandb:
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "memoria"),
            config={
                'transformer': config.transformer.__dict__,
                'state': config.state.__dict__,
                'training': tc.__dict__,
            },
            save_code=True,
        )

    # HuggingFace Hub
    hf_repo = os.environ.get("HF_REPO", None)
    hf_token = os.environ.get("HF_TOKEN", None)
    if push_to_hub and (not hf_repo or not hf_token):
        print("WARNING: push_to_hub=True but HF_REPO or HF_TOKEN not set. Disabling hub push.")
        push_to_hub = False

    # Tokenizer
    tokenizer = get_tokenizer(vocab_size=config.transformer.vocab_size)

    # Model
    print("Building model...")
    model = MemoriaModel(config)
    model.init_weights()
    model = model.to(device)
    print(model.summary())

    # Optimizer
    optimizer = setup_optimizer(model, config)

    # Data
    print("Generating synthetic data...")
    synthetic_data = generate_all_synthetic()
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
        model.load_state_dict(checkpoint['model_state'], strict=False)
        model.state.load_state_cognitive(checkpoint['cognitive_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step']
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
    tokens_per_step = tc.device_batch_size * config.transformer.sequence_len
    grad_accum = max(1, tc.total_batch_size // tokens_per_step)

    print(f"Training: {max_steps} steps, grad_accum={grad_accum}, "
          f"batch_size={tc.device_batch_size}, seq_len={config.transformer.sequence_len}")
    print(f"Phase 1 (α=0): steps 0-{tc.phase1_steps}")
    print(f"Phase 2 (α ramps): steps {tc.phase1_steps}-{tc.phase1_steps + tc.alpha_warmup_steps}")
    print(f"Phase 3 (α={tc.alpha_max}): steps {tc.phase1_steps + tc.alpha_warmup_steps}+")

    # Prefetch first batch (this can take a while on first stream access)
    print("Prefetching first batch (downloading initial data shard)...", flush=True)
    first_batch = next(data_stream)
    print(f"First batch ready. input_ids shape: {first_batch['input_ids'].shape}")
    prefetched = first_batch

    # GC management (from autoresearch — prevents Python GC stalls)
    gc.collect()
    gc.freeze()
    gc.disable()

    for step in range(start_step, max_steps):
        t0 = time.time()

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
            # Get batch (use prefetched if available)
            if prefetched is not None:
                batch = prefetched
                prefetched = None
            else:
                batch = next(data_stream)
            input_ids = batch['input_ids'].unsqueeze(0).to(device)  # [1, T]
            labels = batch['labels'].unsqueeze(0).to(device)

            # Forward + loss
            with autocast_ctx:
                result = model.compute_loss(input_ids, labels, alpha=alpha)

            loss = result['loss'] / grad_accum
            loss.backward()

            total_loss += result['loss'].item()
            total_loss_token += result['loss_token'].item()
            total_loss_fe += result['loss_fe'].item()
            all_candidates.extend(result['candidates'])

        # Optimizer step
        optimizer.step()
        model.zero_grad(set_to_none=True)

        # Fast fail
        if math.isnan(total_loss) or total_loss > 100 * grad_accum:
            print(f"\nFAIL: loss exploded at step {step}: {total_loss}")
            break

        # Pass 2: cognitive update (detached from optimizer graph)
        model.detach_state()

        # Collect read indices from candidates (which beliefs were accessed)
        read_indices = list(set(
            c.matched_slot for c in all_candidates if c.matched_slot >= 0
        ))

        pass2_stats = run_pass2(
            state=model.state,
            candidates=all_candidates,
            read_belief_indices=read_indices,
            current_step=step,
            is_sequence_boundary=True,
            temperature=tc.fe_temperature,
        )

        # Timing
        t1 = time.time()
        dt = t1 - t0
        if step > 5:
            total_training_time += dt

        # Logging
        ema = 0.9
        smooth_loss = ema * smooth_loss + (1 - ema) * (total_loss / grad_accum)
        smooth_loss_token = ema * smooth_loss_token + (1 - ema) * (total_loss_token / grad_accum)
        smooth_loss_fe = ema * smooth_loss_fe + (1 - ema) * (total_loss_fe / grad_accum)
        debiased = smooth_loss / (1 - ema ** (step - start_step + 1))

        if step % tc.log_interval == 0:
            phase = "P1" if alpha == 0 else ("P2" if alpha < tc.alpha_max else "P3")
            print(
                f"\rstep {step:05d} [{phase}] | "
                f"loss: {debiased:.4f} (tok: {smooth_loss_token / (1 - ema ** (step + 1)):.4f}, "
                f"fe: {smooth_loss_fe / (1 - ema ** (step + 1)):.4f}) | "
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
                'loss': total_loss / grad_accum,
                'loss_token': total_loss_token / grad_accum,
                'loss_fe': total_loss_fe / grad_accum,
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

        # Checkpoint
        if step > 0 and step % tc.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, ckpt_path / f"step_{step}.pt")

        # Push to HuggingFace Hub
        if push_to_hub and step > 0 and step % hub_push_every == 0:
            _push_to_hub(model, step, ckpt_path, hf_repo, hf_token)

        # Time budget
        if time_budget and total_training_time >= time_budget:
            print(f"\nTime budget reached ({time_budget}s) at step {step}")
            break

        # Periodic GC
        if (step + 1) % 5000 == 0:
            gc.collect()

    print(f"\nTraining complete. {step + 1} steps, {total_training_time:.1f}s training time.")

    # Final checkpoint
    save_checkpoint(model, optimizer, step, ckpt_path / "final.pt")

    # Final push to hub
    if push_to_hub:
        _push_to_hub(model, step, ckpt_path, hf_repo, hf_token, is_final=True)

    if log_to_wandb:
        import wandb
        wandb.finish()

    return model


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
