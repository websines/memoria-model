"""Main training loop for Memoria.

Three phases:
1. Language foundation (α=0): L_token only, model learns language
2. Cognitive awakening (α ramps): L_fe introduced, state begins organizing
3. Full training (α=α_max): both losses active, all cognitive systems enabled

Each step:
- Forward pass (pass 1): tokens → transformer + state interfaces → logits + candidates
- Compute loss: L_token + α·L_fe (differentiable) + α·0.1·L_utility
- Backward + optimizer step
- Pass 2: surprise → belief update → Hebbian → Telos → consolidation → meta

Uses HuggingFace Accelerate for DDP, mixed precision, and gradient clipping.
Cognitive state broadcast/gather handled separately (not a DDP parameter).
"""

import gc
import math
import os
import time
import threading
import queue
import torch
from contextlib import nullcontext
from pathlib import Path

from ..model.config import MemoriaConfig
from ..model.memoria_model import MemoriaModel
from ..model.pretrained_model import PretrainedMemoriaModel
from ..cognition.pass2 import run_pass2
from ..data.tokenizer import get_tokenizer
from ..data.interleave import interleaved_stream
from ..data.streaming import stream_fineweb_edu
from ..data.synthetic import generate_all_synthetic
from .optimizer import setup_optimizer
from .schedule import get_lr_multiplier, get_alpha
from .distributed import (
    broadcast_state, gather_candidates, gather_read_indices, sync_ranks,
)
from ..interface.write_path import pack_candidates, unpack_candidates
from ..cognition.meta_learning import SPSAController  # kept for checkpoint backward compat
from .cognitive_seed import save_cognitive_seed


class DataPrefetcher:
    """Background thread that prefetches batches from a data stream.

    Keeps a queue of ready-to-go batches so the training loop never waits
    on data loading. Propagates errors from the worker thread.
    """

    def __init__(self, data_stream, batch_size: int, device: torch.device, prefetch_count: int = 2):
        self.stream = data_stream
        self.batch_size = batch_size
        self.device = device
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self._error = None
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
        except Exception as e:
            self._error = e

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._error is not None:
            raise RuntimeError(f"Data prefetcher failed: {self._error}")
        # Short poll intervals (5s) so worker errors surface quickly instead of
        # blocking for the full 120s timeout before checking
        deadline = time.monotonic() + 120
        while True:
            try:
                input_ids, labels = self.queue.get(timeout=5)
                break
            except queue.Empty:
                if self._error is not None:
                    raise RuntimeError(f"Data prefetcher failed: {self._error}")
                if not self.thread.is_alive():
                    raise RuntimeError("Data prefetcher thread died unexpectedly")
                if time.monotonic() > deadline:
                    raise RuntimeError("Data prefetcher timed out after 120s — check data pipeline")
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
    cognitive_seed: str | None = None,  # path to cognitive seed from previous run
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
    tc = config.training

    # ── Accelerate setup ──
    from accelerate import Accelerator, DistributedDataParallelKwargs

    # find_unused_parameters=True is required: cognitive state is dynamic (beliefs
    # appear/disappear, goals come and go), so the set of parameters that contribute
    # to loss changes between steps. static_graph won't work either.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Weights & Biases (main process only)
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

    # HuggingFace Hub (main process only)
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
    if config.backbone == "pretrained":
        model = PretrainedMemoriaModel(config)
    else:
        model = MemoriaModel(config)
    model.init_weights()

    # Optimizer
    optimizer = setup_optimizer(model, config)

    # Prepare model with Accelerate (DDP wrapping + device placement).
    # Optimizer is NOT prepared: in scratch mode it's a _CombinedOptimizer (not an
    # Optimizer subclass), and bf16 mode doesn't need GradScaler wrapping.
    # Optimizer param refs remain valid because DDP reuses the same underlying params.
    model = accelerator.prepare(model)
    base_model = accelerator.unwrap_model(model)

    if is_main:
        print(base_model.summary())
        if world_size > 1:
            print(f"  DDP: {world_size} GPUs via Accelerate, state on rank 0")

    # Data (synthetic only on main process, broadcast if needed)
    if is_main:
        print("Generating synthetic data...")
    synthetic_data = generate_all_synthetic() if is_main else []
    if is_main:
        print(f"Generated {len(synthetic_data)} synthetic sequences")
        print("Starting data stream...")
    data_stream = interleaved_stream(
        tokenizer,
        seq_len=config.transformer.sequence_len,
        weights=(0.7, 0.2, 0.1),
        synthetic_data=synthetic_data if synthetic_data else None,
        stack_languages=["python", "javascript", "rust", "go"],
    )

    # SPSA controller removed — its 3 tunable params (reconsolidation_threshold,
    # match_threshold, goal_dedup_threshold) are now nn.Parameters in MetaParams,
    # trained by backprop through L_fe. No gradient-free optimization needed.
    spsa_controller = None

    # Load cognitive seed from previous run (before resume, which may override)
    if cognitive_seed and is_main:
        from .cognitive_seed import load_cognitive_seed
        load_cognitive_seed(base_model.state, cognitive_seed)

    # Belief advantage tracking (with-state vs without-state loss delta)
    belief_advantage_ema = 0.0

    # Resume
    start_step = 0
    samples_consumed = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        load_result = base_model.load_state_dict(checkpoint['model_state'], strict=False)
        if is_main and (load_result.missing_keys or load_result.unexpected_keys):
            # Filter out expected missing keys (frozen backbone excluded from checkpoint)
            unexpected_missing = [k for k in load_result.missing_keys
                                  if not k.startswith('backbone.')]
            if unexpected_missing:
                print(f"  Resume: {len(unexpected_missing)} missing keys (new params): "
                      f"{unexpected_missing[:5]}{'...' if len(unexpected_missing) > 5 else ''}")
            elif load_result.missing_keys:
                print(f"  Resume: {len(load_result.missing_keys)} missing backbone keys (expected, backbone loaded from pretrained)")
            if load_result.unexpected_keys:
                print(f"  Resume: {len(load_result.unexpected_keys)} unexpected keys (removed params): "
                      f"{load_result.unexpected_keys[:5]}{'...' if len(load_result.unexpected_keys) > 5 else ''}")
        base_model.state.load_state_cognitive(checkpoint['cognitive_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step']
        samples_consumed = checkpoint.get('samples_consumed', 0)
        # spsa_state in old checkpoints is ignored — SPSA replaced by learned MetaParams
        if is_main:
            print(f"Resumed from step {start_step}")

    # Fast-forward data stream past already-consumed samples
    if samples_consumed > 0 and is_main:
        print(f"Fast-forwarding data stream past {samples_consumed} consumed samples...", flush=True)
        for _ in range(samples_consumed):
            next(data_stream)
        print(f"Data stream resumed at position {samples_consumed}")

    # Checkpoint dir
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Training state
    smooth_loss = 0.0
    smooth_loss_token = 0.0
    smooth_loss_fe = 0.0
    t_start = time.time()
    total_training_time = 0.0

    # Gradient accumulation (correctly divides by world_size)
    tokens_per_micro = tc.device_batch_size * config.transformer.sequence_len
    grad_accum = max(1, tc.total_batch_size // (tokens_per_micro * world_size))

    if is_main:
        print(f"Training: {max_steps} steps, grad_accum={grad_accum}, "
              f"batch_size={tc.device_batch_size}, seq_len={config.transformer.sequence_len}")
        print(f"Effective batch: {grad_accum * tokens_per_micro * world_size} tokens/step")
        print(f"Phase 1 (α=0): steps 0-{tc.phase1_steps}")
        print(f"Phase 2 (α ramps): steps {tc.phase1_steps}-{tc.phase1_steps + tc.alpha_warmup_steps}")
        print(f"Phase 3 (α={tc.alpha_max}): steps {tc.phase1_steps + tc.alpha_warmup_steps}+")

    # Background data prefetcher
    if is_main:
        print("Starting data prefetcher (downloading initial data shard)...", flush=True)
    prefetcher = DataPrefetcher(data_stream, tc.device_batch_size, device, prefetch_count=3)
    if is_main:
        print(f"Prefetcher ready. Batch size: {tc.device_batch_size}")

    # torch.compile
    should_compile = (
        config.backbone == "pretrained"
        or os.environ.get('MEMORIA_COMPILE', '') == '1'
    )
    if should_compile:
        try:
            if config.backbone == "pretrained":
                for i, layer in enumerate(base_model.backbone.model.layers):
                    base_model.backbone.model.layers[i] = torch.compile(layer)
            else:
                base_model.transformer = torch.compile(base_model.transformer)
            if is_main:
                print("torch.compile enabled")
        except Exception as e:
            if is_main:
                print(f"torch.compile skipped: {e}")

    # DDP no_sync support
    has_no_sync = hasattr(model, 'no_sync')

    # GC management: disable automatic GC to avoid mid-step pauses,
    # but don't freeze (frozen objects can't be cycle-collected later)
    gc.collect()
    gc.disable()

    belief_dim = config.state.belief_dim

    # Persistent eval stream (main process only). Advancing past 1000 samples
    # separates eval data from early training data and ensures each eval call
    # gets fresh sequences instead of re-evaluating the same 20 every time.
    if is_main:
        eval_data_stream = stream_fineweb_edu(tokenizer, config.transformer.sequence_len)
        for _ in range(1000):
            next(eval_data_stream)
    else:
        eval_data_stream = None

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
        all_read_indices = []
        total_loss = 0.0
        total_loss_token = 0.0
        total_loss_fe = 0.0

        for micro_step in range(grad_accum):
            if step == 0 and is_main:
                print(f"\r  micro-batch {micro_step+1}/{grad_accum}...", end="", flush=True)

            input_ids, labels = prefetcher.next_batch()

            # DDP no_sync for all but last micro-step
            is_last_micro = (micro_step == grad_accum - 1)
            sync_ctx = nullcontext() if (is_last_micro or not has_no_sync) else model.no_sync()

            with sync_ctx:
                result = model(input_ids, targets=labels, alpha=alpha)
                loss = result['loss'] / grad_accum
                accelerator.backward(loss)

            total_loss += loss.item()
            total_loss_token += result['loss_token'].item() / grad_accum
            total_loss_fe += result['loss_fe'].item() / grad_accum
            all_candidates.extend(result['candidates'])
            all_read_indices.extend(result.get('read_indices', []))

        if step == 0 and is_main:
            print(f"\r  Step 0 complete.{' ' * 30}", flush=True)

        # Gradient clipping (skip Muon params — Newton-Schulz orthogonalization
        # produces unit-spectral-norm updates, making pre-clip counterproductive)
        if tc.grad_clip_norm > 0:
            adamw = getattr(optimizer, 'adamw', optimizer)  # _CombinedOptimizer or plain AdamW
            clip_params = [p for g in adamw.param_groups for p in g['params'] if p.grad is not None]
            if clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, tc.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Fast fail
        if math.isnan(total_loss) or total_loss > 100:
            if is_main:
                print(f"\nFAIL: loss exploded at step {step}: {total_loss}")
            break

        # Gather candidates and read indices from all ranks to rank 0
        packed = pack_candidates(all_candidates)
        packed = gather_candidates(packed, rank, world_size, belief_dim)
        all_read_indices = gather_read_indices(all_read_indices, rank, world_size, device)

        # Pass 2: cognitive update on rank 0 only
        base_model.detach_state()

        # Detect actual sequence boundaries (EOS token present in last batch)
        eos_id = tokenizer.eos_token_id
        has_boundary = eos_id is not None and (input_ids == eos_id).any().item()

        if is_main:
            gathered_candidates = unpack_candidates(packed, belief_dim)
            read_indices = list(set(all_read_indices))

            pass2_stats = run_pass2(
                state=base_model.state,
                candidates=gathered_candidates,
                read_belief_indices=read_indices,
                current_step=step,
                is_sequence_boundary=has_boundary,
                total_steps=max_steps,
                belief_advantage=belief_advantage_ema,
            )
        else:
            pass2_stats = {
                'beta': 1.0, 'active_beliefs': 0, 'active_edges': 0,
                'active_goals': 0, 'total_surprise': 0.0,
            }

        sync_ranks(world_size)

        # Track data consumption for checkpoint resume
        samples_consumed += grad_accum * tc.device_batch_size

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

        # Belief advantage evaluation: with-state vs without-state loss delta
        # Measures whether the cognitive state is actually helping predictions
        # Runs at eval_interval / 5 (derived from config, not hardcoded)
        belief_eval_interval = max(1, tc.eval_interval // 5)
        if is_main and step > 0 and step % belief_eval_interval == 0 and alpha > 0:
            base_model.eval()
            with torch.no_grad():
                # Use held-out data for unbiased measurement (not the training micro-batch)
                eval_batch = next(eval_data_stream)
                eval_ids = eval_batch['input_ids'].unsqueeze(0).to(device)
                eval_labels = eval_batch['labels'].unsqueeze(0).to(device)
                result_with = base_model(eval_ids, targets=eval_labels, alpha=alpha)
                loss_with = result_with['loss_token'].item()
                result_without = base_model(eval_ids, targets=eval_labels, alpha=0.0)
                loss_without = result_without['loss_token'].item()
            base_model.train()
            belief_advantage = loss_without - loss_with  # positive = beliefs help
            ba_decay = 0.95  # EMA decay — matches controller's reward_ema_decay convention
            belief_advantage_ema = ba_decay * belief_advantage_ema + (1 - ba_decay) * belief_advantage
            if log_to_wandb:
                import wandb
                wandb.log({
                    'belief_advantage': belief_advantage,
                    'belief_advantage_ema': belief_advantage_ema,
                    'step': step,
                })
            if step % tc.eval_interval == 0:
                print(f"\n  [belief advantage] step {step}: {belief_advantage:.4f} (ema: {belief_advantage_ema:.4f})")

        # Controller REINFORCE loss (train at same interval as belief advantage)
        if is_main and step > 0 and step % belief_eval_interval == 0:
            controller_loss = base_model.state.controller.compute_loss()
            if controller_loss.item() > 0:
                controller_loss.backward()
                # Step immediately so REINFORCE grads don't bleed into next step's
                # supervised loss. Only controller params have grads here (others are
                # None after zero_grad above), so only controller params get updated.
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Periodic eval
        if is_main and step > 0 and step % tc.eval_interval == 0:
            eval_ppl = _run_eval(base_model, eval_data_stream, device)
            print(f"\n  [eval] step {step}: perplexity = {eval_ppl:.2f}")
            if log_to_wandb:
                wandb.log({'eval_perplexity': eval_ppl, 'step': step})

        # Checkpoint (main process only)
        if is_main and step > 0 and step % tc.checkpoint_interval == 0:
            save_checkpoint(base_model, optimizer, step, ckpt_path / f"step_{step}.pt",
                           samples_consumed, spsa_controller)

        # Push to HuggingFace Hub (async, main process only)
        if push_to_hub and step > 0 and step % hub_push_every == 0:
            _push_to_hub_async(base_model, step, ckpt_path, hf_repo, hf_token)

        # Time budget
        if time_budget and total_training_time >= time_budget:
            if is_main:
                print(f"\nTime budget reached ({time_budget}s) at step {step}")
            break

        # Periodic GC (every 500 steps — cognitive state churn creates cycles)
        if (step + 1) % 500 == 0:
            gc.collect()

    # Cleanup
    prefetcher.stop()
    gc.enable()

    if is_main:
        print(f"\nTraining complete. {step + 1} steps, {total_training_time:.1f}s training time.")
        save_checkpoint(base_model, optimizer, step, ckpt_path / "final.pt",
                        samples_consumed, spsa_controller)
        # Save cognitive seed for next run
        save_cognitive_seed(base_model.state, ckpt_path / "cognitive_seed.pt")
        if push_to_hub:
            _push_to_hub_async(base_model, step, ckpt_path, hf_repo, hf_token, is_final=True)

    if log_to_wandb:
        import wandb
        wandb.finish()

    accelerator.end_training()
    return base_model


def _run_eval(model, eval_stream, device, n_batches: int = 20) -> float:
    """Quick eval: compute perplexity on held-out FineWeb samples.

    Uses a persistent eval stream so each call evaluates on fresh data
    instead of re-reading the same first N batches from a new stream.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = next(eval_stream)
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            labels = batch['labels'].unsqueeze(0).to(device)
            result = model(input_ids, targets=labels, alpha=0.0)
            total_loss += result['loss_token'].item()
    model.train()
    return math.exp(min(total_loss / n_batches, 20.0))  # cap to avoid overflow


def save_checkpoint(model, optimizer, step: int, path: Path,
                    samples_consumed: int = 0, spsa_controller=None):
    """Save model + cognitive state + optimizer + stream position checkpoint."""
    # In pretrained mode, skip frozen backbone to avoid multi-GB checkpoint bloat
    model_state = model.state_dict()
    if hasattr(model, 'backbone'):
        model_state = {k: v for k, v in model_state.items()
                       if not k.startswith('backbone.')}

    ckpt = {
        'step': step,
        'model_state': model_state,
        'cognitive_state': model.state.state_dict_cognitive(),
        'optimizer_state': optimizer.state_dict(),
        'samples_consumed': samples_consumed,
    }
    torch.save(ckpt, path)
    print(f"\n  Checkpoint saved: {path} (stream pos: {samples_consumed} samples)")


def _push_to_hub_async(model, step: int, ckpt_path: Path, repo_id: str, token: str, is_final: bool = False):
    """Push checkpoint to HuggingFace Hub in a background thread."""
    # Snapshot state summary on main thread before spawning — avoids racing
    # with pass 2 which concurrently mutates model.state
    summary = {
        'step': step,
        'active_beliefs': model.state.num_active_beliefs(),
        'active_edges': model.state.num_active_edges(),
        'active_goals': model.state.num_active_goals(),
        'beta': model.state.beta,
    }

    def _upload():
        try:
            from huggingface_hub import HfApi
            import json, tempfile
            api = HfApi(token=token)

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

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(summary, f, indent=2)
                tmp_path = f.name
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=f"checkpoints/{tag}_summary.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Training summary at step {step}",
            )
            os.unlink(tmp_path)
            print(f"\n  Pushed to hub: {repo_id}/checkpoints/{ckpt_file.name}")
        except Exception as e:
            print(f"\n  WARNING: Hub push failed: {e}")

    thread = threading.Thread(target=_upload, daemon=True)
    thread.start()
