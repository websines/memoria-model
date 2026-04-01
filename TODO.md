# Training Codebase TODO

## Critical — Fixed

- [x] **Free energy loss contributes zero gradients** — `compute_free_energy()` read from detached state. **Fix**: new `compute_differentiable_free_energy()` in `core/losses.py` computes F_proxy over forward-pass attention weights + retrieved representations + observation projections. Gradients now flow to all 12+ interface parameters. Original `compute_free_energy()` kept for beta/stats only.
  - `memoria/core/losses.py` — new shared module
  - `memoria/interface/read_path.py` — returns attn weights + retrieved per-head
  - `memoria/interface/write_path.py` — returns obs_vectors in computation graph
  - `memoria/interface/layer.py` — passes through new returns
  - `memoria/model/memoria_model.py` — uses differentiable L_fe
  - `memoria/model/pretrained_model.py` — uses differentiable L_fe

- [x] **Effective batch size is 2x with DDP** — `grad_accum` now divides by `accelerator.num_processes`.
  - `memoria/training/train.py:166`

- [x] **`detach_state()` was a no-op wasting memory** — Now a documented no-op (pass). State tensors were never in the computation graph.
  - `memoria/model/memoria_model.py`
  - `memoria/model/pretrained_model.py`

- [x] **No gradient clipping** — Added `grad_clip_norm=1.0` to TrainingConfig. Applied via `accelerator.clip_grad_norm_()`.
  - `memoria/model/config.py`
  - `memoria/training/train.py`

## Serious — Fixed

- [x] **Sequence boundary decay was catastrophically aggressive** — Changed decay factor from 0.95 to 0.995, and only applied every 10 steps. Beliefs now halve in ~1386 steps instead of ~14.
  - `memoria/cognition/meta_learning.py` — decay_factor default 0.995
  - `memoria/cognition/pass2.py` — only apply every 10 steps

- [x] **Write path had per-batch Python loop** — Replaced `for b in range(B)` with single batched matmul over all B*T positions.
  - `memoria/interface/write_path.py` — `_match_and_buffer_batched()`

- [x] **Co-activation pair explosion was O(N^2)** — Capped at ~23 indices (253 pairs max). Samples random subset if more.
  - `memoria/cognition/hebbian.py` — `extract_co_activations()` with max_pairs

- [x] **`find_unused_parameters=True` in DDP** — Replaced hand-rolled DDP with HF Accelerate. No more find_unused_parameters.
  - `memoria/training/train.py` — uses Accelerate
  - `memoria/training/distributed.py` — trimmed to state-specific ops only

- [x] **DataPrefetcher could deadlock** — Added error propagation from worker thread + 120s timeout on queue.get().
  - `memoria/training/train.py` — DataPrefetcher class

## Medium — Fixed

- [x] **Consolidation similarity matrix was O(N^2)** — Caps at 512 beliefs via random sampling.
  - `memoria/cognition/consolidation.py` — max_beliefs_to_check parameter

- [x] **Hebbian edge index rebuilt every step in Python** — Replaced Python dict with tensor broadcast matching (edge_mins/maxs vs pair_mins/maxs).
  - `memoria/cognition/hebbian.py` — tensor-based matching

- [x] **Duplicate `chunked_cross_entropy`** — Extracted to `core/losses.py`, both model files import from there.
  - `memoria/core/losses.py`

- [x] **No periodic eval** — Added `_run_eval()` computing perplexity on 20 FineWeb samples every `eval_interval` steps.
  - `memoria/training/train.py`

- [x] **Synthetic data generated on all ranks** — Only main process generates synthetic data.
  - `memoria/training/train.py`

- [x] **Checkpoint doesn't save data stream position** — Tracks `samples_consumed` counter, saves in checkpoint. On resume, fast-forwards the interleaved stream by draining N samples before starting the prefetcher.
  - `memoria/training/train.py` — counter + save/restore + fast-forward

## Design — Fixed

- [x] **Hub push was synchronous** — Now runs in background thread via `_push_to_hub_async()`.
  - `memoria/training/train.py`

- [x] **Muon optimizer was TODO** — Implemented `Muon` class with Newton-Schulz orthogonalization. 2D matrix params use Muon, everything else uses AdamW, wrapped in `_CombinedOptimizer`.
  - `memoria/training/optimizer.py`

- [x] **Silent data mix changes** — Added logging when code dataset fails and weight redistributes.
  - `memoria/data/interleave.py`

- [x] **No warmup** — Changed default `warmup_ratio` from 0.0 to 0.02.
  - `memoria/model/config.py`

- [x] **SPSA evaluated stale state** — Replaced single-step `spsa_step()` with `SPSAController` class. Applies +Δ for N steps, measures mean F(+Δ), then applies -Δ for N steps, measures mean F(-Δ), then computes gradient. Perturbations now causally affect state via Pass 2 before evaluation. Controller state is checkpointed.
  - `memoria/cognition/meta_learning.py` — `SPSAController` class
  - `memoria/cognition/pass2.py` — accepts controller, calls it every step
  - `memoria/training/train.py` — creates controller, passes to pass2, saves/restores

## Dependencies Added

- `accelerate>=1.0` — DDP, mixed precision, gradient clipping
- Muon optimizer implemented inline (no external dep)
