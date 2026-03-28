# Training Log

## Run 1 — March 28, 2026, ~4:03 AM UTC

**Config:** small (324M params: 318M transformer + 5.8M interface)
**Hardware:** 2x RTX 3090 (WSL2), single GPU training
**Command:** `python -m memoria train --config small --max-steps 5000`
**wandb:** https://wandb.ai/subh03/memoria_prototype
**Run:** `faithful-plasma-4` (axn7ojv0)

**Setup issues resolved:**
- `memoria.data` module not tracked in git (empty `__init__.py` skipped by git) → fixed
- CUDA not available in WSL2 ("GPU access blocked by operating system") → fixed with `wsl --shutdown` restart
- The Stack v2 gated dataset → got HF approval
- FineWeb-Edu timeout → increased `HF_HUB_DOWNLOAD_TIMEOUT=360`
- PyTorch cu128 needed `--no-cache` to avoid cached CPU wheel

**Model:**
```
Transformer: 318.3M params (151K Qwen3 vocab = ~117M in embeddings)
Interfaces:  5.8M params (3 layers at positions [3, 7, 11])
Total trainable: 324.1M params
CognitiveState: 0/4096 beliefs, 0/16384 edges, 0/64 goals, β=1.000
```

**Training plan:**
```
Phase 1 (α=0): steps 0-2000 — language foundation, L_token only
Phase 2 (α ramps): steps 2000-3000 — cognitive awakening, L_fe introduced
Phase 3 (α=0.1): steps 3000+ — full training
grad_accum=32, batch_size=8, seq_len=2048
Data: FineWeb-Edu 10BT (70%) + Stack v2 dedup (20%) + synthetic (10%)
```

**Status:** RUNNING — Step 0 completed successfully. Training in progress.

**Start time:** ~4:05 AM UTC, March 28, 2026
**Estimated total time:** ~125 hours (~5 days) on single 3090
**Estimated completion:** ~April 2, 2026

**Issues resolved during launch:**
- `memoria.data` module not in git → fixed (empty `__init__.py` wasn't tracked)
- CUDA not available in WSL2 → fixed (`wsl --shutdown` restart)
- Stack v2 gated → got HF approval, but still timing out → switched to starcoderdata (ungated)
- FineWeb first shard download slow → added prefetch with progress message
- DataParallel crash (`compute_loss` not on DP wrapper) → reverted to single GPU for v1
- `save_checkpoint` / `_push_to_hub` had `base_model` NameError → fixed scope

---

## Expected Results

### Phase 1: steps 0-2000 (~50 hours) — Language Foundation (α=0)

| Metric | Expected | Failure signal |
|---|---|---|
| L_token | 12 → ~4-5 (normal LM curve) | Stalls above 6 or explodes |
| Active beliefs | ~0 (no L_fe pressure) | N/A |
| β | ~1.0 (no structure yet) | N/A |

**Checkpoint at step 1000:** verify L_token is decreasing normally. If yes, architecture is compatible with standard LM training.

### Phase 2: steps 2000-3000 (~25 hours) — Cognitive Awakening (α ramps 0→0.1)

| Metric | Expected | Failure signal |
|---|---|---|
| L_token | continues to ~3.5-4 | Spikes or diverges when α turns on |
| L_fe | starts high, decreases | Stays flat or increases |
| Active beliefs | 0 → growing (tens to hundreds) | Stays at 0 |
| Active edges | 0 → forming from Hebbian | Stays at 0 |
| β | starts dropping below 1.0 | Stays exactly 1.0 |

**THE CRITICAL PHASE.** If L_token doesn't degrade when L_fe turns on, the core hypothesis holds. If beliefs start accumulating with non-uniform precision, the polar representation works.

### Phase 3: steps 3000-5000 (~50 hours) — Full Training (α=0.1)

| Metric | Expected | Failure signal |
|---|---|---|
| L_token | ~3.5, competitive with baseline | Significantly worse than same-size transformer |
| L_fe | stable, low | Oscillating or increasing |
| Active beliefs | 200-500+ | Plateaus below 50 |
| Active edges | 100-1000+ | None forming |
| β | settles 0.3-0.7 | Stays near 1.0 or drops to 0 |
| Goals | maybe 1-5 intrinsic goals | OK if 0 (may need more steps) |
| Surprise | accumulating and resolving | Always 0 |

### Success criteria (end of run):
```
✅ L_token curve looks like normal LM training (~3.5)
✅ L_fe decreased (beliefs became consistent)
✅ Active beliefs > 100 (state is being used)
✅ Active edges > 50 (structure forming)
✅ β < 0.8 (state has developed some confidence)
✅ No training instability (no NaN, no loss explosion)
```

### Failure criteria (stop and debug):
```
❌ L_token explodes when α > 0 → L_fe gradient is too strong, reduce α_max
❌ L_token significantly worse than baseline → state interface hurting, check read/write paths
❌ Active beliefs stays 0 → write path not producing meaningful candidates
❌ NaN in loss → numerical issue in free energy or polar representation
❌ OOM → reduce batch size or sequence length
```

---

## Monitoring

**wandb:** https://wandb.ai/subh03/memoria_prototype
**Run:** breezy-field-9 (ybqh2qb1)

Key wandb panels to watch:
- `loss` — overall training loss (should decrease smoothly)
- `loss_token` — language modeling loss (should look like normal LM training)
- `loss_fe` — free energy loss (should decrease once α > 0)
- `alpha` — should be 0 for steps 0-2000, ramp to 0.1 by step 3000
- `beta` — should drop below 1.0 in Phase 2+
- `active_beliefs` — should grow in Phase 2+
- `active_edges` — should grow in Phase 2+
- `active_goals` — may appear in Phase 3
- `dt_ms` — time per step (should be ~90-120s)

---

## Issues to watch

- [ ] Does L_token converge normally during Phase 1?
- [ ] Does L_fe destabilize training when α ramps in Phase 2?
- [ ] Do beliefs actually accumulate (active count > 0)?
- [ ] Does β drop below 1.0 (state developing structure)?
- [ ] Does the model generate intrinsic goals (Phase 3)?
- [ ] Memory usage stable on 3090 (should be <12GB)?
- [ ] HF streaming stable over 5 days (no timeouts/crashes)?
- [ ] Checkpoints saving correctly (step_1000.pt, step_2000.pt, etc.)?
