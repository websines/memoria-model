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

**Status:** RUNNING — data streaming started, waiting for first step output

---

## Issues to watch

- [ ] Does L_token converge normally during Phase 1?
- [ ] Does L_fe destabilize training when α ramps in Phase 2?
- [ ] Do beliefs actually accumulate (active count > 0)?
- [ ] Does β drop below 1.0 (state developing structure)?
- [ ] Does the model generate intrinsic goals (Phase 3)?
- [ ] Memory usage stable on 3090 (should be <12GB)?
- [ ] HF streaming stable over hours (no timeouts/crashes)?
