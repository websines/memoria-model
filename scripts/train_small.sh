#!/bin/bash
# Quick training on single 3090 — small config (~125M params)
# Expected time: 2-4 hours

python -m memoria.training.train \
    --config configs/small.yaml \
    --max-steps 5000 \
    --checkpoint-dir checkpoints/small \
    --log-to-wandb
