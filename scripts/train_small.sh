#!/bin/bash
# Quick training on single 3090 — small config (~125M params)
# Expected: 1B tokens ≈ 6 hours on 2x 3090

source .venv/bin/activate
torchrun --nproc_per_node=2 -m memoria train --config small --max-steps 5000 
