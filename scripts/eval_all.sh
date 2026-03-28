#!/bin/bash
# Run all evaluations on a checkpoint
# Usage: ./scripts/eval_all.sh checkpoints/final.pt

CHECKPOINT=${1:-checkpoints/final.pt}
source .venv/bin/activate

mkdir -p results
python -m memoria eval "$CHECKPOINT" --suite all
