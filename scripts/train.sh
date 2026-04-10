#!/bin/bash
# Full training pipeline: warm cache → train
# Usage: bash scripts/train.sh [--max-steps 20000] [--config full]

set -e

CONFIG="${CONFIG:-full}"
MAX_STEPS="${MAX_STEPS:-20000}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════"
echo "  Memoria Training Pipeline"
echo "  Config: $CONFIG | Steps: $MAX_STEPS"
echo "═══════════════════════════════════════════"

# Load env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Step 1: Warm HF cache
echo ""
echo "[1/2] Warming HuggingFace dataset cache..."
echo "      (first run takes ~5-10 min, subsequent runs instant)"
echo ""
# Warmup may exit non-zero due to Python thread cleanup bug (harmless).
# The cache is warm regardless — check output for ✓/✗ marks above.
python scripts/warmup_cache.py || true
echo ""
echo "      Cache warmup complete (check marks above for status)."

# Step 2: Train
echo ""
echo "[2/2] Starting training..."
echo "      Config: $CONFIG | Steps: $MAX_STEPS"
echo ""
accelerate launch -m memoria train --config "$CONFIG" --max-steps "$MAX_STEPS"
