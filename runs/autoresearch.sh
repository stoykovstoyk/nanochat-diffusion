#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

# Quick sanity check
$PYTHON -c "import torch; print(f'torch {torch.__version__} (pre-check OK)')"

# Run the benchmark
START_TIME=$(date +%s%N)

# Run with small config on CPU for fast iteration
# 100 iterations, no checkpoint saving, no eval to keep benchmark fast
cd "$PROJECT_DIR"
$PYTHON -m scripts.diffusion_train \
    --device-type cpu \
    --depth 8 \
    --aspect-ratio 64 \
    --max-seq-len 256 \
    --device-batch-size 8 \
    --lr 4e-4 \
    --warmup-iters 10 \
    --num-diffusion-steps 100 \
    --max-mask-ratio 0.8 \
    --num-iterations 100 \
    --save-every 10000 \
    --eval-iters 10000 \
    --run "autoresearch" \
    2>&1

EXIT_CODE=$?
END_TIME=$(date +%s%N)

# Calculate duration
DURATION=$(echo "scale=3; ($END_TIME - $START_TIME) / 1000000000" | bc)
echo "METRIC training_time_s=${DURATION}s"
echo "METRIC exit_code=${EXIT_CODE}"

exit $EXIT_CODE
