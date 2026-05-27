#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

# Ensure GPU torch is installed
cd "$PROJECT_DIR"
$PYTHON -c "
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA not available — need uv sync --extra gpu')
print(f'torch {torch.__version__}, device: {torch.cuda.get_device_name(0)}')
" 2>&1

# Run the benchmark
START_TIME=$(date +%s%N)

$PYTHON -m scripts.diffusion_train \
    --device-type cuda \
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

DURATION=$(echo "scale=3; ($END_TIME - $START_TIME) / 1000000000" | bc)
echo "METRIC training_time_s=${DURATION}s"
echo "METRIC exit_code=${EXIT_CODE}"

exit $EXIT_CODE
