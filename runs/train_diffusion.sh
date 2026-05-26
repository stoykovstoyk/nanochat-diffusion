#!/bin/bash
# Diffusion LLM Training Run Script
# Adapted from karpathy/nanochat/runs/speedrun.sh

set -e

# Model config
DEPTH=${DEPTH:-8}
ASPECT_RATIO=${ASPECT_RATIO:-64}
N_EMBD=$((DEPTH * ASPECT_RATIO))
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
WINDOW_PATTERN=${WINDOW_PATTERN:-SSSL}

# Training config
NUM_ITERATIONS=${NUM_ITERATIONS:--1}
TARGET_FLOPS=${TARGET_FLOPS:--1.0}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}
LR=${LR:-4e-4}
WARMUP_ITERS=${WARMUP_ITERS:-50}
WARMUP_ITERS=100

# Diffusion-specific config
NUM_DIFFUSION_STEPS=${NUM_DIFFUSION_STEPS:-1000}
MAX_MASK_RATIO=${MAX_MASK_RATIO:-0.8}
NOISE_SCHEDULE=${NOISE_SCHEDULE:-linear}
UNK_TOKEN_ID=${UNK_TOKEN_ID:-32767}
VOCAB_SIZE=${VOCAB_SIZE:-32768}
COMPILE=${COMPILE:-true}

echo "=========================================="
echo "Diffusion LLM Training"
echo "=========================================="
echo "Depth: $DEPTH"
echo "Model dim: $N_EMBD"
echo "Seq len: $MAX_SEQ_LEN"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "LR: $LR"
echo "Num diffusion steps: $NUM_DIFFUSION_STEPS"
echo "Max mask ratio: $MAX_MASK_RATIO"
echo "Noise schedule: $NOISE_SCHEDULE"
echo "=========================================="

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.diffusion_train -- \
    --depth=$DEPTH \
    --aspect-ratio=$ASPECT_RATIO \
    --max-seq-len=$MAX_SEQ_LEN \
    --window-pattern=$WINDOW_PATTERN \
    --device-batch-size=$DEVICE_BATCH_SIZE \
    --lr=$LR \
    --warmup-iters=$WARMUP_ITERS \
    --compile \
    --num-diffusion-steps=$NUM_DIFFUSION_STEPS \
    --max-mask-ratio=$MAX_MASK_RATIO \
    --noise-schedule=$NOISE_SCHEDULE \
    --unk-token-id=$UNK_TOKEN_ID \
    --vocab-size=$VOCAB_SIZE \
    --num-iterations=$NUM_ITERATIONS \
    --target-flops=$TARGET_FLOPS \
    --save-every=1000 \
    --eval-iters=100 \
    --run="diffusion_speedrun"

echo "Training complete!"
