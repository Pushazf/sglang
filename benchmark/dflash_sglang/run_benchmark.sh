#!/usr/bin/env bash
set -euo pipefail

# ── Run the DFlash sglang benchmark ──
python benchmark_sglang.py \
    --dataset-name mt-bench \
    --tp-size 1 \
    --skip-baseline \
    --concurrencies 32 \
    --disable-radix-cache \
    --attention-backends flashinfer \
    --target-model Qwen/Qwen3.5-4B \
    --draft-model z-lab/Qwen3.5-4B-DFlash \
    --output-md results/dflash_sglang_gsm8k.md \
    "$@"
