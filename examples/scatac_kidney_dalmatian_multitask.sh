#!/usr/bin/env bash
# Multi-task Dalmatian training on 14 kidney cell-type pseudobulk tracks.
#
# Trains a multi-task Dalmatian with shared BiasNet (1 channel, frozen,
# pretrained on bulk) and a 14-channel SignalNet. Uses the pre-merged
# bulk peak set for sampling.
#
# Prerequisite: train BiasNet first (see scatac_kidney_biasnet.sh).
#
# Usage:
#   bash examples/scatac_kidney_dalmatian_multitask.sh
#   bash examples/scatac_kidney_dalmatian_multitask.sh --max-epochs 1 --precision full --num-workers 0

set -euo pipefail

# --- Variables ---
DATA_DIR="tests/data"
PSEUDOBULK_DIR="${DATA_DIR}/scatac_kidney_pseudobulk"
BIAS_MODEL="${DATA_DIR}/models/scatac_kidney_biasnet/single-fold/fold_0/model.pt"
OUTPUT_DIR="${DATA_DIR}/models/scatac_kidney_dalmatian_multitask"

PEAKS="${PSEUDOBULK_DIR}/bulk_merge.narrowPeak.bed.gz"
TARGETS_JSON="examples/scatac_kidney_multitask_targets.json"

# --- Train ---
python tools/train_dalmatian_multitask.py \
    --targets-json "${TARGETS_JSON}" \
    --peaks "${PEAKS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --max-epochs 50 \
    --base-loss mse \
    --bias-weight 0.0 \
    --shared-bias \
    --pretrained-bias "${BIAS_MODEL}" \
    --freeze-bias \
    "$@"
