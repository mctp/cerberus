#!/usr/bin/env bash
# Dalmatian training example for scATAC-seq pseudobulk data using Cerberus.
#
# Trains Dalmatian (bias-factorized model) on the kidney scATAC-seq
# bulk pseudobulk dataset (hg38, all cell types merged). Uses the merged
# peak set for sampling. The pseudobulk BigWig and peaks must already
# exist in DATA_DIR (see examples/scatac_kidney_pseudobulk.sh to generate).
#
# Usage:
#   bash examples/scatac_kidney_dalmatian.sh
#   bash examples/scatac_kidney_dalmatian.sh --multi
#   bash examples/scatac_kidney_dalmatian.sh --precision full --num-workers 0

set -euo pipefail

# --- Variables ---
DATA_DIR="tests/data"
PSEUDOBULK_DIR="${DATA_DIR}/scatac_kidney_pseudobulk"
OUTPUT_DIR="${DATA_DIR}/models/scatac_kidney_dalmatian"
BATCH_SIZE=32
MAX_EPOCHS=50

BIGWIG="${PSEUDOBULK_DIR}/bulk.bw"
PEAKS="${PSEUDOBULK_DIR}/bulk_merge.narrowPeak.bed.gz"

# --- Train ---
python tools/train_dalmatian.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --base-loss mse \
    --bias-weight 1.0 \
    "$@"
