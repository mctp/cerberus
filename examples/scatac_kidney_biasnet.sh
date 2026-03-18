#!/usr/bin/env bash
# BiasNet training example for scATAC-seq pseudobulk data using Cerberus.
#
# Trains a standalone BiasNet (Tn5 bias model) on the kidney scATAC-seq
# bulk pseudobulk dataset (hg38, all cell types merged). Trains on
# negative (non-peak) regions only by default, matching ChromBPNet's
# bias-model training paradigm.
#
# The pseudobulk BigWig and peaks must already exist in DATA_DIR
# (see examples/scatac_kidney_pseudobulk.sh to generate).
#
# Usage:
#   bash examples/scatac_kidney_biasnet.sh
#   bash examples/scatac_kidney_biasnet.sh --multi
#   bash examples/scatac_kidney_biasnet.sh --precision full --num-workers 0

set -euo pipefail

# --- Variables ---
DATA_DIR="tests/data"
PSEUDOBULK_DIR="${DATA_DIR}/scatac_kidney_pseudobulk"
OUTPUT_DIR="${DATA_DIR}/models/scatac_kidney_biasnet"
BATCH_SIZE=32
MAX_EPOCHS=50

BIGWIG="${PSEUDOBULK_DIR}/bulk.bw"
PEAKS="${PSEUDOBULK_DIR}/bulk_merge.narrowPeak.bed.gz"

# --- Train ---
python tools/train_biasnet.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    "$@"
