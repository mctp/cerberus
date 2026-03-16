#!/usr/bin/env bash
# Dalmatian training example for K562 ATAC-seq data using Cerberus.
#
# Trains Dalmatian (bias-factorized model) on the K562 ChromBPNet
# benchmark dataset from Zenodo (DOI: 10.5281/zenodo.15713376).
# The dataset (narrowPeak + BigWig) is auto-downloaded on first run.
#
# Usage:
#   bash examples/atac_k562_dalmatian.sh
#   bash examples/atac_k562_dalmatian.sh --multi
#   bash examples/atac_k562_dalmatian.sh --precision full --num-workers 0

set -euo pipefail

# --- Variables ---
DATA_DIR="tests/data"
K562_DIR="${DATA_DIR}/k562_chrombpnet"
OUTPUT_DIR="${DATA_DIR}/models/atac_k562_dalmatian"
BATCH_SIZE=32
MAX_EPOCHS=50

# --- Download dataset if needed ---
python -c "from cerberus.download import download_dataset; download_dataset('${DATA_DIR}', 'k562_chrombpnet')"

BIGWIG="${K562_DIR}/unstranded.bw"
PEAKS="${K562_DIR}/peaks.bed"

# --- Train ---
python tools/train_dalmatian.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --base-loss mse \
    --signal-background-weight 0.01 \
    --bias-weight 0.5 \
    "$@"
