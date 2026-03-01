#!/usr/bin/env bash
# Pomeranian training example for ChIP-seq data using Cerberus.
#
# Trains Pomeranian on the TC32 PROX1 dataset (hg38).
# Genome reference is downloaded automatically to DATA_DIR.
# Set BIGWIG and PEAKS to point to your local files before running.
#
# Usage:
#   bash examples/chip_prox1_tc32_pomeranian.sh
#   bash examples/chip_prox1_tc32_pomeranian.sh --k5
#   bash examples/chip_prox1_tc32_pomeranian.sh --multi

set -euo pipefail

# --- Variables ---
DATA_DIR="tests/data"
OUTPUT_DIR="tests/data/models/chip_prox1_tc32_pomeranian"
BIGWIG="tmp/SI_38975_TC32_DMSO_PROX1.bw"
PEAKS="tmp/SI_38975_TC32_DMSO_PROX1.narrowPeak.gz"
BATCH_SIZE=32
MAX_EPOCHS=50
ALPHA=1.0
LOSS="bpnet"

# --- Train ---
python tools/train_pomeranian.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --alpha "${ALPHA}" \
    --loss "${LOSS}" \
    "$@"
