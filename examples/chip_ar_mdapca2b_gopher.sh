#!/usr/bin/env bash
# Gopher (GlobalProfileCNN) training example for ChIP-seq data using Cerberus.
#
# Trains Gopher on the MDA-PCA-2b AR dataset (hg38).
# Dataset is downloaded automatically to DATA_DIR.
#
# Usage:
#   bash examples/chip_ar_mdapca2b_gopher.sh
#   bash examples/chip_ar_mdapca2b_gopher.sh --multi

set -euo pipefail

# --- Variables ---
DATA_DIR="tests/data"
OUTPUT_DIR="tests/data/models/chip_ar_mdapca2b_gopher"
BATCH_SIZE=32
MAX_EPOCHS=50

# --- Download dataset ---
python - <<EOF
from cerberus.download import download_dataset
from pathlib import Path
download_dataset(Path("${DATA_DIR}") / "dataset", name="mdapca2b_ar")
EOF

BIGWIG="${DATA_DIR}/dataset/mdapca2b_ar/mdapca2b-ar.bigwig"
PEAKS="${DATA_DIR}/dataset/mdapca2b_ar/mdapca2b-ar.narrowPeak.gz"

# --- Train ---
python tools/train_gopher.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    "$@"
