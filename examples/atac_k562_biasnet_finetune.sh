#!/usr/bin/env bash
# BiasNet fine-tuning example: kidney pretrained -> K562 ATAC-seq.
#
# Demonstrates warm-starting a BiasNet from pretrained weights trained
# on a different tissue (kidney scATAC-seq) and fine-tuning on K562
# ATAC-seq negative peaks. This tests whether Tn5 bias models transfer
# across datasets or need tissue-specific training.
#
# Usage:
#   bash examples/atac_k562_biasnet_finetune.sh
#   bash examples/atac_k562_biasnet_finetune.sh --precision full --num-workers 0

set -euo pipefail

# --- Variables ---
DATA_DIR="tests/data"
K562_DIR="${DATA_DIR}/k562_chrombpnet"
OUTPUT_DIR="${DATA_DIR}/models/atac_k562_biasnet_finetune"
PRETRAINED="pretrained/biasnet/scatac_kidney.pt"
BATCH_SIZE=32
MAX_EPOCHS=50

# --- Download dataset if needed ---
python -c "from cerberus.download import download_dataset; download_dataset('${DATA_DIR}', 'k562_chrombpnet')"

BIGWIG="${K562_DIR}/unstranded.bw"
PEAKS="${K562_DIR}/peaks.bed"

# --- Fine-tune ---
python tools/train_biasnet.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --pretrained "${PRETRAINED}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    "$@"
