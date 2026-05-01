#!/usr/bin/env bash
# ChromBPNet training example for scATAC-seq pseudobulk data.
#
# Stage 2 of the ChromBPNet workflow: loads the stage-1 small-BPNet bias
# checkpoint, freezes it, optionally adjusts the bias-count offset on
# non-peak regions, trains the full ChromBPNet model on the kidney bulk
# pseudobulk BigWig produced by the Cerberus pseudobulk pipeline, then writes
# training curves and held-out prediction-evaluation outputs by default.
# The pseudobulk outputs must already exist in DATA_DIR
# (see examples/scatac_kidney_pseudobulk.sh to generate them).
#
# Prerequisite:
#   bash examples/scatac_kidney_chrombpnet_bias.sh
#
# Usage:
#   bash examples/scatac_kidney_chrombpnet.sh

set -euo pipefail

DATA_DIR="tests/data"
PSEUDOBULK_DIR="${DATA_DIR}/scatac_kidney_pseudobulk"
BIAS_MODEL="${DATA_DIR}/models/scatac_kidney_chrombpnet_bias/single-fold/fold_0/model.pt"
OUTPUT_DIR="${DATA_DIR}/models/scatac_kidney_chrombpnet"

BIGWIG="${PSEUDOBULK_DIR}/bulk.bw"
PEAKS="${PSEUDOBULK_DIR}/bulk_merge.narrowPeak.bed.gz"

python tools/train_chrombpnet.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --pretrained-bias "${BIAS_MODEL}" \
    --adjust-bias-logcounts \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --max-epochs 50 \
    "$@"
