#!/usr/bin/env bash
# Pomeranian training example for ChIP-seq data using Cerberus.
#
# Trains Pomeranian on the 22Rv1 FOXA1 dataset (hg38).
# Uses local genome files from data/genome/hg38.
#
# Usage:
#   bash examples/chip_foxa1_22rv1_pomeranian.sh
#   bash examples/chip_foxa1_22rv1_pomeranian.sh --k5
#   bash examples/chip_foxa1_22rv1_pomeranian.sh --multi

set -euo pipefail

# --- Variables ---
OUTPUT_DIR="data/models/chip_foxa1_22rv1_pomeranian"
BIGWIG="data/FOXA1_22Rv1/GSM3508093_SI_19463.bw"
PEAKS="data/FOXA1_22Rv1/GSM3508093_SI_19463-macs2.bed.gz"
# Cross-dataset: LNCaP (same TF, different cell line)
LNCAP_BIGWIG="data/FOXA1_LNCAP/GSM3508079_SI_16486.bw"
LNCAP_PEAKS="data/FOXA1_LNCAP/GSM3508079_SI_16486-macs2.bed.gz"
FASTA="data/genome/hg38/hg38.fa"
BLACKLIST="data/genome/hg38/blacklist.bed"
GAPS="data/genome/hg38/gaps.bed"
BATCH_SIZE=32
MAX_EPOCHS=50
ALPHA=adaptive
LOSS="bpnet"

# Determine fold subdirectory based on --multi flag
FOLD="single-fold"
for arg in "$@"; do
    [[ "$arg" == "--multi" ]] && FOLD="multi-fold"
done
MODEL_DIR="${OUTPUT_DIR}/${FOLD}"

# --- Train ---
python tools/train_pomeranian.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --fasta "${FASTA}" \
    --blacklist "${BLACKLIST}" \
    --gaps "${GAPS}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --alpha "${ALPHA}" \
    --loss "${LOSS}" \
    "$@"

# --- Plot ---
python tools/plot_training_results.py "${MODEL_DIR}"

# --- Predict (in-sample: 22Rv1) ---
python tools/export_predictions.py \
    "${MODEL_DIR}" \
    "${PEAKS}" \
    "${BIGWIG}" \
    --output "${MODEL_DIR}/predictions.tsv.gz" \
    --batch_size 128 \
    --device cuda

# --- Predict (cross-dataset: LNCaP — same TF, different cell line) ---
# High performance here suggests the model learned FOXA1 binding, not 22Rv1-specific chromatin.
python tools/export_predictions.py \
    "${MODEL_DIR}" \
    "${LNCAP_PEAKS}" \
    "${LNCAP_BIGWIG}" \
    --output "${MODEL_DIR}/predictions_lncap.tsv.gz" \
    --batch_size 128 \
    --device cuda
