#!/usr/bin/env bash
# BPNet residual architecture sweep example for ChIP-seq data using Cerberus.
#
# Trains BPNet on the 22Rv1 FOXA1 dataset (hg38) for all supported residual
# architectures in both standard and --stable modes.
# Uses local genome files from data/genome/hg38.
#
# Usage:
#   bash examples/chip_foxa1_22rv1_bpnet_all_residual_architectures.sh
#   bash examples/chip_foxa1_22rv1_bpnet_all_residual_architectures.sh --multi

set -euo pipefail

# --- Variables ---
OUTPUT_ROOT="data/models/chip_foxa1_22rv1_bpnet_residual_architectures"
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
SEED=1234

RESIDUAL_ARCHITECTURES=(
    "residual_post-activation_conv"
    "residual_pre-activation_conv"
    "activated_residual_pre-activation_conv"
)

TRAINING_MODES=(
    "default"
    "stable"
)

# Determine fold subdirectory based on --multi flag
FOLD="single-fold"
for arg in "$@"; do
    [[ "$arg" == "--multi" ]] && FOLD="multi-fold"
done

for RESIDUAL_ARCHITECTURE in "${RESIDUAL_ARCHITECTURES[@]}"; do
    for TRAINING_MODE in "${TRAINING_MODES[@]}"; do
        OUTPUT_DIR="${OUTPUT_ROOT}/${RESIDUAL_ARCHITECTURE}/${TRAINING_MODE}"
        MODEL_DIR="${OUTPUT_DIR}/${FOLD}"

        STABLE_ARGS=()
        if [[ "${TRAINING_MODE}" == "stable" ]]; then
            STABLE_ARGS=(--stable)
        fi

        echo "=== Running BPNet with residual architecture: ${RESIDUAL_ARCHITECTURE}, mode: ${TRAINING_MODE} ==="

        # --- Train ---
        python tools/train_bpnet.py \
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
            --seed "${SEED}" \
            --residual-architecture "${RESIDUAL_ARCHITECTURE}" \
            "${STABLE_ARGS[@]}" \
            "$@"

        # --- Plot ---
        python tools/plot_training_results.py "${MODEL_DIR}"

        # --- Predict (in-sample: 22Rv1, held-out test chromosomes only) ---
        python tools/export_predictions.py \
            "${MODEL_DIR}" \
            "${PEAKS}" \
            "${BIGWIG}" \
            --output "${MODEL_DIR}/predictions.tsv.gz" \
            --batch_size 128 \
            --device cuda \
            --seed "${SEED}" \
            --eval-split test

        # --- Predict (cross-dataset: LNCaP — same TF, different cell line, test chromosomes only) ---
        # High performance here suggests the model learned FOXA1 binding, not 22Rv1-specific chromatin.
        python tools/export_predictions.py \
            "${MODEL_DIR}" \
            "${LNCAP_PEAKS}" \
            "${LNCAP_BIGWIG}" \
            --output "${MODEL_DIR}/predictions_lncap.tsv.gz" \
            --batch_size 128 \
            --device cuda \
            --seed "${SEED}" \
            --eval-split test
    done
done
