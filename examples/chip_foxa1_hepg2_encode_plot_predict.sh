#!/usr/bin/env bash
# Plot + prediction export for HepG2 ENCODE FOXA1 models trained by:
#   examples/chip_foxa1_hepg2_encode_bpnet_pomeranian.sh
#
# Exports two prediction sets per model on the test split:
#   1) Peaks only
#   2) Peaks + background negatives (--include-background)
#
# Usage:
#   bash examples/chip_foxa1_hepg2_encode_plot_predict.sh
#   bash examples/chip_foxa1_hepg2_encode_plot_predict.sh --multi

set -euo pipefail

DATA_DIR="data/ENCODE_FOXA1_HepG2"
PREFIX="encode_foxa1_hepg2_rep1"
BIGWIG="${DATA_DIR}/${PREFIX}.counts.bw"
PEAKS="${DATA_DIR}/${PREFIX}.peaks.bed.gz"

BP_OUTPUT_ROOT="data/models/chip_foxa1_hepg2_encode_rep1_bpnet_residual_architectures"
POM_OUTPUT_DIR="data/models/chip_foxa1_hepg2_encode_rep1_pomeranian"

SEED=1234
BATCH_SIZE=128
DEVICE="cuda"

RESIDUAL_ARCHITECTURES=(
    "residual_post-activation_conv"
    "residual_pre-activation_conv"
    "activated_residual_pre-activation_conv"
)

TRAINING_MODES=(
    "default"
    "stable"
)

FOLD="single-fold"
for arg in "$@"; do
    if [[ "${arg}" == "--multi" ]]; then
        FOLD="multi-fold"
    else
        echo "Unsupported argument: ${arg}" >&2
        echo "Only --multi is supported." >&2
        exit 1
    fi
done

[[ -s "${BIGWIG}" ]] || { echo "Missing BigWig: ${BIGWIG}" >&2; exit 1; }
[[ -s "${PEAKS}" ]] || { echo "Missing peaks BED: ${PEAKS}" >&2; exit 1; }

# --- BPNet residual architecture sweep outputs ---
for RESIDUAL_ARCHITECTURE in "${RESIDUAL_ARCHITECTURES[@]}"; do
    for TRAINING_MODE in "${TRAINING_MODES[@]}"; do
        MODEL_DIR="${BP_OUTPUT_ROOT}/${RESIDUAL_ARCHITECTURE}/${TRAINING_MODE}/${FOLD}"
        [[ -d "${MODEL_DIR}" ]] || {
            echo "Missing model directory: ${MODEL_DIR}" >&2
            echo "Train first with examples/chip_foxa1_hepg2_encode_bpnet_pomeranian.sh" >&2
            exit 1
        }

        echo "=== Plot + predict BPNet: ${RESIDUAL_ARCHITECTURE}, mode: ${TRAINING_MODE} ==="

        python tools/plot_training_results.py "${MODEL_DIR}"

        # Test peaks only
        python tools/export_predictions.py \
            "${MODEL_DIR}" \
            "${PEAKS}" \
            "${BIGWIG}" \
            --output "${MODEL_DIR}/predictions_test_peaks.tsv.gz" \
            --batch_size "${BATCH_SIZE}" \
            --device "${DEVICE}" \
            --seed "${SEED}" \
            --eval-split test

        # Test peaks + negatives/background
        python tools/export_predictions.py \
            "${MODEL_DIR}" \
            "${PEAKS}" \
            "${BIGWIG}" \
            --output "${MODEL_DIR}/predictions_test_with_background.tsv.gz" \
            --batch_size "${BATCH_SIZE}" \
            --device "${DEVICE}" \
            --seed "${SEED}" \
            --eval-split test \
            --include-background
    done
done

# --- Pomeranian output ---
POM_MODEL_DIR="${POM_OUTPUT_DIR}/${FOLD}"
[[ -d "${POM_MODEL_DIR}" ]] || {
    echo "Missing model directory: ${POM_MODEL_DIR}" >&2
    echo "Train first with examples/chip_foxa1_hepg2_encode_bpnet_pomeranian.sh" >&2
    exit 1
}

echo "=== Plot + predict Pomeranian ==="

python tools/plot_training_results.py "${POM_MODEL_DIR}"

# Test peaks only
python tools/export_predictions.py \
    "${POM_MODEL_DIR}" \
    "${PEAKS}" \
    "${BIGWIG}" \
    --output "${POM_MODEL_DIR}/predictions_test_peaks.tsv.gz" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --eval-split test

# Test peaks + negatives/background
python tools/export_predictions.py \
    "${POM_MODEL_DIR}" \
    "${PEAKS}" \
    "${BIGWIG}" \
    --output "${POM_MODEL_DIR}/predictions_test_with_background.tsv.gz" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --eval-split test \
    --include-background
