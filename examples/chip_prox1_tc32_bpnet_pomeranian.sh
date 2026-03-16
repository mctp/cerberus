#!/usr/bin/env bash
# TC32 PROX1 training sweep for Cerberus.
#
# This script trains:
#   1) BPNet across all residual architectures in default and stable modes.
#   2) Pomeranian on the same input data.
#
# Usage:
#   bash examples/chip_prox1_tc32_bpnet_pomeranian.sh
#   bash examples/chip_prox1_tc32_bpnet_pomeranian.sh --multi

set -euo pipefail

BIGWIG="test_pomeranian/data/SI_38975_TC32_DMSO_PROX1.bw"
PEAKS="test_pomeranian/data/SI_38975_TC32_DMSO_PROX1NarrowPeakNoBL.narrowpeak"

FASTA="data/genome/hg38/hg38.fa"
BLACKLIST="data/genome/hg38/blacklist.bed"
GAPS="data/genome/hg38/gaps.bed"

BP_OUTPUT_ROOT="data/models/chip_prox1_tc32_bpnet_residual_architectures"
POM_OUTPUT_DIR="data/models/chip_prox1_tc32_pomeranian"

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

# Shared args supported by both tools.
COMMON_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--multi" ]]; then
        COMMON_ARGS+=("${arg}")
    else
        echo "Unsupported argument for this combined example: ${arg}" >&2
        echo "Only --multi is supported." >&2
        exit 1
    fi
done

[[ -s "${BIGWIG}" ]] || { echo "Missing BigWig: ${BIGWIG}" >&2; exit 1; }
[[ -s "${PEAKS}" ]] || { echo "Missing peaks BED: ${PEAKS}" >&2; exit 1; }

# --- Train BPNet residual architecture sweep (default + stable) ---
for RESIDUAL_ARCHITECTURE in "${RESIDUAL_ARCHITECTURES[@]}"; do
    for TRAINING_MODE in "${TRAINING_MODES[@]}"; do
        OUTPUT_DIR="${BP_OUTPUT_ROOT}/${RESIDUAL_ARCHITECTURE}/${TRAINING_MODE}"

        STABLE_ARGS=()
        if [[ "${TRAINING_MODE}" == "stable" ]]; then
            STABLE_ARGS=(--stable)
        fi

        echo "=== Running BPNet with residual architecture: ${RESIDUAL_ARCHITECTURE}, mode: ${TRAINING_MODE} ==="

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
            "${COMMON_ARGS[@]}"
    done
done

# --- Train Pomeranian ---
python tools/train_pomeranian.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --fasta "${FASTA}" \
    --blacklist "${BLACKLIST}" \
    --gaps "${GAPS}" \
    --output-dir "${POM_OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --alpha "${ALPHA}" \
    --loss "${LOSS}" \
    --seed "${SEED}" \
    "${COMMON_ARGS[@]}"
