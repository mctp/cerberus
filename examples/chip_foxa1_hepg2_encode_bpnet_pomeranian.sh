#!/usr/bin/env bash
# ENCODE FOXA1 HepG2 preprocessing + training example for Cerberus.
#
# This script:
#   1) Converts ENCODE replicate 1 BAM + BED to Cerberus-ready inputs.
#   2) Trains BPNet across all residual architectures and default/stable modes.
#   3) Trains Pomeranian on the same replicate 1 inputs.
#
# Usage:
#   bash examples/chip_foxa1_hepg2_encode_bpnet_pomeranian.sh
#   bash examples/chip_foxa1_hepg2_encode_bpnet_pomeranian.sh --multi

set -euo pipefail

DATA_DIR="data/ENCODE_FOXA1_HepG2"
PREPROCESS_SCRIPT="tools/preprocess_bam_bed_for_cerberus.sh"
PREFIX="encode_foxa1_hepg2_rep1"

# ENCODE files for FOXA1 HepG2 (replicate 1 BAM + peaks BED)
BAM_REP1="${DATA_DIR}/ENCFF280BKL.bam"
BED_REP1="${DATA_DIR}/ENCFF658EER.bed.gz"

BIGWIG="${DATA_DIR}/${PREFIX}.counts.bw"
PEAKS="${DATA_DIR}/${PREFIX}.peaks.bed.gz"

FASTA="data/genome/hg38/hg38.fa"
BLACKLIST="data/genome/hg38/blacklist.bed"
GAPS="data/genome/hg38/gaps.bed"

BP_OUTPUT_ROOT="data/models/chip_foxa1_hepg2_encode_rep1_bpnet_residual_architectures"
POM_OUTPUT_DIR="data/models/chip_foxa1_hepg2_encode_rep1_pomeranian"

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

# --- Preprocess BAM + BED into Cerberus inputs ---
bash "${PREPROCESS_SCRIPT}" \
    --bam "${BAM_REP1}" \
    --bed "${BED_REP1}" \
    --out-dir "${DATA_DIR}" \
    --prefix "${PREFIX}" \
    --threads 8 \
    --mapq 30 \
    --ignore-duplicates \
    --blacklist "${BLACKLIST}"

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
