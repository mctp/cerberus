#!/usr/bin/env bash
# FOXA1 22Rv1 exo_WT training + exo_WT/exo_I176M prediction example for Cerberus.
#
# This script:
#   1) Trains BPNet (residual_pre-activation_conv, default mode) on exo_WT.
#   2) Trains default Pomeranian on exo_WT.
#   3) Plots training results for both models.
#   4) Exports test-split predictions for exo_WT and exo_I176M with:
#      - peaks only
#      - peaks + background negatives (--include-background)
#   5) Generates predicted BigWigs for one test chromosome for both models.
#
# Usage:
#   bash examples/chip_foxa1_22rv1_exo_wt_i176m_bpnet_pomeranian.sh
#   bash examples/chip_foxa1_22rv1_exo_wt_i176m_bpnet_pomeranian.sh --multi
#   bash examples/chip_foxa1_22rv1_exo_wt_i176m_bpnet_pomeranian.sh --device cpu
#   bash examples/chip_foxa1_22rv1_exo_wt_i176m_bpnet_pomeranian.sh --test-chrom chr1

set -euo pipefail

# --- Training data: exo_WT ---
WT_BIGWIG="data/FOXA1_22Rv1/exo_WT/GSM3508110_SI_22218.bw"
WT_PEAKS="data/FOXA1_22Rv1/exo_WT/GSM3508110_SI_22218-macs2.bed.gz"

# --- Cross-condition evaluation data: exo_I176M ---
I176M_BIGWIG="data/FOXA1_22Rv1/exo_I176M/GSM3508112_SI_22221.bw"
I176M_PEAKS="data/FOXA1_22Rv1/exo_I176M/GSM3508112_SI_22221-macs2.bed.gz"

# --- Genome reference ---
FASTA="data/genome/hg38/hg38.fa"
BLACKLIST="data/genome/hg38/blacklist.bed"
GAPS="data/genome/hg38/gaps.bed"

# --- Outputs ---
BP_OUTPUT_DIR="data/models/chip_foxa1_22rv1_exo_wt_bpnet_residual_pre-activation_conv_default"
POM_OUTPUT_DIR="data/models/chip_foxa1_22rv1_exo_wt_pomeranian"

# --- Fixed model choice requested ---
RESIDUAL_ARCHITECTURE="residual_pre-activation_conv"

# --- Defaults ---
TRAIN_BATCH_SIZE=32
PREDICT_BATCH_SIZE=128
MAX_EPOCHS=50
ALPHA="adaptive"
LOSS="bpnet"
SEED=1234
DEVICE="cuda"
FOLD="single-fold"
TEST_CHROM=""
BIGWIG_BATCH_SIZE=64
BIGWIG_STRIDE=500
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --multi)
            FOLD="multi-fold"
            TRAIN_ARGS+=(--multi)
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --train-batch-size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --predict-batch-size)
            PREDICT_BATCH_SIZE="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --test-chrom)
            TEST_CHROM="$2"
            shift 2
            ;;
        --bigwig-batch-size)
            BIGWIG_BATCH_SIZE="$2"
            shift 2
            ;;
        --bigwig-stride)
            BIGWIG_STRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unsupported argument: $1" >&2
            echo "Supported args: --multi --device <cuda|cpu|mps> --seed <int> --train-batch-size <int> --predict-batch-size <int> --max-epochs <int> --test-chrom <chrom> --bigwig-batch-size <int> --bigwig-stride <int>" >&2
            exit 1
            ;;
    esac
done

[[ -s "${WT_BIGWIG}" ]] || { echo "Missing exo_WT BigWig: ${WT_BIGWIG}" >&2; exit 1; }
[[ -s "${WT_PEAKS}" ]] || { echo "Missing exo_WT peaks: ${WT_PEAKS}" >&2; exit 1; }
[[ -s "${I176M_BIGWIG}" ]] || { echo "Missing exo_I176M BigWig: ${I176M_BIGWIG}" >&2; exit 1; }
[[ -s "${I176M_PEAKS}" ]] || { echo "Missing exo_I176M peaks: ${I176M_PEAKS}" >&2; exit 1; }
[[ -s "${FASTA}" ]] || { echo "Missing FASTA: ${FASTA}" >&2; exit 1; }
[[ -s "${BLACKLIST}" ]] || { echo "Missing blacklist: ${BLACKLIST}" >&2; exit 1; }
[[ -s "${GAPS}" ]] || { echo "Missing gaps: ${GAPS}" >&2; exit 1; }

echo "=== Train BPNet (residual_pre-activation_conv, default) on exo_WT ==="
python tools/train_bpnet.py \
    --bigwig "${WT_BIGWIG}" \
    --peaks "${WT_PEAKS}" \
    --fasta "${FASTA}" \
    --blacklist "${BLACKLIST}" \
    --gaps "${GAPS}" \
    --output-dir "${BP_OUTPUT_DIR}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --alpha "${ALPHA}" \
    --loss "${LOSS}" \
    --seed "${SEED}" \
    --residual-architecture "${RESIDUAL_ARCHITECTURE}" \
    "${TRAIN_ARGS[@]}"

echo "=== Train Pomeranian (default) on exo_WT ==="
python tools/train_pomeranian.py \
    --bigwig "${WT_BIGWIG}" \
    --peaks "${WT_PEAKS}" \
    --fasta "${FASTA}" \
    --blacklist "${BLACKLIST}" \
    --gaps "${GAPS}" \
    --output-dir "${POM_OUTPUT_DIR}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --alpha "${ALPHA}" \
    --loss "${LOSS}" \
    --seed "${SEED}" \
    "${TRAIN_ARGS[@]}"

BP_MODEL_DIR="${BP_OUTPUT_DIR}/${FOLD}"
POM_MODEL_DIR="${POM_OUTPUT_DIR}/${FOLD}"

[[ -d "${BP_MODEL_DIR}" ]] || { echo "Missing BPNet model directory: ${BP_MODEL_DIR}" >&2; exit 1; }
[[ -d "${POM_MODEL_DIR}" ]] || { echo "Missing Pomeranian model directory: ${POM_MODEL_DIR}" >&2; exit 1; }

echo "=== Plot training results ==="
python tools/plot_training_results.py "${BP_MODEL_DIR}"
python tools/plot_training_results.py "${POM_MODEL_DIR}"

run_exports() {
    local model_dir="$1"
    local peaks="$2"
    local bigwig="$3"
    local prefix="$4"

    # Peaks only
    python tools/export_predictions.py \
        "${model_dir}" \
        "${peaks}" \
        "${bigwig}" \
        --output "${model_dir}/${prefix}_test_peaks.tsv.gz" \
        --batch_size "${PREDICT_BATCH_SIZE}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --eval-split test

    # Peaks + negatives/background
    python tools/export_predictions.py \
        "${model_dir}" \
        "${peaks}" \
        "${bigwig}" \
        --output "${model_dir}/${prefix}_test_with_background.tsv.gz" \
        --batch_size "${PREDICT_BATCH_SIZE}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --eval-split test \
        --include-background
}

echo "=== Export BPNet predictions (exo_WT and exo_I176M; with/without negatives) ==="
run_exports "${BP_MODEL_DIR}" "${WT_PEAKS}" "${WT_BIGWIG}" "predictions_exo_wt"
run_exports "${BP_MODEL_DIR}" "${I176M_PEAKS}" "${I176M_BIGWIG}" "predictions_exo_i176m"

echo "=== Export Pomeranian predictions (exo_WT and exo_I176M; with/without negatives) ==="
run_exports "${POM_MODEL_DIR}" "${WT_PEAKS}" "${WT_BIGWIG}" "predictions_exo_wt"
run_exports "${POM_MODEL_DIR}" "${I176M_PEAKS}" "${I176M_BIGWIG}" "predictions_exo_i176m"

echo "=== Generate predicted BigWigs (single test chromosome) ==="
python - <<PY
from pathlib import Path
import torch
from cerberus.model_ensemble import ModelEnsemble
from cerberus.dataset import CerberusDataset
from cerberus.predict_bigwig import predict_to_bigwig
from cerberus.genome import create_genome_folds

DEVICE = "${DEVICE}"
REQUESTED_TEST_CHROM = "${TEST_CHROM}"
BP_MODEL_DIR = "${BP_MODEL_DIR}"
POM_MODEL_DIR = "${POM_MODEL_DIR}"
BIGWIG_BATCH_SIZE = int("${BIGWIG_BATCH_SIZE}")
BIGWIG_STRIDE = int("${BIGWIG_STRIDE}")


def resolve_test_chrom(genome_config: dict, requested: str) -> str:
    fold_args = genome_config.get("fold_args", {})
    test_fold_idx = fold_args.get("test_fold")
    if test_fold_idx is None:
        raise ValueError(
            "Model genome_config.fold_args is missing 'test_fold'; cannot resolve test chromosome."
        )

    folds = create_genome_folds(
        genome_config["chrom_sizes"],
        genome_config["fold_type"],
        fold_args,
    )
    test_chroms = sorted(folds[test_fold_idx].keys())
    if not test_chroms:
        raise ValueError("No test chromosomes were resolved from model folds.")

    if requested:
        if requested not in test_chroms:
            raise ValueError(
                f"Requested --test-chrom '{requested}' is not a test chromosome. "
                f"Available test chromosomes: {', '.join(test_chroms)}"
            )
        return requested

    return test_chroms[0]


def export_bw(model_dir: str, out_prefix: str):
    device = torch.device(DEVICE if DEVICE else ("cuda" if torch.cuda.is_available() else "cpu"))
    ensemble = ModelEnsemble(model_dir, device=device, search_paths=[Path.cwd(), Path("tests/data")])
    cfg = ensemble.cerberus_config

    test_chrom = resolve_test_chrom(cfg["genome_config"], REQUESTED_TEST_CHROM)
    single_chrom_genome_config = {
        **cfg["genome_config"],
        "allowed_chroms": [test_chrom],
        "chrom_sizes": {test_chrom: cfg["genome_config"]["chrom_sizes"][test_chrom]},
    }

    dataset = CerberusDataset(
        genome_config=single_chrom_genome_config,
        data_config=cfg["data_config"],
        sampler_config=None,
        in_memory=False,
        is_train=False,
    )

    out_bw = Path(model_dir) / f"{out_prefix}.test_{test_chrom}.pred.bw"
    predict_to_bigwig(
        output_path=out_bw,
        dataset=dataset,
        model_ensemble=ensemble,
        stride=BIGWIG_STRIDE,
        use_folds=["test"],
        aggregation="model",
        batch_size=BIGWIG_BATCH_SIZE,
    )
    return test_chrom, out_bw


bp_chrom, bp_bw = export_bw(BP_MODEL_DIR, "predictions_exo_wt_bpnet")
pom_chrom, pom_bw = export_bw(POM_MODEL_DIR, "predictions_exo_wt_pomeranian")

print(f"BPNet test chromosome: {bp_chrom}")
print(f"Pomeranian test chromosome: {pom_chrom}")
print("Predicted BigWigs written:")
print(f"  {bp_bw}")
print(f"  {pom_bw}")
PY

echo "=== Done ==="
echo "BPNet outputs: ${BP_MODEL_DIR}/predictions_exo_{wt,i176m}_test_{peaks,with_background}.tsv.gz"
echo "Pomeranian outputs: ${POM_MODEL_DIR}/predictions_exo_{wt,i176m}_test_{peaks,with_background}.tsv.gz"
echo "Predicted BigWigs: ${BP_MODEL_DIR}/*.pred.bw and ${POM_MODEL_DIR}/*.pred.bw"
