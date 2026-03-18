#!/usr/bin/env bash
# Evaluate FOXA1 22Rv1-trained models on LNCaP shared/unique peak subsets.
# Exports test-split comparison metrics (peaks-only + peaks/background negatives)
# and generates predicted BigWigs for one test chromosome.
#
# Usage:
#   bash examples/chip_foxa1_22rv1_shared_unique_predict_plot.sh
#   bash examples/chip_foxa1_22rv1_shared_unique_predict_plot.sh --device cpu
#   bash examples/chip_foxa1_22rv1_shared_unique_predict_plot.sh --test-chrom chr1

set -euo pipefail

# --- Defaults ---
BP_MODEL_DIR="data/models/chip_foxa1_22rv1_bpnet_residual_architectures/residual_pre-activation_conv/default/single-fold"
POM_MODEL_DIR="data/models/chip_foxa1_22rv1_pomeranian/single-fold"

OBS_BW="data/FOXA1_LNCAP/GSM3508079_SI_16486.bw"
SHARED_PEAKS="data/FOXA1_22Rv1_vs_LNCAP/LNCaP.shared.bed.gz"
UNIQUE_PEAKS="data/FOXA1_22Rv1_vs_LNCAP/LNCaP.unique.bed.gz"

OUT_DIR="data/FOXA1_22Rv1_vs_LNCAP/model_comparison_with_negatives"
DEVICE="cuda"
BATCH_SIZE=128
SEED=1234
TEST_CHROM=""
BIGWIG_BATCH_SIZE=64
BIGWIG_STRIDE=500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${OUT_DIR}"

[[ -d "${BP_MODEL_DIR}" ]] || { echo "Missing BPNet model dir: ${BP_MODEL_DIR}" >&2; exit 1; }
[[ -d "${POM_MODEL_DIR}" ]] || { echo "Missing Pomeranian model dir: ${POM_MODEL_DIR}" >&2; exit 1; }
[[ -s "${OBS_BW}" ]] || { echo "Missing observed BigWig: ${OBS_BW}" >&2; exit 1; }
[[ -s "${SHARED_PEAKS}" ]] || { echo "Missing shared peaks: ${SHARED_PEAKS}" >&2; exit 1; }
[[ -s "${UNIQUE_PEAKS}" ]] || { echo "Missing unique peaks: ${UNIQUE_PEAKS}" >&2; exit 1; }

run_export() {
    local model_dir="$1"
    local peaks="$2"
    local run_prefix="$3"

    # Peaks only (test chromosomes only)
    python tools/export_predictions.py \
        "${model_dir}" \
        "${peaks}" \
        "${OBS_BW}" \
        --output "${OUT_DIR}/${run_prefix}.tsv.gz" \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --eval-split test

    # Peaks + negatives/background (test chromosomes only)
    python tools/export_predictions.py \
        "${model_dir}" \
        "${peaks}" \
        "${OBS_BW}" \
        --output "${OUT_DIR}/${run_prefix}_with_negatives.tsv.gz" \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --eval-split test \
        --include-background
}

echo "=== Export log-count predictions (shared/unique peaks, test split, with and without negatives) ==="
run_export "${BP_MODEL_DIR}" "${SHARED_PEAKS}" "bpnet_predictions_lncap_shared"
run_export "${BP_MODEL_DIR}" "${UNIQUE_PEAKS}" "bpnet_predictions_lncap_unique"
run_export "${POM_MODEL_DIR}" "${SHARED_PEAKS}" "pomeranian_predictions_lncap_shared"
run_export "${POM_MODEL_DIR}" "${UNIQUE_PEAKS}" "pomeranian_predictions_lncap_unique"

echo "=== Summarize metrics ==="
python - <<'PY'
import json
from pathlib import Path

out_dir = Path("data/FOXA1_22Rv1_vs_LNCAP/model_comparison")
metric_files = [
    out_dir / "bpnet_predictions_lncap_shared.metrics.json",
    out_dir / "bpnet_predictions_lncap_shared_with_negatives.metrics.json",
    out_dir / "bpnet_predictions_lncap_unique.metrics.json",
    out_dir / "bpnet_predictions_lncap_unique_with_negatives.metrics.json",
    out_dir / "pomeranian_predictions_lncap_shared.metrics.json",
    out_dir / "pomeranian_predictions_lncap_shared_with_negatives.metrics.json",
    out_dir / "pomeranian_predictions_lncap_unique.metrics.json",
    out_dir / "pomeranian_predictions_lncap_unique_with_negatives.metrics.json",
]

rows = []
for p in metric_files:
    if not p.exists():
        continue
    m = json.loads(p.read_text())["metrics"]
    rows.append(
        (
            p.stem.replace(".metrics", ""),
            m["pearson"],
            m["pearson_log_counts"],
            m["mse_profile"],
            m["mse_log_counts"],
            m["loss"],
        )
    )

summary_path = out_dir / "metrics_summary.tsv"
with summary_path.open("w") as handle:
    handle.write("run\tpearson_profile\tpearson_log_counts\tmse_profile\tmse_log_counts\tloss\n")
    for row in rows:
        handle.write(
            f"{row[0]}\t{row[1]:.6f}\t{row[2]:.6f}\t{row[3]:.6f}\t{row[4]:.6f}\t{row[5]:.6f}\n"
        )

print(f"Wrote {summary_path}")
for row in rows:
    print(
        f"{row[0]}: pearson={row[1]:.4f}, "
        f"pearson_log_counts={row[2]:.4f}, mse_log_counts={row[4]:.4f}"
    )
PY

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
OUT_DIR = Path("${OUT_DIR}")


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


def export_bw(model_dir: str, out_prefix: str, batch_size: int, stride: int, requested_test_chrom: str):
    device = torch.device(DEVICE if DEVICE else ("cuda" if torch.cuda.is_available() else "cpu"))
    ensemble = ModelEnsemble(model_dir, device=device, search_paths=[Path.cwd(), Path("tests/data")])
    cfg = ensemble.cerberus_config

    test_chrom = resolve_test_chrom(cfg["genome_config"], requested_test_chrom)
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

    out_bw = OUT_DIR / f"{out_prefix}.test_{test_chrom}.pred.bw"
    predict_to_bigwig(
        output_path=out_bw,
        dataset=dataset,
        model_ensemble=ensemble,
        stride=stride,
        use_folds=["test"],
        aggregation="model",
        batch_size=batch_size,
    )
    return test_chrom, out_bw


bp_chrom, bp_bw = export_bw(
    "${BP_MODEL_DIR}",
    "bpnet_residual_preact",
    batch_size=${BIGWIG_BATCH_SIZE},
    stride=${BIGWIG_STRIDE},
    requested_test_chrom=REQUESTED_TEST_CHROM,
)
pom_chrom, pom_bw = export_bw(
    "${POM_MODEL_DIR}",
    "pomeranian",
    batch_size=${BIGWIG_BATCH_SIZE},
    stride=${BIGWIG_STRIDE},
    requested_test_chrom=REQUESTED_TEST_CHROM,
)

print(f"BPNet test chromosome: {bp_chrom}")
print(f"Pomeranian test chromosome: {pom_chrom}")
print("Predicted BigWigs written:")
print(f"  {bp_bw}")
print(f"  {pom_bw}")
PY

echo "=== Done ==="
echo "Outputs:"
echo "  Metrics TSV: ${OUT_DIR}/metrics_summary.tsv"
echo "  Prediction TSVs (including negatives): ${OUT_DIR}/*.tsv.gz"
echo "  Predicted BigWigs (single test chromosome): ${OUT_DIR}/*.pred.bw"
