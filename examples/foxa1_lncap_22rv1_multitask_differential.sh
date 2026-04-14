#!/usr/bin/env bash
# =============================================================================
# FOXA1 LNCaP vs 22Rv1 — Multitask-Differential BPNet
#
# Trains a two-phase differential chromatin accessibility model on FOXA1
# ChIP-seq data from two prostate cancer cell lines.
#
# Data (GEO accession GSE115601):
#   LNCaP : GSM3508079  (FOXA1 ChIP-seq, RPM-normalised bigwig)
#   22Rv1 : GSM3508093  (FOXA1 ChIP-seq, RPM-normalised bigwig)
#
# Phase 1 — Multi-task MultitaskBPNet trained jointly on both conditions
#            (architecture: bpAI-TAC, Chandra et al. 2025)
# Phase 2 — Differential fine-tuning with DifferentialCountLoss
#            (log2FC computed directly from the depth-normalised bigwigs,
#             no BAM counting needed)
# Interpret — DeepLIFTSHAP through DifferentialAttributionTarget
#             (delta_log_counts mode) + TF-MoDISco motif discovery
#
# Prerequisites:
#   pip install cerberus captum tfmodisco-lite
#   (genome FASTA / blacklist auto-downloaded on first run)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit to match your environment
# ---------------------------------------------------------------------------

DATA_DIR=/nfs/turbo/umms-mcieslik/haotingc/cerberus_developmental_experiments/data

BIGWIG_22RV1="${DATA_DIR}/FOXA1_22Rv1/GSM3508093_SI_19463.bw"
PEAKS_22RV1="${DATA_DIR}/FOXA1_22Rv1/GSM3508093_SI_19463-macs2.bed.gz"

BIGWIG_LNCAP="${DATA_DIR}/FOXA1_LNCAP/GSM3508079_SI_16486.bw"
PEAKS_LNCAP="${DATA_DIR}/FOXA1_LNCAP/GSM3508079_SI_16486-macs2.bed.gz"

OUTPUT_DIR="/nfs/turbo/umms-mcieslik/haotingc/cerberus_developmental_experiments/models/foxa1_lncap_22rv1_differential_epoch50"
DATA_CACHE="/nfs/turbo/umms-mcieslik/haotingc/cerberus_developmental_experiments/data/genome"

# Optional: MEME motif database for TF-MoDISco report
MEME_DB="/home/haotingc/turbo/cerberus_developmental_experiments/data/motifs/motif_databases/HOCOMOCO/H12CORE_meme_format.meme"

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

ACCELERATOR="auto"    # auto | gpu | cpu | mps
DEVICES="auto"        # auto | 1 | 2 | ...
PRECISION="bf16"      # bf16 | mps | full
NUM_WORKERS=8
SEED=42

# ---------------------------------------------------------------------------
# Phase 1 hyperparameters (MultitaskBPNet)
# ---------------------------------------------------------------------------

BATCH_SIZE=64
PHASE1_EPOCHS=50
PHASE1_LR=1e-3
JITTER=256           # augmentation half-width
ALPHA="adaptive"     # count loss weight: 'adaptive' or float
COUNT_PSEUDOCOUNT=150.0   # pseudocount for count head log-transform
PATIENCE=10

# Architecture
FILTERS=64
N_LAYERS=8

# ---------------------------------------------------------------------------
# Phase 2 hyperparameters (DifferentialCountLoss)
# ---------------------------------------------------------------------------

PHASE2_EPOCHS=50
PHASE2_LR=1e-4
PHASE2_BATCH=64
PHASE2_PATIENCE=7
DIFF_PSEUDOCOUNT=1.0    # pseudocount in bigwig-signal units (RPM)
ABS_WEIGHT=0.0          # 0 = Naqvi default (delta only); >0 adds absolute regularisation

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

N_INTERP=2000           # number of test peaks for attribution
INTERP_BATCH=32
DLS_BASELINES=20        # dinucleotide-shuffled baselines per sequence
MODISCO_WINDOW=400
MAX_SEQLETS=2000

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

SCRIPT="$(dirname "$0")/../tools/train_multitask_differential_bpnet.py"

echo "=== FOXA1 LNCaP vs 22Rv1 — Multitask-Differential BPNet ==="
echo "Output : ${OUTPUT_DIR}"

python "${SCRIPT}" \
    --bigwig-a  "${BIGWIG_22RV1}"   \
    --peaks-a   "${PEAKS_22RV1}"    \
    --name-a    "22Rv1"             \
    --bigwig-b  "${BIGWIG_LNCAP}"   \
    --peaks-b   "${PEAKS_LNCAP}"    \
    --name-b    "LNCaP"             \
    \
    --genome    hg38                \
    --data-dir  "${DATA_CACHE}"     \
    --output-dir "${OUTPUT_DIR}"    \
    \
    --accelerator  "${ACCELERATOR}" \
    --devices      "${DEVICES}"     \
    --precision    "${PRECISION}"   \
    --num-workers  "${NUM_WORKERS}" \
    --seed         "${SEED}"        \
    \
    --batch-size        "${BATCH_SIZE}"        \
    --phase1-epochs     "${PHASE1_EPOCHS}"     \
    --phase1-lr         "${PHASE1_LR}"         \
    --jitter            "${JITTER}"            \
    --alpha             "${ALPHA}"             \
    --count-pseudocount "${COUNT_PSEUDOCOUNT}" \
    --patience          "${PATIENCE}"          \
    --filters           "${FILTERS}"           \
    --n-layers          "${N_LAYERS}"          \
    \
    --phase2-epochs      "${PHASE2_EPOCHS}"      \
    --phase2-lr          "${PHASE2_LR}"          \
    --phase2-batch-size  "${PHASE2_BATCH}"       \
    --phase2-patience    "${PHASE2_PATIENCE}"    \
    --diff-pseudocount   "${DIFF_PSEUDOCOUNT}"   \
    --abs-weight         "${ABS_WEIGHT}"         \
    \
    --stable            \
    --interpret         \
    --n-interp          "${N_INTERP}"       \
    --interp-batch-size "${INTERP_BATCH}"   \
    --dls-n-baselines   "${DLS_BASELINES}"  \
    --modisco-window    "${MODISCO_WINDOW}" \
    --max-seqlets       "${MAX_SEQLETS}"    \
    ${MEME_DB:+--meme-db "${MEME_DB}"}      \
    "$@"

echo "Done. Results in ${OUTPUT_DIR}/"
echo "  Phase 1 model  : ${OUTPUT_DIR}/phase1/single-fold/*/model.pt"
echo "  Phase 2 model  : ${OUTPUT_DIR}/phase2/model.pt"
echo "  Phase 2 losses : ${OUTPUT_DIR}/phase2/plots/phase2_losses.png"
echo "  Attribution    : ${OUTPUT_DIR}/interpretation/{ohe,attr}.npz"
echo "  TF-MoDISco     : ${OUTPUT_DIR}/interpretation/modisco_results.h5"
echo "  Motif report   : ${OUTPUT_DIR}/interpretation/report/"
