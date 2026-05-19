#!/usr/bin/env bash
# ChromBPNet bias-branch training example for scATAC-seq pseudobulk data.
#
# Stage 1 of the ChromBPNet workflow: trains the small BPNet bias branch on
# non-peak regions using the bulk kidney pseudobulk BigWig produced by the
# Cerberus pseudobulk pipeline.  The pseudobulk outputs must already exist
# in DATA_DIR (see examples/scatac_kidney_pseudobulk.sh to generate them).
#
# Pseudobulk reproducibility: this workflow was tested against
# ``snapatac2==2.8.0``.  pyproject.toml does not pin snapatac2 -- if exact
# byte-level reproducibility of the upstream pseudobulks is required,
# install that version explicitly:
#     pip install "snapatac2==2.8.0"
#
# Usage:
#   bash examples/scatac_kidney_chrombpnet_bias.sh
#   bash examples/scatac_kidney_chrombpnet_bias.sh --multi

set -euo pipefail

DATA_DIR="tests/data"
PSEUDOBULK_DIR="${DATA_DIR}/scatac_kidney_pseudobulk"
OUTPUT_DIR="${DATA_DIR}/models/scatac_kidney_chrombpnet_bias"

BIGWIG="${PSEUDOBULK_DIR}/bulk.bw"
PEAKS="${PSEUDOBULK_DIR}/bulk_merge.narrowPeak.bed.gz"

python tools/train_chrombpnet_bias.py \
    --bigwig "${BIGWIG}" \
    --peaks "${PEAKS}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 32 \
    --max-epochs 50 \
    "$@"
