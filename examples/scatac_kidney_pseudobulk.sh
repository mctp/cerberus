#!/usr/bin/env bash
# Pseudobulk BigWig and peak calling example for scATAC-seq data.
#
# Generates per-cell-type BigWig coverage tracks, MACS3 narrowPeak files,
# a bulk (all-cells) BigWig + peak set, and a merged peak set from the
# kidney scATAC-seq dataset (27,034 cells, 14 cell types, hg38).
#
# Prerequisites:
#   - snapatac2, macs3, bgzip, tabix must be installed
#   - Dataset is downloaded automatically to DATA_DIR
#
# Usage:
#   bash examples/scatac_kidney_pseudobulk.sh
#   bash examples/scatac_kidney_pseudobulk.sh --sequential
#   bash examples/scatac_kidney_pseudobulk.sh --n-jobs 16

set -euo pipefail

# --- Variables ---
DATA_DIR="tests/data"
OUTPUT_DIR="tests/data/scatac_kidney_pseudobulk"
N_JOBS=8

# --- Download dataset ---
python - <<EOF
from cerberus.download import download_dataset
from pathlib import Path
download_dataset(Path("${DATA_DIR}") / "dataset", name="kidney_scatac")
EOF

FRAGMENTS="${DATA_DIR}/dataset/kidney_scatac/fragments.tsv.bgz"
H5AD="${DATA_DIR}/dataset/kidney_scatac/gene_activity.h5ad"

# --- Generate pseudobulk BigWigs + peaks ---
python tools/scatac_pseudobulk.py \
    "${FRAGMENTS}" \
    "${H5AD}" \
    "${OUTPUT_DIR}" \
    --genome hg38 \
    --groupby cell_type \
    --call-peaks \
    --bulk \
    --merge \
    --counting-strategy insertion \
    --normalization raw \
    --n-jobs "${N_JOBS}" \
    "$@"
