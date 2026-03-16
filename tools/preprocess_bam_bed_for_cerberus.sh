#!/usr/bin/env bash
# Preprocess BAM + BED/narrowPeak files into Cerberus-ready inputs:
#   1) Base-resolution count BigWig from BAM (bamCoverage)
#   2) Merged BED peaks file for interval sampling
#
# Outputs:
#   <out-dir>/<prefix>.counts.bw
#   <out-dir>/<prefix>.peaks.bed.gz
#
# Example:
#   bash tools/preprocess_bam_bed_for_cerberus.sh \
#     --bam data/ENCODE_FOXA1_K562/ENCFF543GEC_replicate1.bam \
#     --bam data/ENCODE_FOXA1_K562/ENCFF933WEU_replicate2.bam \
#     --bed data/ENCODE_FOXA1_K562/ENCFF624PSE_replicate1.bed.gz \
#     --bed data/ENCODE_FOXA1_K562/ENCFF122DVT_replicate2.bed.gz \
#     --out-dir data/ENCODE_FOXA1_K562 \
#     --prefix encode_foxa1_k562 \
#     --threads 8 \
#     --mapq 30 \
#     --ignore-duplicates

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  preprocess_bam_bed_for_cerberus.sh \
    --bam <rep1.bam> [--bam <rep2.bam> ...] \
    --bed <rep1.bed|rep1.bed.gz> [--bed <rep2.bed|rep2.bed.gz> ...] \
    --out-dir <output_dir> \
    --prefix <sample_name> \
    [--threads 8] \
    [--bin-size 1] \
    [--mapq 30] \
    [--blacklist <blacklist.bed>] \
    [--ignore-duplicates]

Description:
  Converts BAM alignments to a base-resolution count BigWig using deepTools
  bamCoverage, and builds a merged BED peak set from one or more BED files.

Required tools:
  samtools, bamCoverage, gzip
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

samtools_index_compat() {
    local bam="$1"
    # Older samtools builds do not support -@ for index.
    if samtools index -@ "${THREADS}" "${bam}" >/dev/null 2>&1; then
        return 0
    fi
    samtools index "${bam}"
}


declare -a BAMS=()
declare -a BEDS=()
OUT_DIR=""
PREFIX=""
THREADS=8
BIN_SIZE=1
MAPQ=30
BLACKLIST=""
IGNORE_DUPLICATES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bam)
            [[ $# -ge 2 ]] || die "Missing value for --bam"
            BAMS+=("$2")
            shift 2
            ;;
        --bed)
            [[ $# -ge 2 ]] || die "Missing value for --bed"
            BEDS+=("$2")
            shift 2
            ;;
        --out-dir)
            [[ $# -ge 2 ]] || die "Missing value for --out-dir"
            OUT_DIR="$2"
            shift 2
            ;;
        --prefix)
            [[ $# -ge 2 ]] || die "Missing value for --prefix"
            PREFIX="$2"
            shift 2
            ;;
        --threads)
            [[ $# -ge 2 ]] || die "Missing value for --threads"
            THREADS="$2"
            shift 2
            ;;
        --bin-size)
            [[ $# -ge 2 ]] || die "Missing value for --bin-size"
            BIN_SIZE="$2"
            shift 2
            ;;
        --mapq)
            [[ $# -ge 2 ]] || die "Missing value for --mapq"
            MAPQ="$2"
            shift 2
            ;;
        --blacklist)
            [[ $# -ge 2 ]] || die "Missing value for --blacklist"
            BLACKLIST="$2"
            shift 2
            ;;
        --ignore-duplicates)
            IGNORE_DUPLICATES=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ ${#BAMS[@]} -gt 0 ]] || die "At least one --bam is required"
[[ ${#BEDS[@]} -gt 0 ]] || die "At least one --bed is required"
[[ -n "${OUT_DIR}" ]] || die "--out-dir is required"
[[ -n "${PREFIX}" ]] || die "--prefix is required"

require_cmd samtools
require_cmd bamCoverage
require_cmd gzip

mkdir -p "${OUT_DIR}"

echo "[1/4] Ensuring BAM index files..."
for bam in "${BAMS[@]}"; do
    if [[ -s "${bam}.bai" || -s "${bam%.bam}.bai" ]]; then
        continue
    fi
    samtools_index_compat "${bam}"
done

TARGET_BAM="${BAMS[0]}"
if [[ ${#BAMS[@]} -gt 1 ]]; then
    echo "[2/4] Merging ${#BAMS[@]} BAM files..."
    TARGET_BAM="${OUT_DIR}/${PREFIX}.merged.bam"
    samtools merge -@ "${THREADS}" -f "${TARGET_BAM}" "${BAMS[@]}"
    samtools_index_compat "${TARGET_BAM}"
else
    echo "[2/4] Single BAM provided; skipping merge."
fi

BIGWIG_OUT="${OUT_DIR}/${PREFIX}.counts.bw"
echo "[3/4] Creating base-resolution count BigWig with bamCoverage..."

declare -a BAMCOVERAGE_ARGS=(
    --bam "${TARGET_BAM}"
    --outFileName "${BIGWIG_OUT}"
    --outFileFormat bigwig
    --binSize "${BIN_SIZE}"
    --normalizeUsing None
    --numberOfProcessors "${THREADS}"
    --minMappingQuality "${MAPQ}"
)

if [[ "${IGNORE_DUPLICATES}" == true ]]; then
    BAMCOVERAGE_ARGS+=(--ignoreDuplicates)
fi

if [[ -n "${BLACKLIST}" ]]; then
    BAMCOVERAGE_ARGS+=(--blackListFileName "${BLACKLIST}")
fi

bamCoverage "${BAMCOVERAGE_ARGS[@]}"

echo "[4/4] Building merged peak BED..."
PEAKS_OUT="${OUT_DIR}/${PREFIX}.peaks.bed.gz"
TMP_SORTED="$(mktemp "${OUT_DIR}/${PREFIX}.tmp.sorted.XXXXXX.bed")"
TMP_MERGED="$(mktemp "${OUT_DIR}/${PREFIX}.tmp.merged.XXXXXX.bed")"
trap 'rm -f "${TMP_SORTED}" "${TMP_MERGED}"' EXIT

{
    for bed in "${BEDS[@]}"; do
        if [[ "${bed}" == *.gz ]]; then
            gzip -cd "${bed}"
        else
            cat "${bed}"
        fi
    done
} | awk 'BEGIN{OFS="\t"} !/^#/ && !/^track/ && !/^browser/ && NF >= 3 {print $1, $2, $3}' \
  | sort -k1,1 -k2,2n -k3,3n > "${TMP_SORTED}"

awk '
BEGIN { OFS="\t" }
NR == 1 { c=$1; s=$2; e=$3; next }
{
    if ($1 == c && $2 <= e) {
        if ($3 > e) {
            e = $3
        }
    } else {
        print c, s, e
        c = $1; s = $2; e = $3
    }
}
END {
    if (NR > 0) {
        print c, s, e
    }
}
' "${TMP_SORTED}" > "${TMP_MERGED}"

gzip -c "${TMP_MERGED}" > "${PEAKS_OUT}"

echo
echo "Preprocessing complete."
echo "BigWig: ${BIGWIG_OUT}"
echo "Peaks : ${PEAKS_OUT}"
echo
echo "Use with Cerberus:"
echo "  python tools/train_bpnet.py --bigwig \"${BIGWIG_OUT}\" --peaks \"${PEAKS_OUT}\" --output-dir data/models/${PREFIX}_bpnet"
echo "  python tools/train_pomeranian.py --bigwig \"${BIGWIG_OUT}\" --peaks \"${PEAKS_OUT}\" --output-dir data/models/${PREFIX}_pomeranian"
