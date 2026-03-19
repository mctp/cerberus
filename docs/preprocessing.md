# Preprocessing BAM/BED for Cerberus

Cerberus training tools (`tools/train_bpnet.py`, `tools/train_pomeranian.py`) expect:

- A signal file in BigWig format (`--bigwig`)
- A training-interval file in BED or narrowPeak format (`--peaks`)

This repository includes a preprocessing helper script that converts BAM + BED inputs into those files using `bamCoverage`.

## Script

`tools/preprocess_bam_bed_for_cerberus.sh`

## Requirements

- `samtools`
- `deepTools` (`bamCoverage`)
- `gzip`

## ENCODE FOXA1 K562 Example

```bash
bash tools/preprocess_bam_bed_for_cerberus.sh \
  --bam data/ENCODE_FOXA1_K562/ENCFF543GEC_replicate1.bam \
  --bam data/ENCODE_FOXA1_K562/ENCFF933WEU_replicate2.bam \
  --bed data/ENCODE_FOXA1_K562/ENCFF624PSE_replicate1.bed.gz \
  --bed data/ENCODE_FOXA1_K562/ENCFF122DVT_replicate2.bed.gz \
  --out-dir data/ENCODE_FOXA1_K562 \
  --prefix encode_foxa1_k562_encsr819lhg \
  --threads 8 \
  --mapq 30 \
  --ignore-duplicates \
  --blacklist data/genome/hg38/blacklist.bed
```

Outputs:

- `data/ENCODE_FOXA1_K562/encode_foxa1_k562_encsr819lhg.counts.bw`
- `data/ENCODE_FOXA1_K562/encode_foxa1_k562_encsr819lhg.peaks.bed.gz`

## Use Outputs for Training

BPNet:

```bash
python tools/train_bpnet.py \
  --bigwig data/ENCODE_FOXA1_K562/encode_foxa1_k562_encsr819lhg.counts.bw \
  --peaks data/ENCODE_FOXA1_K562/encode_foxa1_k562_encsr819lhg.peaks.bed.gz \
  --fasta data/genome/hg38/hg38.fa \
  --blacklist data/genome/hg38/blacklist.bed \
  --gaps data/genome/hg38/gaps.bed \
  --output-dir data/models/chip_foxa1_k562_encode_bpnet
```

Pomeranian:

```bash
python tools/train_pomeranian.py \
  --bigwig data/ENCODE_FOXA1_K562/encode_foxa1_k562_encsr819lhg.counts.bw \
  --peaks data/ENCODE_FOXA1_K562/encode_foxa1_k562_encsr819lhg.peaks.bed.gz \
  --fasta data/genome/hg38/hg38.fa \
  --blacklist data/genome/hg38/blacklist.bed \
  --gaps data/genome/hg38/gaps.bed \
  --output-dir data/models/chip_foxa1_k562_encode_pomeranian
```

## Full Example Pipeline Script

Use `examples/chip_foxa1_k562_encode_bpnet_pomeranian.sh` to run preprocessing and both trainings end-to-end.

## Notes

- `bamCoverage` is run with `--binSize 1 --normalizeUsing None` to produce base-resolution raw counts.
- Multiple BAMs are merged with `samtools merge` before coverage generation.
- Multiple BED files are concatenated, sorted, and merged into a single interval set.
