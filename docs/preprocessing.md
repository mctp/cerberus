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

## scATAC pseudobulk normalisation

For multi-task training on scATAC-seq cell-type pseudobulks, raw
per-cell-type BigWigs often have different library sizes and different
baseline peak-height scales. `tools/scatac_normalize_pseudobulk.py` is a
self-contained tool that writes normalized BigWigs with two multiplicative
passes:

1. **CPM scaling** converts raw coverage to counts-per-million using each
   group's library size. Library sizes are read from `group_summary.tsv` when
   available, or computed from each BigWig's total signal.
2. **Constitutive-anchor baseline rescaling** fits one additional scalar per
   group. The scalar is estimated from strong peaks that are low-specificity
   across groups, then applied to the entire group BigWig.

The final output value for every record in a group BigWig is:

```text
normalized_signal[group] = raw_signal[group] * final_scale[group]
final_scale[group] = cpm_scale[group] * baseline_weight[group]
cpm_scale[group] = 1_000_000 / library_size[group]
```

### Constitutive-anchor selection

The method uses a shared consensus peak universe, but the final anchor set is
chosen separately for each group. This is intentional and matches the
CREsted-style peak-height normalization strategy.

For each consensus peak, the script scores a fixed-width window centered on the
narrowPeak summit when available, otherwise on the peak center. The default
window is 1000 bp and the default statistic is the BigWig `sum` over that
window.

The scored peak-by-group matrix is first converted to CPM units:

```text
cpm_matrix[peak, group] = raw_peak_signal[peak, group] * cpm_scale[group]
```

The script then computes one Gini value per peak across groups. Low Gini means
the peak is broadly accessible rather than specific to one group. The low-Gini
cutoff is:

```text
gini_threshold = mean(peak_gini) - gini_std_threshold * std(peak_gini)
```

For each group independently:

```text
candidates[group] = peaks where cpm_matrix[:, group] > peak_threshold
top_peaks[group] = top_k_percent strongest candidates in that group
anchors[group] = top_peaks[group] intersect low_gini_peaks
constitutive_mean[group] = mean(cpm_matrix[anchors[group], group])
```

If a group has fewer than `min_anchor_peaks` anchors, the tool can fall back to
the lowest-Gini peaks among that group's top peaks unless
`--no-anchor-fallback` is set.

### Baseline weights

After each group has a `constitutive_mean`, the tool chooses a reference anchor
level. The default is `--reference-strategy max`:

```text
reference = max(constitutive_mean[group] for group in groups)
baseline_weight[group] = reference / constitutive_mean[group]
```

This means each group's strong, low-specificity anchor set is brought to the
same average height as the highest such group. The script does not modify
individual peaks separately; `baseline_weight[group]` is a single scalar applied
to the full BigWig for that group.

### Interpretation

This is not a fixed housekeeping-peak normalization where every group is
measured on the exact same regions. Instead:

```text
anchors[A] = top strong low-Gini peaks for group A
anchors[B] = top strong low-Gini peaks for group B
```

These anchor sets usually overlap, because low-Gini peaks are shared by design,
but they are not required to be identical. The reason for this choice is that a
single fixed anchor set can be weak or noisy in some groups. Per-group top peaks
ensure every group estimates its baseline from high-confidence accessible
regions, while the low-Gini filter removes strongly group-specific peaks.

The tradeoff is important. This normalization is appropriate when global
differences in peak height are treated mainly as technical depth or baseline
scale effects. It is not appropriate if the biological signal of interest is a
true global increase or decrease in accessibility across an entire cell type,
because the normalization deliberately removes that kind of global scale
difference.

This mirrors CREsted's `crested.pp.normalize_peaks` approach, which computes
normalization weights from the top values per cell type after low-Gini filtering
and rescales each cell type by a single weight. See the CREsted
[`normalize_peaks` documentation](https://crested.readthedocs.io/en/latest/api/_autosummary/crested.pp.normalize_peaks.html)
and
[`_normalization.py`](https://github.com/aertslab/CREsted/blob/main/src/crested/pp/_normalization.py).

```bash
python tools/scatac_normalize_pseudobulk.py \
    path/to/pseudobulk_dir path/to/output_dir \
    --input-scale raw \
    --reference-strategy max \
    --top-k-percent 0.01 --gini-std-threshold 1.0
```

Outputs per group:

| File | Contents |
|---|---|
| `<group>.bw` | Normalised BigWig (CPM × baseline weight) |
| `<group>.narrowPeak.bed.gz` | Copied from input (if present) |
| `normalization_summary.tsv` | Per-group: library size, CPM scale, anchor count, baseline weight, final scale |
| `constitutive_anchor_peaks.tsv` | Diagnostic: chosen anchor peak indices and Gini values |
| `normalization_metadata.json` | All parameters + summary counts |
| `targets.json` | Ready-to-use `{group: {bigwig, peaks}}` spec for `tools/train_chrombpnet_multitask.py` |

The output `targets.json` plugs directly into [tools/train_chrombpnet_multitask.py](usage.md#generic-training-tools).
