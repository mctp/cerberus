# How scooby Gets Single-Cell Data into Training Targets

Reference: `~/Code/s2f-models/repos/scooby`

## Key Insight

scooby does **not** train on pseudobulk BigWig tracks. It trains on **single-cell sparse coverage matrices** stored in AnnData `.obsm`, computing dense coverage on the fly per genomic interval. Pseudobulk BigWigs are only generated for **evaluation**.

---

## 1. Raw Data to Sparse Matrices

### ATAC-seq

Starting material: 10x Cell Ranger `fragments.tsv.gz` + `.tbi`.

```python
# Import fragment file into AnnData (scdata.py notebook cell)
atac_coverage = sp.pp.import_fragments(
    "fragments.tsv.gz",
    chrom_sizes=sp.genome.hg38,
    min_num_fragments=0,
    whitelist=adata_count.obs_names,   # filter to QC-passing barcodes
    sorted_by_barcode=False,
)
# Convert fragment intervals to Tn5 insertion sites (2 cuts per fragment)
sp.pp.fragments_to_insertions(atac_coverage)
```

Result: `atac_coverage.obsm['insertion']` — a sparse CSR matrix of shape `(n_cells, genome_length)` where each entry counts Tn5 insertion events at that base position for that cell. The genome is linearized (all chromosomes concatenated) with offsets stored in `adata.uns['reference_sequences']`.

### RNA-seq

Starting material: Cell Ranger BAM file.

```python
# BAM -> strand-specific fragment files (custom snapatac2_scooby)
sp.pp.make_fragment_file(
    bam_file="gex_possorted_bam.bam",
    output_file="gex.fragments.bed.gz",
    stranded=True,       # separate plus/minus strand files
    is_paired=False,     # RNA reads are single-end
    shift_left=0,        # no Tn5 shift correction (ATAC-specific)
    shift_right=0,
    barcode_tag="CB",
    umi_tag="UB",
    xf_filter=True,      # Cell Ranger quality flag
)
# Produces: gex.fragments.bed.plus.gz, gex.fragments.bed.minus.gz

# Import each strand into AnnData
rna_coverage_plus = sp.pp.import_fragments("gex.fragments.bed.plus.gz", ...)
rna_coverage_minus = sp.pp.import_fragments("gex.fragments.bed.minus.gz", ...)
```

Result: `rna_coverage_{plus,minus}.obsm['fragment_single']` — sparse CSR, same shape as ATAC, one per strand.

### Storage

All three AnnData objects are saved as `.h5ad` files:
- `snapatac_rna_plus.h5ad` (obsm: `fragment_single`)
- `snapatac_rna_minus.h5ad` (obsm: `fragment_single`)
- `snapatac_atac.h5ad` (obsm: `insertion`)

For multi-sample datasets, per-sample AnnDatas are concatenated via `anndata.experimental.concat_on_disk()`.

---

## 2. Cell Embeddings (Not Cell-Type Labels)

scooby does not use discrete cell-type labels. Instead it uses a continuous **cell embedding** vector per cell, learned via PoissonMultiVI (scvi-tools):

```python
scvi.external.POISSONMULTIVI.setup_anndata(adata_train)
model = scvi.external.POISSONMULTIVI(adata_train, n_genes=..., n_regions=...)
model.train()
X_emb = model.get_latent_representation()  # (n_cells, emb_dim)
```

The embedding is saved as a parquet file (`embedding.pq`). During training, the model receives cell embedding indices and looks up the corresponding vectors to condition its predictions.

---

## 3. Training: On-the-Fly Dense Coverage from Sparse Matrices

### No pseudobulk step. Training reads sparse matrices directly.

The core training dataset is `onTheFlyMultiomeDataset` (or `onTheFlyDataset` for RNA-only) in `scooby/data/scdata.py`.

**Per-sample data flow in `__getitem__`:**

1. **Sample cells**: randomly pick 64 cells (configurable `cell_sample_size`)
2. **For each cell**, optionally aggregate with k-nearest neighbors (but in practice `no_neighbors.npz` means true single-cell — no aggregation)
3. **Slice the sparse matrix** for the genomic interval:

```python
# _sparse_to_coverage_atac (scdata.py:55-75)
m = adata.obsm["insertion"][cell_indices]   # (n_sampled_cells, genome_length)
m = m[:, start:end]                         # slice to interval columns
dense_matrix = m.sum(0).A[0]               # sum across cells -> (interval_length,)
```

4. **Normalize** (RNA path, `_process_rna`):
```python
tensor = tensor / custom_read_length               # divide by read length (90bp)
seq_cov = F.avg_pool1d(tensor, kernel_size=32, stride=32) * 32  # bin to 32bp
seq_cov = -1 + (1 + seq_cov) ** 0.75               # power transform
# soft clip at clip_soft (5), hard clip at 768
```

5. **Normalize** (ATAC path, `_process_atac`):
```python
seq_cov = F.avg_pool1d(tensor, kernel_size=32, stride=32) * 32  # bin to 32bp
if normalize_atac:
    seq_cov = seq_cov * 0.05                        # optional scaling
# no power transform for ATAC
```

6. **Concatenate** modalities: `[rna_plus, rna_minus, atac]` stacked as 3 output tracks

### Training targets shape

For a 524,288 bp context window binned at 32bp: `(n_cells, 16384, n_tracks)` where `n_tracks` = 2 (RNA-only) or 3 (multiome).

Each cell gets its own target profile — the model predicts per-cell coverage conditioned on the cell embedding.

---

## 4. Pseudobulk BigWigs: Evaluation Only

Pseudobulk BigWigs are generated **only for evaluation**, not training.

```python
# Assign cell type labels (e.g., leiden clustering) to the coverage AnnData
rna_coverage_plus.obs = adata_count.obs  # copies 'leiden' column

sp.ex.export_coverage(
    rna_coverage_plus,
    groupby='leiden',        # one BigWig per cluster
    bin_size=1,              # 1bp resolution
    out_dir="pseudobulks",
    normalization=None,      # raw counts, no normalization
    n_jobs=-1,
    max_frag_length=None,
    suffix='.bw',
    prefix="plus.",          # -> pseudobulks/plus.0.bw, plus.1.bw, ...
)
```

Output: `pseudobulks/{plus,minus}.{cluster_id}.bw` — one BigWig per cluster per strand.

The evaluation dataset `onTheFlyPseudobulkDataset` loads these with `pybigtools`:

```python
# _process_paths (scdata.py:256-295)
bigwigs = [pybigtools.open(f"pseudobulks/plus.{cell_type}.bw")]
vals = bw.values(chrom, start, end)
tensor = tensor / custom_read_length     # 90bp
seq_cov = F.avg_pool1d(tensor, 32, 32) * 32
seq_cov = -1 + (1 + seq_cov) ** 0.75    # same power transform
# soft clip at 384 (higher than training's 5), hard clip at 768
```

---

## 5. Summary: Two Data Paths

| | Training | Evaluation |
|---|---|---|
| **Source** | Sparse matrices in AnnData `.obsm` | Pseudobulk BigWig files |
| **Granularity** | Per-cell (64 cells sampled per interval) | Per-cell-type aggregate |
| **Aggregation** | Sum across sampled cells (+ optional neighbors) | SnapATAC2 `export_coverage(groupby=...)` |
| **Resolution** | 32bp bins (avg_pool1d) | 32bp bins (avg_pool1d) |
| **Normalization** | RNA: /read_len, x^0.75, soft-clip@5. ATAC: optional *0.05 | /read_len, x^0.75, soft-clip@384 |
| **Tool** | Direct sparse matrix slicing | pybigtools BigWig reader |

---

## 6. Implications for Cerberus

scooby's approach reveals two distinct integration strategies:

**Strategy A (simpler)**: Pre-compute pseudobulk BigWigs via `sp.ex.export_coverage()`, feed them to cerberus as regular BigWig targets via `SignalExtractor`. Zero cerberus changes. Loses per-cell resolution.

**Strategy B (scooby-like)**: Build a `FragmentExtractor` (or sparse-matrix extractor) that does the on-the-fly slicing from AnnData sparse matrices. This is closer to what scooby actually does for training. Would require storing the genome-wide sparse matrices and doing the cell-sampling/aggregation inside the extractor.

**Key difference from our plan**: The plan doc's `FragmentExtractor` design (Section 2) queries a tabix-indexed `.tsv.gz` file. scooby skips tabix entirely — it pre-loads fragments into genome-wide sparse matrices via SnapATAC2 and slices columns directly. The sparse matrix approach is faster for random access across many cells but requires more memory (entire genome in sparse format per modality).
