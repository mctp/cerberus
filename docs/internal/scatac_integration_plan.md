# Single-Cell ATAC-seq Integration Plan

## Overview

This document covers the integration of single-cell ATAC-seq data into cerberus, including a review of the Python scATAC-seq package ecosystem, a concrete `FragmentExtractor` design, and architectural refactorings that would make fragment (and future format) support a first-class citizen.

---

## 1. scATAC-seq Package Ecosystem Review

### 1.1 Integration Surface

Cerberus's key constraint is the `BaseSignalExtractor` protocol (`signal.py:10-17`) — any signal source must implement `extract(interval: Interval) -> torch.Tensor` returning shape `(Channels, Length)`. Currently all signal enters through **BigWig files** via `pybigtools`. The dataloader design doc already identifies scATAC as a future extension (`dataloader.md:364-366`).

Three integration pathways exist, ranked by effort.

### 1.2 Pathway 1: Pseudobulk BigWig Export (zero cerberus changes)

These packages preprocess scATAC fragment files into per-cell-type BigWig coverage tracks that cerberus consumes directly via `SignalExtractor`.

| Package | Function | Notes |
|---------|----------|-------|
| **SnapATAC2** | `snap.ex.export_coverage(adata, groupby='cell_type', suffix='.bw')` | Rust-backed (fast), scverse ecosystem, AnnData native. Exports BigWig per group. Used by scooby for this exact purpose. |
| **pycisTopic** | `export_pseudobulk(obj, variable='cell_type', bigwig_path=...)` | Exports both fragment BED + BigWig per cell type. Some known bugs in older versions. |
| **muon** | `ac.tl.count_fragments_features()` + external BigWig conversion | No direct BigWig export, but provides fragment counting infrastructure. Requires extra step. |

**Recommendation:** SnapATAC2 is the clear winner. A preprocessing script would:

1. `snap.pp.import_data()` — read fragments + QC
2. Cluster / annotate cell types
3. `snap.ex.export_coverage(groupby='cell_type')` — write BigWigs
4. Use the BigWigs directly in cerberus `DataConfig` as target channels

This gives one model per cell type or one multi-headed model with cell types as output channels.

### 1.3 Pathway 2: FragmentExtractor (moderate effort)

Build a `FragmentExtractor` implementing `BaseSignalExtractor` that computes coverage on-the-fly from tabix-indexed fragment files. Enables dynamic filtering by cell type, fragment size, etc. without re-exporting BigWigs.

| Package | Fragment I/O | Speed | Integration Fit |
|---------|-------------|-------|-----------------|
| **SnapATAC2** | Rust-backed fragment reader | Excellent | Best candidate. GIL-free like `pybigtools`. |
| **pysam** | `pysam.TabixFile` for indexed fragments | Good | Already a cerberus dependency. Straightforward but Python-bound (GIL). |
| **muon** | Higher-level fragment counting | Moderate | Less control over pileup computation. |

See Section 2 for the full `FragmentExtractor` design.

### 1.4 Pathway 3: Per-Cell Models (new paradigm)

Training models that predict individual cell accessibility from sequence requires a fundamentally different output structure.

| Package/Model | Approach | Cerberus Fit |
|---------------|----------|-------------|
| **scooby** | Fine-tunes Borzoi with LoRA + cell-type embedding head. Predicts per-cell coverage from sequence. | Most relevant. Architecture closest to cerberus's S2F paradigm. Published in Nature Methods. |
| **scBasset** | CNN predicting cell x peak accessibility from sequence. Part of scvi-tools. | Different paradigm — binary accessibility per peak, not base-resolution profiles. |
| **PeakVI** | VAE for scATAC latent space. Not sequence-based. | Not relevant — operates on count matrices, not DNA sequence. |

### 1.5 Recommendation Summary

| Strategy | Effort | What You Get | Best Package |
|----------|--------|-------------|-------------|
| Pseudobulk BigWig | None (preprocessing only) | Train cerberus models on cell-type-specific ATAC signal | SnapATAC2 |
| FragmentExtractor | ~1-2 new classes | Dynamic pseudobulk, fragment-size filtering, on-the-fly grouping | SnapATAC2 (Rust I/O) or pysam |
| Per-cell prediction | New model architecture + data pipeline | Predict accessibility per individual cell from sequence | Study scooby's approach |

---

## 2. FragmentExtractor Design

### 2.1 The Fragment File Format

A 10x scATAC fragment file (`fragments.tsv.gz`) is a tabix-indexed, tab-separated file with one row per Tn5 insertion fragment:

```
chr1    10073   10133   AGTTCGACATCGATCA-1    1
chr1    10082   10348   CTAAGCCTCCATACTG-1    1
chr1    10092   10246   GCATTGAAGTTACCGA-1    2
|       |       |       |                     |
chrom   start   end     barcode               duplicate_count
```

The tabix index (`.tbi`) enables fast random-access queries by genomic region — the same query pattern that `SignalExtractor` performs against BigWig files.

### 2.2 Core Operation

The job replaces the BigWig lookup with a fragment pileup:

```
SignalExtractor:   interval -> query BigWig   -> base-resolution array -> (C, L)
FragmentExtractor: interval -> query fragments -> filter barcodes -> pileup -> (C, L)
```

For a given `Interval("chr1", 1000, 3000)`:

1. **Query** the tabix-indexed fragment file for all fragments overlapping `chr1:1000-3000`
2. **Filter** to only fragments from barcodes belonging to the desired cell group(s)
3. **Pileup** — for each base position, count how many fragments cover it
4. **Return** `torch.Tensor` of shape `(Channels, Length)`

### 2.3 Pileup Example

```
Position:     1000  1001  1002  1003  1004  ...  2998  2999

Fragment A:   |=========================|                   barcode AGTC... (T-cell)
Fragment B:         |================|                      barcode AGTC... (T-cell)
Fragment C:               |====================|            barcode CTAG... (B-cell)
Fragment D:                     |========================|  barcode CTAG... (B-cell)

T-cell pileup: 1     2     2     2     1    ...   0     0   <- channel 0
B-cell pileup: 0     0     1     2     2    ...   1     1   <- channel 1
```

Each channel corresponds to a barcode group (cell type, cluster, or individual cell). The output tensor is `(2, 2000)` — structurally identical to a 2-channel BigWig extraction.

### 2.4 Count Modes

**`insertion` mode** — counts Tn5 transposase cut sites (the two ends of each fragment). This is what most ATAC-seq analyses use because the cut sites indicate where chromatin was accessible. Each fragment contributes signal at exactly 2 positions. This is what tools like MACS2 and SnapATAC2 use by default.

**`fragment` mode** — counts full fragment coverage (every base between the two cut sites). This gives a smoother signal more analogous to ChIP-seq BigWig tracks. May be more appropriate if existing cerberus models were trained on BigWig tracks computed this way.

### 2.5 Fragment Size Filtering

Fragment size is biologically informative in ATAC-seq:

| Fragment size | What it represents |
|---|---|
| < 147 bp | Nucleosome-free regions (NFR) — active regulatory elements |
| 147-294 bp | Mono-nucleosome |
| 294-441 bp | Di-nucleosome |

By setting `min_fragment_size=0, max_fragment_size=147`, you get a channel representing only nucleosome-free accessibility — a cleaner signal for regulatory element prediction. You could create multiple channels per cell type by fragment size class.

### 2.6 Implementation

```python
class FragmentExtractor(BaseSignalExtractor):
    """
    Computes base-resolution coverage from a tabix-indexed fragment file
    for one or more barcode groups (cell types / pseudobulks).
    """
    def __init__(
        self,
        fragments_path: Path,
        barcode_groups: dict[str, set[str]],
        min_fragment_size: int = 0,
        max_fragment_size: int = int(1e9),
        count_mode: str = "insertion",
    ):
        self.fragments_path = fragments_path
        self.channels = sorted(barcode_groups.keys())
        self.barcode_groups = barcode_groups
        self.min_fragment_size = min_fragment_size
        self.max_fragment_size = max_fragment_size
        self.count_mode = count_mode
        self._tabix = None

    def _load(self):
        """Lazy loader — fork-safe via __getstate__."""
        import pysam
        self._tabix = pysam.TabixFile(str(self.fragments_path))

    def __getstate__(self):
        """Pickle support: drop file handle for DataLoader workers."""
        state = self.__dict__.copy()
        state["_tabix"] = None
        return state

    def extract(self, interval: Interval) -> torch.Tensor:
        if self._tabix is None:
            self._load()

        length = interval.end - interval.start
        pileup = np.zeros((len(self.channels), length), dtype=np.float32)

        # Build barcode -> channel index lookup
        barcode_to_idx = {}
        for idx, name in enumerate(self.channels):
            for bc in self.barcode_groups[name]:
                barcode_to_idx[bc] = idx

        try:
            for row in self._tabix.fetch(interval.chrom, interval.start, interval.end):
                fields = row.split("\t")
                frag_start = int(fields[1])
                frag_end = int(fields[2])
                barcode = fields[3]

                ch_idx = barcode_to_idx.get(barcode)
                if ch_idx is None:
                    continue

                frag_size = frag_end - frag_start
                if frag_size < self.min_fragment_size or frag_size > self.max_fragment_size:
                    continue

                if self.count_mode == "insertion":
                    left = frag_start - interval.start
                    right = frag_end - 1 - interval.start
                    if 0 <= left < length:
                        pileup[ch_idx, left] += 1
                    if 0 <= right < length:
                        pileup[ch_idx, right] += 1
                else:
                    s = max(0, frag_start - interval.start)
                    e = min(length, frag_end - interval.start)
                    pileup[ch_idx, s:e] += 1

        except ValueError:
            pass  # chrom not in fragment file

        return torch.from_numpy(pileup)
```

### 2.7 Usage in Cerberus

Since it satisfies `BaseSignalExtractor`, it plugs directly into `CerberusDataset`:

```python
barcode_groups = {
    "T-cell": {"AGTCGATC-1", "CTGAATCG-1", ...},
    "B-cell": {"TAGCTTAG-1", "GCATCGAT-1", ...},
    "Monocyte": {"ATCGATCG-1", ...},
}

frag_extractor = FragmentExtractor(
    fragments_path=Path("fragments.tsv.gz"),
    barcode_groups=barcode_groups,
    count_mode="insertion",
    max_fragment_size=147,
)

dataset = CerberusDataset(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    target_signal_extractor=frag_extractor,
)
```

### 2.8 Performance Considerations

| Concern | Mitigation |
|---|---|
| Tabix query speed | pysam's TabixFile is C-backed, queries are O(log n) via the index. |
| GIL | pysam releases GIL during I/O but the per-row Python loop holds it. Each DataLoader worker gets its own handle via `__getstate__` + lazy reload. |
| Hot regions | Dense promoters may have thousands of fragments per window. Could accelerate with Cython/Rust helper. |
| Memory | No genome-wide data held in memory. Fragment file stays on disk. |
| Alternative backend | SnapATAC2's Rust-backed fragment reader is GIL-free. If pysam becomes a bottleneck, wrapping SnapATAC2's I/O would give better multi-worker throughput. |

---

## 3. Architectural Refactorings

The `FragmentExtractor` works today via manual extractor injection. But four coupled rigidity points in the current architecture prevent fragment support from being a true first-class citizen — configurable via YAML, auto-created by the framework, and extensible to future formats (BAM, HDF5, Zarr) without modifying existing code.

### 3.1 Problem: `dict[str, Path]` Can't Describe Fragments

`DataConfig` (`config.py:80-106`) defines inputs/targets as:

```python
inputs: dict[str, Path]    # channel_name -> file_path
targets: dict[str, Path]
```

A BigWig channel needs only a path. A fragment channel needs a path **plus** barcode groups, fragment size range, and count mode. The current type can't express this, which means fragment-based configs can't round-trip through YAML.

**Proposed change: `ChannelSpec`**

```python
class ChannelSpec(TypedDict, total=False):
    """Rich channel specification. Only 'path' is required."""
    path: Path
    extractor_cls: str              # e.g., "cerberus.signal.FragmentExtractor"
    extractor_args: dict[str, Any]  # barcode_groups_path, count_mode, etc.

# DataConfig becomes:
inputs: dict[str, Path | ChannelSpec]
targets: dict[str, Path | ChannelSpec]
```

A bare `Path` is shorthand for `{"path": path}` — fully backward compatible. The YAML would look like:

```yaml
# Current (still works)
targets:
  H3K4me3: /data/h3k4me3.bw

# New: fragment-backed channels
targets:
  T-cell_ATAC:
    path: /data/fragments.tsv.gz
    extractor_cls: cerberus.signal.FragmentExtractor
    extractor_args:
      barcodes_path: /data/t_cell_barcodes.txt
      count_mode: insertion
      max_fragment_size: 147
  B-cell_ATAC:
    path: /data/fragments.tsv.gz
    extractor_cls: cerberus.signal.FragmentExtractor
    extractor_args:
      barcodes_path: /data/b_cell_barcodes.txt
      count_mode: insertion
```

This follows the same pattern cerberus already uses for models, losses, and metrics — string-based class references resolved via `import_class()`.

### 3.2 Problem: `UniversalExtractor` Hardcodes Format Knowledge

`UniversalExtractor` (`signal.py:103-183`) routes by file extension:

```python
if suffix in ('.bw', '.bigwig'):
    self.bw_paths[name] = path
elif suffix in ('.bb', '.bigbed'):
    self.bb_paths[name] = path
elif suffix in ('.bed', '.bed.gz'):
    self.bed_paths[name] = path
```

Every new format requires editing this class. This is an Open-Closed Principle violation.

**Proposed change: Replace with `create_extractor` factory + `CompositeExtractor`**

```python
_EXTRACTOR_REGISTRY: dict[str, type[BaseSignalExtractor]] = {
    '.bw': SignalExtractor,
    '.bigwig': SignalExtractor,
    '.bb': BigBedMaskExtractor,
    '.bigbed': BigBedMaskExtractor,
    '.bed': BedMaskExtractor,
    '.bed.gz': BedMaskExtractor,
    '.tsv.gz': FragmentExtractor,
}

def create_extractor(
    channels: dict[str, Path | ChannelSpec],
    in_memory: bool = False,
) -> BaseSignalExtractor:
    """
    Factory that creates a CompositeExtractor from channel specs.
    Uses explicit extractor_cls if provided, falls back to extension-based lookup.
    """
    extractors: dict[str, BaseSignalExtractor] = {}

    for name, spec in channels.items():
        if isinstance(spec, (str, Path)):
            path = Path(spec)
            ext_cls = _EXTRACTOR_REGISTRY.get(path.suffix.lower())
            if ext_cls is None:
                raise ValueError(f"Unknown file extension: {path.suffix}")
            extractors[name] = ext_cls({name: path})
        else:
            cls = import_class(spec["extractor_cls"])
            extractors[name] = cls(**spec.get("extractor_args", {}))

    return CompositeExtractor(extractors)


class CompositeExtractor(BaseSignalExtractor):
    """Composes multiple single-channel extractors into a multi-channel extractor."""
    def __init__(self, extractors: dict[str, BaseSignalExtractor]):
        self.channels = sorted(extractors.keys())
        self.extractors = extractors

    def extract(self, interval: Interval) -> torch.Tensor:
        return torch.stack([
            self.extractors[name].extract(interval).squeeze(0)
            for name in self.channels
        ])
```

New formats never touch existing code — they register themselves or get specified explicitly in config. `CompositeExtractor` replaces `UniversalExtractor` with a simpler job: hold named extractors, stack their outputs.

### 3.3 Problem: `CerberusDataset.__init__` Mixes Creation and Usage

`CerberusDataset.__init__` (`dataset.py:50-185`) creates folds, samplers, extractors, **and** transforms. This means you can't change how extractors are created without modifying the Dataset class. The auto-creation logic at lines 140-160 hardcodes `UniversalExtractor`.

**Proposed change: Factor creation into a builder function**

`CerberusDataset` should only **receive** its dependencies, never create them:

```python
# CerberusDataset becomes a pure data-retrieval class
class CerberusDataset(Dataset):
    def __init__(
        self,
        sampler: Sampler,
        sequence_extractor: BaseSequenceExtractor | None,
        input_extractor: BaseSignalExtractor | None,
        target_extractor: BaseSignalExtractor | None,
        transforms: Compose,
        is_train: bool,
    ): ...

# Creation logic lives in a builder
def build_dataset(
    genome_config: GenomeConfig,
    data_config: DataConfig,
    sampler_config: SamplerConfig,
    in_memory: bool = False,
) -> CerberusDataset:
    folds = create_genome_folds(...)
    sampler = create_sampler(...)
    seq_extractor = create_sequence_extractor(genome_config, data_config, in_memory)
    input_extractor = create_extractor(data_config["inputs"], in_memory)
    target_extractor = create_extractor(data_config["targets"], in_memory)
    transforms = create_default_transforms(data_config)

    return CerberusDataset(
        sampler=sampler,
        sequence_extractor=seq_extractor,
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        transforms=transforms,
        is_train=True,
    )
```

This gives two clean usage paths:

1. **Config-driven** (YAML -> `build_dataset()`): auto-creates everything, supports fragments via `ChannelSpec`
2. **Manual** (Python API): construct your own extractors and pass them directly

### 3.4 Problem: `ReverseComplement` Hardcodes Channel Semantics

`ReverseComplement` (`transform.py`) assumes the first 4 channels are one-hot DNA:

```python
seq = inputs[:4]        # hardcoded slice
seq = seq.flip(0)       # reverse channel order (ACGT -> TGCA)
seq = seq.flip(1)       # reverse position order
```

This works when inputs are always `[A, C, G, T, signal1, signal2, ...]`. But the assumption that "first 4 = DNA" is baked into the transform rather than derived from configuration.

**Proposed change: Parameterize `n_seq_channels`**

```python
class ReverseComplement(DataTransform):
    def __init__(self, n_seq_channels: int = 4):
        self.n_seq = n_seq_channels

    def __call__(self, inputs, targets, interval):
        if random.random() < 0.5:
            seq = inputs[:self.n_seq]
            signals = inputs[self.n_seq:]
            seq = seq.flip(0).flip(1)
            signals = signals.flip(1)
            inputs = torch.cat([seq, signals], dim=0)
            targets = targets.flip(1)
            interval.strand = '-' if interval.strand == '+' else '+'
        return inputs, targets, interval
```

`create_default_transforms` would derive `n_seq_channels` from `data_config["use_sequence"]` (4 if True, 0 if False).

---

## 4. Impact Assessment

| Change | Lines Touched | Risk | Impact |
|---|---|---|---|
| `ChannelSpec` in DataConfig | ~40 (config.py) | Low — backward compatible | High — enables YAML-driven fragment configs |
| ExtractorFactory replacing UniversalExtractor | ~80 (signal.py) | Medium — touches data path | High — open/closed for new formats |
| Factor creation out of CerberusDataset | ~100 (dataset.py + datamodule.py) | Medium — restructure, same behavior | Medium — cleaner separation of concerns |
| Parameterize ReverseComplement | ~10 (transform.py) | Low — additive | Low — removes hardcoded assumption |

Changes 1 and 2 are highest value. They make fragments (and any future format — BAM, HDF5, Zarr) a first-class citizen without special-casing. Change 3 is good hygiene that makes 1 and 2 cleaner. Change 4 is small but prevents subtle bugs.

The guiding principle: **extractors should follow the same pattern as models** — string-based class references, factory instantiation, config-driven. Cerberus already has this pattern; it's just not applied to the data loading side yet.

---

## 5. References

- [SnapATAC2](https://github.com/scverse/SnapATAC2) — Rust-backed scATAC-seq toolkit (scverse ecosystem)
- [SnapATAC2 export_coverage API](https://scverse.org/SnapATAC2/version/dev/api/_autosummary/snapatac2.ex.export_coverage.html)
- [pycisTopic](https://pycistopic.readthedocs.io/en/latest/features.html) — Topic modeling for scATAC-seq
- [muon](https://muon.scverse.org/) — Multimodal omics framework (scverse ecosystem)
- [scooby](https://github.com/gagneurlab/scooby) — Single-cell sequence-to-function model ([Nature Methods](https://www.nature.com/articles/s41592-025-02854-5))
- [scBasset](https://docs.scvi-tools.org/en/stable/user_guide/models/scbasset.html) — Sequence-based scATAC modeling
- [PeakVI](https://www.sciencedirect.com/science/article/pii/S2667237522000376) — Deep generative model for scATAC
- [EpiScanpy](https://episcanpy.readthedocs.io/en/latest/) — Epigenomics single-cell analysis
- [scATAC-seq analysis notes](https://github.com/mdozmorov/scATAC-seq_notes) — Community-curated tool list
