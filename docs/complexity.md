# Sequence Complexity Metrics

The `cerberus.complexity` module provides functions to calculate sequence complexity scores, including GC content and DUST score.

## Functions

### `calculate_gc_content`

Calculates the GC content (fraction of G and C nucleotides) of a sequence.

**Note:** Ambiguous bases (like 'N') are excluded from the calculation. The score is the fraction of G+C relative to the total count of A, C, G, and T bases.

**Signature:**
```python
def calculate_gc_content(sequence: str) -> float
```

**Args:**
*   `sequence`: Input DNA sequence (string).

**Returns:**
*   `float`: GC content ratio (0.0 to 1.0).

---

### `calculate_dust_score`

Calculates the DUST score, a measure of low-complexity regions based on k-mer repetition.

**Signature:**
```python
def calculate_dust_score(
    sequence: str,
    k: int = 3,
    normalize: bool = True
) -> float
```

**Formula:**
\[ Score = \frac{\sum_{i} c_i (c_i - 1) / 2}{L - k + 1} \]
where \(c_i\) is the count of the \(i\)-th unique k-mer in the sequence, and \(L\) is the sequence length.

**Behavior:**
*   **Input:** Supports single strings only.
*   **k:** The k-mer length (default 3). Must be \(\le 5\).
*   **Non-ACGT Characters:** Characters like 'N' are treated as a distinct 5th base type (index 4). For example, "NNN" is treated as a repeat of the "N" base, contributing to the complexity score.
*   **Interpretation:** Higher scores indicate lower complexity (more repetition). A sequence like "AAAAAAAA" has a high DUST score, while a random sequence has a low score.
*   **Normalization:** If `normalize=True`, the raw score is transformed using \(\tanh(\log(\text{Ratio}) / 1.5)\), where \(\text{Ratio} = (\text{RawScore} + \epsilon) / \text{ExpRandom}\). This maps the unbounded score `[0, inf)` to `[0, 1)`. The scaling factor `k=1.5` spreads out scores for sequences with intermediate complexity (ratios 1.5–2.5), pushing them into the 0.2–0.4 range, while simple repeats saturate near 1.0.

**Example:**
```python
from cerberus.complexity import calculate_dust_score

# Single sequence
score = calculate_dust_score("AAAAAAAA", k=3)
# Returns: ~0.99 (Normalized)
```

---

### `calculate_log_cpg_ratio`

Calculates the normalized Observed/Expected CpG ratio score.

**Signature:**
```python
def calculate_log_cpg_ratio(
    sequence: str,
    epsilon: float = 1.0,
    normalize: bool = True
) -> float
```

**Formula (Unnormalized):**
\[ Val = \log_2 \left( \frac{\text{Obs}_{\text{CG}} + \epsilon}{\text{Exp}_{\text{CG}} + \epsilon} \right) \]
where:
*   \(\text{Obs}_{\text{CG}}\) is the count of CG dinucleotides.
*   \(\text{Exp}_{\text{CG}} = \frac{\text{Count(C)} \times \text{Count(G)}}{L}\) (where \(L\) is the full sequence length, including Ns).

**Behavior:**
*   **Input:** Supports single strings only.
*   **Normalization:** If `normalize=True` (default), the log-ratio \( Val \) is transformed using a logistic sigmoid: \[ Score = \frac{\tanh(Val / 2) + 1}{2} \]
    *   **Range:** `[0, 1]`
    *   **Neutral (Val=0, Ratio=1):** Maps to `0.5`.
    *   **Enriched (Val>0):** Maps to `(0.5, 1]`. Typical CpG islands (>0.6 ratio) map to >0.7.
    *   **Depleted (Val<0):** Maps to `[0, 0.5)`.
*   **Epsilon:** A smoothing factor (default `1.0`) to prevent division by zero or log of zero and to dampen ratios for sequences with low counts.

**Example:**
```python
from cerberus.complexity import calculate_log_cpg_ratio

# CpG Island-like
score = calculate_log_cpg_ratio("CGCGCGCG")
# Returns: ~0.64 (Enriched)

# Depleted
score = calculate_log_cpg_ratio("CCTTTTGG")
# Returns: ~0.36 (Depleted)
```

---

### `compute_intervals_complexity`

Computes selected complexity metrics for a collection of intervals.

**Signature:**
```python
def compute_intervals_complexity(
    intervals: Iterable[Interval],
    fasta_path: Path | str,
    metrics: list[str] | None = None,
    center_size: int | None = None,
) -> np.ndarray
```

**Args:**
*   `intervals`: Iterable of Interval objects.
*   `fasta_path`: Path to the genome FASTA file.
*   `metrics`: List of metrics to compute. Options: 'gc', 'dust', 'cpg'. If None, computes all.
*   `center_size`: If set, crop each interval to its center N bp before computing
    metrics. Intervals smaller than `center_size` are left unchanged. This is useful
    for large-context models (32kb+) where full-interval metrics regress toward the
    genome mean, making complexity matching ineffective.

**Returns:**
*   `np.ndarray`: A (N, M) array where columns correspond to the requested metrics in order.

---

### `get_bin_index`

Computes the multi-dimensional bin index for a given metric row.

**Signature:**
```python
def get_bin_index(row: np.ndarray, bins: int) -> tuple[int, ...] | None
```

**Args:**
*   `row`: A 1D numpy array representing metrics for a single interval.
*   `bins`: The number of bins per dimension.

**Returns:**
*   A tuple of integers representing the bin coordinates, or None if the row contains NaNs.

---

### `compute_hist`

Computes a histogram of metric occurrences across defined bins.

**Signature:**
```python
def compute_hist(metrics: np.ndarray, bins: int) -> dict[tuple[int, ...], int]
```

**Args:**
*   `metrics`: A numpy array of shape (N, D) or (N,), where N is samples and D is dimensions.
*   `bins`: The number of bins per dimension.

**Returns:**
*   A dictionary mapping bin indices (tuples) to counts.
