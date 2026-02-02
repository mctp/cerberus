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
*   `float`: The GC content ratio (0.0 to 1.0).

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
*   **Normalization:** If `normalize=True`, the raw score is transformed using \(\tanh(\ln(\text{Score} + 1) / 2)\). This maps the unbounded score `[0, inf)` to `[0, 1)`, providing a non-linear scaling that saturates for highly repetitive sequences.

**Example:**
```python
from cerberus.complexity import calculate_dust_score

# Single sequence
score = calculate_dust_score("AAAAAAAA", k=3)
# Returns: 2.5
```

---

### `calculate_log_cpg_ratio`

Calculates the log-transformed Observed/Expected CpG ratio.

**Signature:**
```python
def calculate_log_cpg_ratio(
    sequence: str,
    epsilon: float = 1.0,
    normalize: bool = True
) -> float
```

**Formula:**
\[ Score = \log_2 \left( \frac{\text{Obs}_{\text{CG}} + \epsilon}{\text{Exp}_{\text{CG}} + \epsilon} \right) \]
where:
*   \(\text{Obs}_{\text{CG}}\) is the count of CG dinucleotides.
*   \(\text{Exp}_{\text{CG}} = \frac{\text{Count(C)} \times \text{Count(G)}}{L}\) (where \(L\) is the full sequence length, including Ns).

**Behavior:**
*   **Input:** Supports single strings only.
*   **Non-ACGT Characters:** 'N's (and other non-ACGT characters) are included in the sequence length \(L\), which dilutes the expected CpG count compared to a sequence of only valid bases.
*   **Values:**
    *   `0.0`: Neutral (Observed ≈ Expected).
    *   `> 0`: Enriched (CpG Island-like).
    *   `< 0`: Depleted (Methylation suppression).
*   **Normalization:** If `normalize=True`, the score is transformed using \(( \tanh(\text{score} / 2) + 1 ) / 2\). This maps the result to \((0, 1)\), with 0.5 representing neutral.
*   **Epsilon:** A smoothing factor (default `1.0`) to prevent division by zero or log of zero and to dampen ratios for sequences with low counts.

**Example:**
```python
from cerberus.complexity import calculate_log_cpg_ratio

# CpG Island-like
score = calculate_log_cpg_ratio("CGCGCGCG")
# Returns: ~1.0 (2x enrichment)

# Depleted
score = calculate_log_cpg_ratio("CCTTTTGG")
# Returns: Very negative
```
