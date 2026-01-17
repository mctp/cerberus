# Mouse Genome (mm10) Support Plan

## Overview
This document outlines the plan to add support for the mouse genome (mm10) to Cerberus. currently, the codebase handles `hg38` via specific functions. We will generalize these to support `mm10` and potentially other genomes in the future.

## Key Changes

### 1. `src/cerberus/download.py`
Refactor `download_human_reference` into a generic `download_reference_genome` function. This function will use a configuration dictionary to determine which URLs to download for a given genome.

**Resources for mm10:**
*   **FASTA**: `http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz`
*   **Blacklist**: `https://github.com/Boyle-Lab/Blacklist/raw/master/lists/mm10-blacklist.v2.bed.gz`
*   **Gaps**: `http://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/gap.txt.gz`
*   **cCREs**: `https://hgdownload.soe.ucsc.edu/gbdb/mm10/encode3/ccre/encodeCcreCombined.bb`
*   **Mappability**: *To be determined*. No direct Hoffman track equivalent found. Will be optional.

### 2. `src/cerberus/genome.py`
*   Add `create_mouse_genome_config` function.
*   Ensure `create_genome_config` correctly handles `species="mouse"`.
*   Verify `_SPECIES_CONFIG` contains the correct chromosome lists for mouse.

## Implementation Steps

1.  **Modify `src/cerberus/download.py`**:
    *   Define `GENOME_RESOURCES` dictionary.
    *   Implement `download_reference_genome`.
    *   Update `download_human_reference` to call `download_reference_genome` (or alias it).

2.  **Modify `src/cerberus/genome.py`**:
    *   Implement `create_mouse_genome_config`.

3.  **Testing**:
    *   Verify downloads (using mocks where appropriate).
    *   Verify configuration generation.

## Technical Considerations
*   **Hardcoded paths**: Ensure file names (e.g., `mm10.fa`) are generated dynamically based on the genome name.
*   **Backward Compatibility**: Ensure existing calls to `download_human_reference` still work as expected.
