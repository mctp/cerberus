import gzip
import heapq
import logging
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

from interlap import InterLap

logger = logging.getLogger(__name__)

from cerberus.config import GenomeConfig

# Filenames of the packaged Borzoi cross-validation fold definitions, keyed by
# species.  These are the upstream ``sequences_{species}.bed.gz`` files from
# calico/borzoi (8 folds, 196,608 bp windows), usable directly as the
# ``path`` argument of the ``"bed_partition"`` fold strategy.
_FOLD_BED_FILES = {
    "human": "sequences_human.bed.gz",
    "mouse": "sequences_mouse.bed.gz",
}


def fold_bed_path(species: str) -> Path:
    """Return the path to a packaged Borzoi cross-validation fold BED.

    The returned file is a 4-column ``(chrom, start, end, fold_label)`` BED
    (gzip-compressed) with ``fold0``..``fold7`` labels, suitable for the
    ``"bed_partition"`` fold strategy:

        >>> from cerberus import create_genome_config, fold_bed_path
        >>> cfg = create_genome_config(
        ...     name="hg38", fasta_path=fasta, species="human",
        ...     fold_type="bed_partition",
        ...     fold_args={"k": 8, "path": str(fold_bed_path("human")),
        ...                "test_fold": 3, "val_fold": 4},
        ... )

    Args:
        species: ``"human"`` or ``"mouse"`` (case-insensitive).

    Returns:
        Filesystem path to the packaged fold BED.

    Raises:
        ValueError: If no packaged fold BED exists for *species*.
    """
    fname = _FOLD_BED_FILES.get(species.lower())
    if fname is None:
        raise ValueError(
            f"No packaged fold BED for species {species!r}. "
            f"Available: {sorted(_FOLD_BED_FILES)}"
        )
    return Path(str(files("cerberus.data").joinpath(fname)))


_HUMAN_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]
_MOUSE_CHROMS = [str(i) for i in range(1, 20)] + ["X", "Y"]


def _make_sort_key(order_list: list[str]) -> Callable[[str], tuple]:
    """Creates a sort key function based on a list of chromosomes."""
    order_map = {c: i for i, c in enumerate(order_list)}

    def sort_key(chrom: str) -> tuple:
        base = chrom[3:] if chrom.startswith("chr") else chrom
        if base in order_map:
            return (0, order_map[base])
        return (1, base)

    return sort_key


_human_sort_key = _make_sort_key(_HUMAN_CHROMS + ["M", "MT"])
_mouse_sort_key = _make_sort_key(_MOUSE_CHROMS + ["M", "MT"])

# Configuration map for species-specific logic
_SPECIES_CONFIG = {
    "human": {"standard_chroms": _HUMAN_CHROMS, "sort_key": _human_sort_key},
    "mouse": {"standard_chroms": _MOUSE_CHROMS, "sort_key": _mouse_sort_key},
}


def create_genome_config(
    name: str,
    fasta_path: Path,
    species: str,
    fold_type: str = "chrom_partition",
    fold_args: dict[str, Any] | None = None,
    allowed_chroms: list[str] | None = None,
    exclude_intervals: dict[str, Path] | None = None,
) -> GenomeConfig:
    """
    Creates a valid GenomeConfig by parsing the FASTA index.

    This function reads the FASTA index (.fai) to determine available chromosomes and their sizes,
    filters them based on the species and allowed chromosomes, and constructs the configuration dictionary.

    Args:
        name: Name of the genome assembly (e.g., 'hg38').
        fasta_path: Path to the FASTA file. The .fai index is expected to exist at `<fasta_path>.fai`.
        species: Species name (e.g., 'human', 'mouse'). Must be one of the supported species.
        fold_type: Strategy for creating folds (default: 'chrom_partition').
        fold_args: Arguments for the folding strategy (default: {'k': 5}).
        allowed_chroms: Optional list of chromosomes to include. If None, defaults to standard chromosomes
                        for the species (e.g., 1-22, X, Y for human).
        exclude_intervals: Optional dictionary mapping names to BED files of regions to exclude.

    Returns:
        GenomeConfig: A dictionary containing the genome configuration, including chromosome sizes and paths.

    Raises:
        NotImplementedError: If the species is not supported.
        FileNotFoundError: If the FASTA file or its index is not found.
    """
    species_key = species.lower()
    if species_key not in _SPECIES_CONFIG:
        raise NotImplementedError(
            f"Species '{species}' is not supported. Supported: {list(_SPECIES_CONFIG.keys())}"
        )

    spec_config = _SPECIES_CONFIG[species_key]
    standard_chroms = spec_config["standard_chroms"]
    sort_key = spec_config["sort_key"]

    fasta_path = Path(fasta_path)
    fai_path = fasta_path.with_suffix(fasta_path.suffix + ".fai")

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if not fai_path.exists():
        raise FileNotFoundError(f"FASTA index not found: {fai_path}")

    # 1. Read available chromosomes
    available = {}
    with open(fai_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                available[parts[0]] = int(parts[1])

    # 2. Determine allowed chromosomes
    if allowed_chroms is None:
        # Default to standard chromosomes (excluding M/MT by default)
        allowed_chroms = []

        for c in standard_chroms:
            if c in available:
                allowed_chroms.append(c)
            elif f"chr{c}" in available:
                allowed_chroms.append(f"chr{c}")

    # 3. Filter and build chrom_sizes
    chrom_sizes = {}
    for c in allowed_chroms:
        if c in available:
            chrom_sizes[c] = available[c]

    # 4. Sort
    sorted_chroms = dict(sorted(chrom_sizes.items(), key=lambda x: sort_key(x[0])))

    if fold_args is None:
        fold_args_resolved = {"k": 5}
    else:
        fold_args_resolved = fold_args

    logger.debug(
        f"Created genome config for {name} ({species}) with {len(sorted_chroms)} chromosomes."
    )

    return GenomeConfig(
        name=name,
        fasta_path=fasta_path,
        exclude_intervals=exclude_intervals or {},
        allowed_chroms=list(sorted_chroms.keys()),
        chrom_sizes=sorted_chroms,
        fold_type=fold_type,
        fold_args=fold_args_resolved,
    )


def create_genome_folds(
    chrom_sizes: dict[str, int], fold_type: str, fold_args: dict[str, Any]
) -> list[dict[str, InterLap]]:
    """
    Creates genome folds based on the specified strategy.

    Folds are mutually exclusive sets of genomic intervals used for cross-validation.

    Args:
        chrom_sizes: Dictionary mapping chromosome names to their lengths.
        fold_type: Strategy for creating folds. Supported: ``'chrom_partition'``
            (greedy size-balanced split of whole chromosomes) and
            ``'bed_partition'`` (region-level split read from a BED file).
        fold_args: Arguments for the folding strategy (plain dict).
            ``'chrom_partition'``: ``k`` (int).
            ``'bed_partition'``: ``k`` (int), ``path`` (str path to a 4-column
            ``(chrom, start, end, fold_id)`` BED, optionally gzipped).

    Returns:
        list[dict[str, InterLap]]: A list of k dictionaries. Each dictionary maps chromosome names
        to InterLap objects representing the included intervals for that fold.

    Raises:
        ValueError: If `fold_type` is unknown.
    """
    if fold_type == "chrom_partition":
        k = fold_args["k"]
        return _create_folds_chrom_partition(chrom_sizes, k)
    elif fold_type == "bed_partition":
        return _create_folds_bed_partition(
            chrom_sizes, Path(fold_args["path"]), fold_args["k"]
        )
    else:
        raise ValueError(f"Unknown fold_type: {fold_type}")


def _parse_fold_id(token: str) -> int:
    """Parse a fold-id BED token into an integer.

    Accepts a bare integer (``"3"``) or a Borzoi-style label (``"fold3"`` /
    ``"fold_3"``).  This lets the upstream ``sequences_{species}.bed.gz`` files
    be used verbatim.
    """
    t = token.strip()
    low = t.lower()
    if low.startswith("fold"):
        t = low[4:].lstrip("_")
    try:
        return int(t)
    except ValueError as exc:
        raise ValueError(
            f"Invalid fold id {token!r} (expected int or 'foldN')"
        ) from exc


def _create_folds_bed_partition(
    chrom_sizes: dict[str, int], path: Path, k: int
) -> list[dict[str, InterLap]]:
    """Build region-level folds from a 4-column BED file.

    The BED has columns ``(chrom, start, end, fold_id)`` where ``fold_id`` is an
    integer in ``[0, k)`` (or a ``fold<N>`` label).  Half-open ``[start, end)``
    rows are stored as closed ``[start, end-1]`` intervals (matching
    :mod:`cerberus.exclude`).  Multiple rows may map to the same fold; rows of a
    single fold may overlap each other freely (e.g. Borzoi's overlapping
    sliding windows).  The genome need not be fully covered — intervals whose
    centre falls outside every fold are dropped at split time.

    Args:
        chrom_sizes: Allowed chromosomes and their lengths.  Rows on other
            chromosomes are skipped with a warning.
        path: Path to the BED (optionally gzipped).
        k: Number of folds.  ``fold_id`` values must lie in ``[0, k)``.

    Returns:
        A length-``k`` list of ``{chrom: InterLap}`` dicts (empty dicts for
        folds with no rows).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If ``k <= 0``, a fold id is out of range, or two *different*
            folds overlap (which would make fold ownership ambiguous).
    """
    if k <= 0:
        raise ValueError(f"bed_partition requires k > 0, got {k}")
    if not path.exists():
        raise FileNotFoundError(f"Fold BED not found: {path}")

    folds: list[dict[str, InterLap]] = [{} for _ in range(k)]
    skipped_chroms: set[str] = set()
    n_rows = 0

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"bed_partition expects >=4 columns (chrom start end fold_id); "
                    f"got {len(parts)}: {line!r}"
                )
            chrom, start_s, end_s, fold_s = parts[0], parts[1], parts[2], parts[3]
            if chrom not in chrom_sizes:
                skipped_chroms.add(chrom)
                continue
            start, end = int(start_s), int(end_s)
            if end <= start:
                continue
            fold_id = _parse_fold_id(fold_s)
            if not 0 <= fold_id < k:
                raise ValueError(
                    f"fold id {fold_id} out of range [0, {k}) in {path}: {line!r}"
                )
            if chrom not in folds[fold_id]:
                folds[fold_id][chrom] = InterLap()
            # Half-open [start, end) -> closed [start, end-1].
            folds[fold_id][chrom].add((start, end - 1))
            n_rows += 1

    if skipped_chroms:
        logger.warning(
            "bed_partition: skipped %d row group(s) on chromosomes absent from "
            "chrom_sizes: %s",
            len(skipped_chroms),
            sorted(skipped_chroms),
        )

    _validate_cross_fold_disjoint(folds, path)

    n_nonempty = sum(1 for fold in folds if fold)
    logger.info(
        "bed_partition: loaded %d region(s) into %d/%d non-empty fold(s) from %s",
        n_rows,
        n_nonempty,
        k,
        path,
    )
    return folds


def _validate_cross_fold_disjoint(folds: list[dict[str, InterLap]], path: Path) -> None:
    """Raise if any two *different* folds claim an overlapping region.

    Overlap *within* a single fold is allowed (and expected for dense sliding
    windows); only cross-fold overlap is an error, because it makes the
    centre-based fold ownership used by ``partition_intervals_by_fold``
    ambiguous.
    """
    for i in range(len(folds)):
        for chrom, tree_i in folds[i].items():
            for j in range(i + 1, len(folds)):
                tree_j = folds[j].get(chrom)
                if tree_j is None:
                    continue
                for s, e in tree_i:
                    if (s, e) in tree_j:
                        raise ValueError(
                            f"bed_partition: folds {i} and {j} overlap on {chrom} "
                            f"near {s}-{e + 1} in {path}; fold regions of different "
                            "folds must be disjoint."
                        )


def _create_folds_chrom_partition(
    chrom_sizes: dict[str, int], k: int
) -> list[dict[str, InterLap]]:
    """
    Distribute chromosomes into roughly equal sized folds using a greedy approach.
    Returns a list of dictionaries, where each dictionary maps chromosome names
    to InterLap objects representing the included intervals for that fold.
    """
    # Sort chromosomes by size descending
    sorted_chroms = sorted(chrom_sizes.items(), key=lambda x: x[1], reverse=True)

    # Priority queue to hold (current_size, fold_index, chrom_list)
    # We want to add next chrom to the smallest fold
    folds = [(0, i, []) for i in range(k)]
    heapq.heapify(folds)

    for chrom, size in sorted_chroms:
        current_size, fold_index, chrom_list = heapq.heappop(folds)
        chrom_list.append(chrom)
        current_size += size
        heapq.heappush(folds, (current_size, fold_index, chrom_list))

    # Extract just the chromosome lists, sorted by fold index
    folds.sort(key=lambda x: x[1])

    # Convert to interval maps
    result_folds = []
    for _, _, chrom_list in folds:
        fold_intervals = {}
        for chrom in chrom_list:
            if chrom not in chrom_sizes:
                continue

            # Create InterLap for the whole chromosome
            # Use closed coordinates [start, end-1]
            il = InterLap()
            il.add((0, chrom_sizes[chrom] - 1))
            fold_intervals[chrom] = il
        result_folds.append(fold_intervals)

    return result_folds


def create_human_genome_config(
    genome_dir: Path | str,
    fold_type: str = "chrom_partition",
    fold_args: dict[str, Any] | None = None,
    allowed_chroms: list[str] | None = None,
) -> GenomeConfig:
    """
    Creates GenomeConfig from a downloaded bundle (specifically for human hg38).

    This helper function assumes the directory structure created by `cerberus.download.download_human_reference`.
    It automatically locates 'hg38.fa', 'blacklist.bed', and 'gaps.bed'.

    Args:
        genome_dir: Directory containing the downloaded resources.
        fold_type: Strategy for creating folds (default: 'chrom_partition').
        fold_args: Arguments for the folding strategy (default: {'k': 5}).
        allowed_chroms: Optional list of chromosomes to include.

    Returns:
        GenomeConfig: Config object populated with FASTA path, exclude intervals, and other settings.

    Raises:
        FileNotFoundError: If the FASTA file is not found in the directory.
    """
    d = Path(genome_dir)
    fasta = d / "hg38.fa"
    blacklist = d / "blacklist.bed"
    gaps = d / "gaps.bed"

    if not fasta.exists():
        raise FileNotFoundError(
            f"FASTA not found at {fasta}. Please run cerberus.download.download_human_reference(..., name='{d.name}') first."
        )

    exclude_intervals = {}
    if blacklist.exists():
        exclude_intervals["blacklist"] = blacklist
    if gaps.exists():
        exclude_intervals["unmappable"] = gaps

    return create_genome_config(
        name=d.name,
        fasta_path=fasta,
        species="human",
        fold_type=fold_type,
        fold_args=fold_args,
        allowed_chroms=allowed_chroms,
        exclude_intervals=exclude_intervals,
    )


def create_mouse_genome_config(
    genome_dir: Path | str,
    fold_type: str = "chrom_partition",
    fold_args: dict[str, Any] | None = None,
    allowed_chroms: list[str] | None = None,
) -> GenomeConfig:
    """
    Creates GenomeConfig from a downloaded bundle (specifically for mouse mm10).

    This helper function assumes the directory structure created by `cerberus.download.download_reference_genome(..., genome='mm10')`.
    It automatically locates 'mm10.fa', 'blacklist.bed', and 'gaps.bed'.

    Args:
        genome_dir: Directory containing the downloaded resources.
        fold_type: Strategy for creating folds (default: 'chrom_partition').
        fold_args: Arguments for the folding strategy (default: {'k': 5}).
        allowed_chroms: Optional list of chromosomes to include.

    Returns:
        GenomeConfig: Config object populated with FASTA path, exclude intervals, and other settings.

    Raises:
        FileNotFoundError: If the FASTA file is not found in the directory.
    """
    d = Path(genome_dir)
    fasta = d / "mm10.fa"
    blacklist = d / "blacklist.bed"
    gaps = d / "gaps.bed"

    if not fasta.exists():
        raise FileNotFoundError(
            f"FASTA not found at {fasta}. Please run cerberus.download.download_reference_genome(..., genome='mm10') first."
        )

    exclude_intervals = {}
    if blacklist.exists():
        exclude_intervals["blacklist"] = blacklist
    if gaps.exists():
        exclude_intervals["unmappable"] = gaps

    return create_genome_config(
        name=d.name,
        fasta_path=fasta,
        species="mouse",
        fold_type=fold_type,
        fold_args=fold_args,
        allowed_chroms=allowed_chroms,
        exclude_intervals=exclude_intervals,
    )
