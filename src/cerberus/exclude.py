from pathlib import Path
import logging
from interlap import InterLap

logger = logging.getLogger(__name__)

# NOTE: InterLap uses CLOSED coordinates [start, end].
# We convert standard HALF-OPEN genomic intervals [start, end) to [start, end-1]
# for storage and querying to ensure correct overlap detection.


def get_exclude_intervals(
    exclude_intervals: dict[str, Path],
) -> dict[str, InterLap]:
    """
    Parses BED files to create InterLap objects representing excluded regions.

    Args:
        exclude_intervals: A dictionary mapping names (e.g., 'blacklist') to file paths (BED format).
                           Regions in these files are added to the exclusion list.

    Returns:
        dict[str, InterLap]: A dictionary mapping chromosome names to InterLap objects.
                             Each InterLap object contains the union of all excluded intervals for that chromosome.
    
    Raises:
        FileNotFoundError: If any of the specified files do not exist.
    """
    intervals: dict[str, InterLap] = {}

    logger.debug(f"Loading {len(exclude_intervals)} exclude interval file(s)...")
    for path in exclude_intervals.values():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Exclude file not found: {path}")

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", "track", "browser")):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue

                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue

                if end <= start:
                    continue

                if chrom not in intervals:
                    intervals[chrom] = InterLap()

                # Convert half-open [start, end) to closed [start, end-1] for InterLap
                intervals[chrom].add((start, end - 1))

    total = sum(len(list(il)) for il in intervals.values())
    logger.debug(f"Loaded {total} exclude intervals across {len(intervals)} chromosome(s)")
    return intervals


def is_excluded(
    exclude_intervals: dict[str, InterLap],
    chrom: str,
    start: int,
    end: int,
) -> bool:
    """
    Checks if a genomic region overlaps with any excluded region.

    Args:
        exclude_intervals: Dictionary mapping chromosome names to InterLap objects.
        chrom: Chromosome name.
        start: Start position (0-based inclusive).
        end: End position (0-based exclusive).

    Returns:
        bool: True if the region [start, end) overlaps with any interval in exclude_intervals[chrom].
              False otherwise.
    """
    if not exclude_intervals or chrom not in exclude_intervals:
        return False

    # Convert query interval [start, end) to [start, end-1]
    return (start, end - 1) in exclude_intervals[chrom]
