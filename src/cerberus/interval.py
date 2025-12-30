from dataclasses import dataclass
from pathlib import Path

from cerberus.config import GenomeConfig

@dataclass
class Interval:
    """
    Represents a genomic interval.

    This is the fundamental data structure used throughout Cerberus to define genomic regions.
    It supports 0-based, half-open coordinates [start, end), consistent with BED files and Python slicing.

    Attributes:
        chrom: Chromosome name (e.g., 'chr1', '1', 'X').
        start: Start position (0-based, inclusive).
        end: End position (0-based, exclusive).
        strand: Strand ('+' or '-'). Default is '+'.
    """
    chrom: str
    start: int
    end: int
    strand: str = "+"

    def __len__(self):
        """Returns the length of the interval (end - start)."""
        return self.end - self.start

    def __str__(self):
        """Returns a string representation: 'chrom:start-end(strand)'."""
        return f"{self.chrom}:{self.start}-{self.end}({self.strand})"

    def center(self, width: int) -> "Interval":
        """
        Returns a new interval of the specified width, centered within this interval.
        
        Args:
            width: The desired length of the new interval.
            
        Returns:
            A new Interval object centered on the current one.
        """
        current_len = len(self)
        offset = (current_len - width) // 2
        new_start = self.start + offset
        new_end = new_start + width
        return Interval(self.chrom, new_start, new_end, self.strand)


def parse_intervals(
    intervals: list[str], interval_paths: list[Path], genome_config: "GenomeConfig"
) -> list[Interval]:
    """
    Parses intervals from a list of strings and paths to BED files.

    Args:
        intervals: List of strings (e.g. ["chr1", "chr2:1000-2000"]).
        interval_paths: List of paths to BED files.
        genome_config: Genome configuration containing chromosome sizes.

    Returns:
        List of Interval objects.
    """
    parsed = []
    chrom_sizes = genome_config["chrom_sizes"]

    # If both lists are empty, default to whole genome
    if not intervals and not interval_paths:
        for chrom in genome_config["allowed_chroms"]:
            if chrom in chrom_sizes:
                parsed.append(Interval(chrom, 0, chrom_sizes[chrom]))
        return parsed

    # Process intervals from strings
    for item in intervals:
        if ":" in item:
            try:
                chrom, coords = item.split(":")
                start, end = map(int, coords.split("-"))
                parsed.append(Interval(chrom, start, end))
            except ValueError:
                raise ValueError(
                    f"Invalid interval format: {item}. Expected 'chr:start-end'."
                )
        else:
            chrom = item
            if chrom not in chrom_sizes:
                raise ValueError(f"Chromosome {chrom} not found in genome config.")
            parsed.append(Interval(chrom, 0, chrom_sizes[chrom]))

    # Process intervals from files
    for p in interval_paths:
        with open(p) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    parsed.append(Interval(parts[0], int(parts[1]), int(parts[2])))

    return parsed


def merge_intervals(
    intervals: list[Interval]
) -> list[Interval]:
    """
    Sorts and merges overlapping or adjacent intervals.
    """
    if not intervals:
        return []

    # Sort by chrom, then start
    sorted_intervals = sorted(intervals, key=lambda x: (x.chrom, x.start))

    merged = []
    current_interval = sorted_intervals[0]
    current_chrom = current_interval.chrom
    current_start = current_interval.start
    current_end = current_interval.end

    for i in range(1, len(sorted_intervals)):
        interval = sorted_intervals[i]
        chrom = interval.chrom
        start = interval.start
        end = interval.end

        if chrom == current_chrom and start <= current_end:
            # Overlap or adjacent, merge
            current_end = max(current_end, end)
        else:
            merged.append(Interval(current_chrom, current_start, current_end))
            current_chrom = chrom
            current_start = start
            current_end = end

    merged.append(Interval(current_chrom, current_start, current_end))
    return merged
