from dataclasses import dataclass

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
