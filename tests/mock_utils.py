from typing import Protocol

import numpy as np
import torch

from cerberus.interval import Interval
from cerberus.samplers import ListSampler
from cerberus.sequence import BaseSequenceExtractor, SequenceExtractor
from cerberus.signal import BaseSignalExtractor


class MotifInserter(Protocol):
    """Protocol for inserting motifs into a sequence tensor."""
    def __call__(self, sequence: torch.Tensor, interval: Interval) -> torch.Tensor: ...

class SignalGenerator(Protocol):
    """Protocol for generating signal from a sequence tensor."""
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor: ...

class MockSampler(ListSampler):
    """
    Generates synthetic intervals for testing.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        chroms: list[str] | None = None,
        chrom_size: int = 1_000_000,
        interval_length: int = 200,
        seed: int = 42
    ):
        self.num_samples = num_samples
        self.chroms = chroms if chroms is not None else ["chr1"]
        self.chrom_size = chrom_size
        self.interval_length = interval_length
        self.seed = seed
        self.chrom_sizes = {chrom: chrom_size for chrom in self.chroms}
        self.folds = [] # Not used by MockSampler's custom split_folds
        self.exclude_intervals = {} # Not used
        
        self._intervals: list[Interval] = []
        self._generate_intervals()

    def _generate_intervals(self):
        rng = np.random.RandomState(self.seed)
        self._intervals = []
        for _ in range(self.num_samples):
            chrom = rng.choice(self.chroms)
            # Ensure valid range
            max_start = self.chrom_size - self.interval_length
            if max_start < 0:
                raise ValueError("Interval length larger than chromosome size")
            
            start = rng.randint(0, max_start)
            end = start + self.interval_length
            self._intervals.append(Interval(chrom, start, end, "+"))

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple["ListSampler", "ListSampler", "ListSampler"]:
        """
        Randomly splits the mock samples into train/val/test.
        Ignores fold arguments and does a simple 80/10/10 split for simplicity,
        or we can respect the fold indices if we had defined folds.
        
        For mock purposes, we'll just split indices deterministically.
        """
        n = len(self._intervals)
        indices = np.arange(n)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)
        
        # 80/10/10 split
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        return (
            self._subset(train_idx.tolist()),
            self._subset(val_idx.tolist()),
            self._subset(test_idx.tolist()),
        )

class MockSequenceExtractor(BaseSequenceExtractor):
    """
    Generates synthetic sequence or wraps a real extractor, with motif injection.
    """
    def __init__(
        self,
        fasta_path: str | None = None,
        encoding: str = "ACGT",
        motif_inserters: list[MotifInserter] | None = None,
        background_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    ):
        self.encoding = encoding
        self.motif_inserters = motif_inserters or []
        self.background_probs = background_probs
        
        if fasta_path:
            self.real_extractor = SequenceExtractor(fasta_path, encoding)
        else:
            self.real_extractor = None

    def extract(self, interval: Interval) -> torch.Tensor:
        length = interval.end - interval.start
        
        if self.real_extractor:
            seq_tensor = self.real_extractor.extract(interval)
        else:
            # Generate random background
            # Use deterministic seed based on interval for reproducibility
            seed = hash((interval.chrom, interval.start, interval.end)) % (2**32)
            rng = np.random.RandomState(seed)
            
            # 0=A, 1=C, 2=G, 3=T (assuming ACGT)
            # We construct a tensor (4, L)
            # First generate indices
            indices = rng.choice(4, size=length, p=self.background_probs)
            
            # One-hot encode
            seq_tensor = torch.zeros((4, length), dtype=torch.float32)
            seq_tensor[indices, torch.arange(length)] = 1.0
            
        # Apply motif inserters
        for inserter in self.motif_inserters:
            seq_tensor = inserter(seq_tensor, interval)
            
        return seq_tensor

class MockSignalExtractor(BaseSignalExtractor):
    """
    Generates signal based on the sequence provided by a SequenceExtractor.
    """
    def __init__(
        self,
        sequence_extractor: BaseSequenceExtractor,
        signal_generator: SignalGenerator | None = None
    ):
        self.sequence_extractor = sequence_extractor
        self.signal_generator = signal_generator or self._default_generator

    def _default_generator(self, sequence: torch.Tensor) -> torch.Tensor:
        # Default behavior: zeros
        return torch.zeros((1, sequence.shape[1]), dtype=torch.float32)

    def extract(self, interval: Interval) -> torch.Tensor:
        # Get sequence
        # Note: We assume sequence_extractor is deterministic!
        seq_tensor = self.sequence_extractor.extract(interval)
        
        # Generate signal
        signal = self.signal_generator(seq_tensor)
        
        return signal

# --- Implementations for the specific task ---

def insert_ggaa_motifs(
    sequence: torch.Tensor,
    interval: Interval,
    prob: float = 0.5,
    max_motifs: int = 5
) -> torch.Tensor:
    """
    Inserts 'GGAA' motifs at random positions.
    """
    length = sequence.shape[1]
    seed = hash((interval.chrom, interval.start, interval.end, "ggaa")) % (2**32)
    rng = np.random.RandomState(seed)
    
    # Decide if we insert motifs
    if rng.rand() > prob:
        return sequence
    
    num_motifs = rng.randint(1, max_motifs + 1)
    
    # GGAA in one-hot (ACGT encoding: A=0, C=1, G=2, T=3)
    # G=2, A=0
    # Shape (4, 4)
    motif_len = 4
    # GGAA
    # 0: . . . A
    # 1: . . . .
    # 2: G G . .
    # 3: . . . .
    
    # We construct the indices manually to match "ACGT"
    # A=0, C=1, G=2, T=3
    # G G A A -> [2, 2, 0, 0]
    motif_indices = [2, 2, 0, 0]
    
    # Clone sequence to avoid inplace modification of shared memory if any (though here we likely created it fresh)
    sequence = sequence.clone()
    
    for _ in range(num_motifs):
        # Pick random position
        if length < motif_len:
            continue
        pos = rng.randint(0, length - motif_len + 1)
        
        # Zero out the position
        sequence[:, pos : pos + motif_len] = 0
        
        # Set bits
        for i, base_idx in enumerate(motif_indices):
            sequence[base_idx, pos + i] = 1.0
            
    return sequence

class GaussianSignalGenerator:
    """
    Generates Gaussian peaks centered on 'GGAA' motifs.
    """
    def __init__(self, sigma: float = 10.0, base_height: float = 10.0):
        self.sigma = sigma
        self.base_height = base_height

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (4, L)
        # Find GGAA motifs
        # ACGT encoding: G=2, A=0. GGAA -> [2, 2, 0, 0]
        
        # We can use convolution to find motifs? Or just simple iteration for correctness.
        # Convolution with matching kernel is fast.
        
        # Kernel for GGAA
        # 4 channels, width 4
        # We want to match G at pos 0, G at pos 1, A at pos 2, A at pos 3
        # Weights: 1.0 at correct positions.
        # Threshold: 4.0 (perfect match)
        
        weights = torch.zeros((1, 4, 4))
        # (OutChan, InChan, Width)
        # Channel 2 (G) at pos 0, 1
        weights[0, 2, 0] = 1.0
        weights[0, 2, 1] = 1.0
        # Channel 0 (A) at pos 2, 3
        weights[0, 0, 2] = 1.0
        weights[0, 0, 3] = 1.0
        
        # Convolve
        # sequence is (4, L). Need (Batch, 4, L) -> (1, 4, L)
        inp = sequence.unsqueeze(0)
        
        # Conv1d
        match_score = torch.nn.functional.conv1d(inp, weights)
        # match_score shape: (1, 1, L - 3)
        
        # Find exact matches (value approx 4.0)
        is_match = match_score[0, 0] > 3.9
        
        # Get indices
        match_indices = torch.nonzero(is_match).flatten()
        
        # Output signal
        L = sequence.shape[1]
        signal = torch.zeros(L, dtype=torch.float32)
        
        if len(match_indices) == 0:
            return signal.unsqueeze(0) # (1, L)
            
        # Create coordinate grid
        x = torch.arange(L, dtype=torch.float32)
        
        # Add gaussians
        # We handle clusters by summing them.
        # "height proportional to number of consecutive GGAAs"
        # If we just sum Gaussians, two close peaks will form a higher peak.
        # This naturally satisfies "consecutive/clustered -> higher/wider signal".
        
        for idx in match_indices:
            # Center of GGAA (length 4) is idx + 1.5
            center = idx.float() + 1.5
            
            gauss = self.base_height * torch.exp(-0.5 * ((x - center) / self.sigma) ** 2)
            signal += gauss
            
        return signal.unsqueeze(0) # (1, L)
