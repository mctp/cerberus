from typing import Any, cast
import torch
from torch.utils.data import Dataset
import numpy as np
from interlap import InterLap
from .config import (
    DataConfig,
    SamplerConfig,
    GenomeConfig,
    validate_data_config,
    validate_genome_config,
    validate_sampler_config,
    validate_data_and_sampler_compatibility,
)
from .exclude import get_exclude_intervals
from .samplers import Sampler, create_sampler
from .genome import GenomeConfig, create_genome_folds
from .sequence import SequenceExtractor, InMemorySequenceExtractor, BaseSequenceExtractor
from .signal import SignalExtractor, InMemorySignalExtractor, BaseSignalExtractor
from .transform import DataTransform, Compose, create_default_transforms


class CerberusDataset(Dataset):
    """
    A PyTorch Dataset for loading genomic data using the Cerberus framework.
    
    This dataset integrates genome configuration, data samplers, and feature extractors
    to provide a unified interface for training deep learning models on genomic data.
    
    It supports:
    - On-the-fly or in-memory data loading.
    - Custom sampling strategies (intervals, sliding windows, multi-sampler).
    - Data transformations (jitter, reverse complement, etc.).
    - K-fold cross-validation splitting.
    """
    genome_config: GenomeConfig
    data_config: DataConfig
    sampler_config: SamplerConfig
    folds: list[dict[str, InterLap]]
    exclude_intervals: dict[str, InterLap]
    sampler: Sampler
    sequence_extractor: BaseSequenceExtractor | None
    input_signal_extractor: BaseSignalExtractor | None
    target_signal_extractor: BaseSignalExtractor | None
    transforms: Compose
    in_memory: bool

    def __init__(
        self,
        genome_config: GenomeConfig,
        data_config: DataConfig,
        sampler_config: SamplerConfig,
        sequence_extractor: BaseSequenceExtractor | None = None,
        input_signal_extractor: BaseSignalExtractor | None = None,
        target_signal_extractor: BaseSignalExtractor | None = None,
        sampler: Sampler | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
        transforms: list[DataTransform] | None = None,
        in_memory: bool = False,
    ):
        """
        Initializes the CerberusDataset.

        Args:
            genome_config: Genome configuration dictionary or object.
            data_config: Data configuration dictionary or object.
            sampler_config: Sampler configuration dictionary or object.
            sequence_extractor: Optional custom sequence extractor. If None, created based on config.
            input_signal_extractor: Optional custom input signal extractor. If None, created based on config.
            target_signal_extractor: Optional custom target signal extractor. If None, created based on config.
            sampler: Optional custom sampler. If None, created based on config.
            exclude_intervals: Optional pre-computed exclude intervals. If None, loaded from config.
            transforms: Optional list of transforms. If None, defaults are created from data_config.
            in_memory: Whether to load data into memory (default: False).
        
        Raises:
            ValueError: If configurations are invalid.
        """

        self.genome_config = validate_genome_config(genome_config)
        self.sampler_config = validate_sampler_config(sampler_config)
        self.data_config = validate_data_config(data_config)
        self.in_memory = in_memory

        validate_data_and_sampler_compatibility(self.data_config, self.sampler_config)

        # Initialize Folds
        self.folds = create_genome_folds(
            self.genome_config["chrom_sizes"],
            fold_type=self.genome_config["fold_type"],
            fold_args=self.genome_config["fold_args"],
        )

        # Get exclude intervals
        if exclude_intervals is not None:
            self.exclude_intervals = exclude_intervals
        else:
            self.exclude_intervals = self._get_exclude_intervals()

        # Initialize Sampler
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = self._initialize_sampler()

        # Initialize Sequence Extractor
        if sequence_extractor is not None:
            self.sequence_extractor = sequence_extractor
        else:
            if self.data_config["use_sequence"]:
                if self.in_memory:
                    self.sequence_extractor = InMemorySequenceExtractor(
                        fasta_path=self.genome_config["fasta_path"],
                        encoding=self.data_config["encoding"],
                    )
                else:
                    self.sequence_extractor = SequenceExtractor(
                        fasta_path=self.genome_config["fasta_path"],
                        encoding=self.data_config["encoding"],
                    )
            else:
                self.sequence_extractor = None

        # Initialize Input Signal Extractor
        if input_signal_extractor is not None:
            self.input_signal_extractor = input_signal_extractor
        elif self.data_config["inputs"]:
            if self.in_memory:
                self.input_signal_extractor = InMemorySignalExtractor(
                    bigwig_paths=self.data_config["inputs"]
                )
            else:
                self.input_signal_extractor = SignalExtractor(
                    bigwig_paths=self.data_config["inputs"]
                )
        else:
            self.input_signal_extractor = None

        # Initialize Target Signal Extractor
        if target_signal_extractor is not None:
            self.target_signal_extractor = target_signal_extractor
        elif self.data_config["targets"]:
            if self.in_memory:
                self.target_signal_extractor = InMemorySignalExtractor(
                    bigwig_paths=self.data_config["targets"]
                )
            else:
                self.target_signal_extractor = SignalExtractor(
                    bigwig_paths=self.data_config["targets"]
                )
        else:
            self.target_signal_extractor = None
            
        # Ensure at least one input source is available
        if self.sequence_extractor is None and self.input_signal_extractor is None:
            raise ValueError(
                "No input sources provided. Either enable sequence input (use_sequence=True) "
                "or provide input signals."
            )

        # Initialize Transforms
        if transforms:
            self.transforms = Compose(transforms)
        else:
            # Auto-create from DataConfig
            defaults = create_default_transforms(self.data_config)
            self.transforms = Compose(defaults)

    def _get_exclude_intervals(self) -> dict[str, InterLap]:
        """Loads excluded intervals from files specified in config."""
        return get_exclude_intervals(
            self.genome_config["exclude_intervals"],
        )

    def _initialize_sampler(self) -> Sampler:
        """Creates the data sampler based on configuration."""
        return create_sampler(
            self.sampler_config,
            self.genome_config["chrom_sizes"],
            self.exclude_intervals,
            folds=self.folds,
        )

    def __len__(self) -> int:
        """Returns the total number of samples available."""
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Retrieves a sample at the given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'inputs': Tensor of shape (Channels, Length). Concatenation of sequence (4 channels)
                            and optional input signals.
                - 'targets': Tensor of shape (Target_Channels, Output_Length).
                - 'intervals': String representation of the genomic interval.
        """
        interval = self.sampler[idx]

        # Extract inputs
        input_tensors = []
        
        # Extract sequence
        if self.sequence_extractor is not None:
            input_tensors.append(self.sequence_extractor.extract(interval))  # (4, L)

        # Extract input signals
        if self.input_signal_extractor is not None:
            input_tensors.append(self.input_signal_extractor.extract(interval))  # (C, L)
            
        # Stack along channel dimension
        inputs = torch.cat(input_tensors, dim=0)

        # Extract targets
        if self.target_signal_extractor:
            targets = self.target_signal_extractor.extract(interval)
        else:
            targets = torch.empty(0)

        # Apply transforms
        if self.transforms:
            inputs, targets, interval = self.transforms(inputs, targets, interval)

        return {
            "inputs": inputs,
            "targets": targets,
            "intervals": str(interval),
        }

    def __getitems__(self, indices: list[int]):
        """Batch retrieval optimization (optional)."""
        # Batch optimization if needed, for now loop
        return [self.__getitem__(idx) for idx in indices]

    def _subset(self, sampler: Sampler) -> "CerberusDataset":
        """
        Internal method to create a new dataset instance using the provided sampler.
        Shares the sequence extractor, configuration, and exclude intervals with the current instance.
        """
        return CerberusDataset(
            genome_config=self.genome_config,
            sampler_config=self.sampler_config,
            data_config=self.data_config,
            sequence_extractor=self.sequence_extractor,
            input_signal_extractor=self.input_signal_extractor,
            target_signal_extractor=self.target_signal_extractor,
            sampler=sampler,
            exclude_intervals=self.exclude_intervals,
            transforms=self.transforms.transforms if self.transforms else None,
            in_memory=self.in_memory,
        )

    def split_folds(
        self, test_fold: int | None = None, val_fold: int | None = None
    ) -> tuple["CerberusDataset", "CerberusDataset", "CerberusDataset"]:
        """
        Split the dataset into train, validation, and test sets using k-fold cross-validation.
        
        This method delegates to the underlying sampler's split_folds method, creating new
        dataset instances for each split that share the underlying data extractors.

        Args:
            test_fold: Index of the fold to use for testing.
            val_fold: Index of the fold to use for validation.

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        train_sampler, val_sampler, test_sampler = self.sampler.split_folds(
            test_fold=test_fold, val_fold=val_fold
        )

        return (
            self._subset(train_sampler),
            self._subset(val_sampler),
            self._subset(test_sampler),
        )

    def resample(self, seed: int | None = None):
        """
        Trigger resampling if the underlying sampler supports it.
        
        This is useful for MultiSampler to re-draw indices based on scaling factors
        at the beginning of each epoch.

        Args:
            seed: Random seed for reproducibility.
        """
        self.sampler.resample(seed=seed)
