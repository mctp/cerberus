from typing import Any
import numpy as np
import torch
import logging
from torch.utils.data import Dataset
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
from .signal import BaseSignalExtractor, UniversalExtractor
from .transform import DataTransform, Compose, create_default_transforms
from .interval import Interval, resolve_interval

logger = logging.getLogger(__name__)


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
    sampler_config: SamplerConfig | None
    folds: list[dict[str, InterLap]]
    exclude_intervals: dict[str, InterLap]
    sampler: Sampler | None
    sequence_extractor: BaseSequenceExtractor | None
    input_signal_extractor: BaseSignalExtractor | None
    target_signal_extractor: BaseSignalExtractor | None
    transforms: Compose
    deterministic_transforms: Compose
    in_memory: bool
    is_train: bool

    def __init__(
        self,
        genome_config: GenomeConfig,
        data_config: DataConfig,
        sampler_config: SamplerConfig | None = None,
        sequence_extractor: BaseSequenceExtractor | None = None,
        input_signal_extractor: BaseSignalExtractor | None = None,
        target_signal_extractor: BaseSignalExtractor | None = None,
        sampler: Sampler | None = None,
        exclude_intervals: dict[str, InterLap] | None = None,
        transforms: list[DataTransform] | None = None,
        deterministic_transforms: list[DataTransform] | None = None,
        in_memory: bool = False,
        is_train: bool = True,
        seed: int = 42,
        prepare_cache: dict[str, np.ndarray] | None = None,
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
            transforms: Optional list of transforms for training. If None, defaults are created from data_config.
            deterministic_transforms: Optional list of deterministic transforms for val/test.
                                      Must be provided if transforms is provided.
            in_memory: Whether to load data into memory (default: False).
            is_train: Whether this dataset is used for training (enables random transforms).
            seed: Random seed for sampler initialization.
            prepare_cache: Pre-computed data from prepare_data() (e.g. complexity metrics cache).

        Raises:
            ValueError: If configurations are invalid.
        """

        self.genome_config = validate_genome_config(genome_config)
        self.data_config = validate_data_config(data_config)
        self.in_memory = in_memory
        self.is_train = is_train
        self.seed = seed
        self.prepare_cache = prepare_cache

        if sampler_config is not None:
            self.sampler_config = validate_sampler_config(sampler_config)
            validate_data_and_sampler_compatibility(self.data_config, self.sampler_config)
        else:
            self.sampler_config = None

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
        elif self.sampler_config is not None:
            logger.debug(f"Initializing sampler of type {self.sampler_config['sampler_type']}...")
            self.sampler = self._initialize_sampler()
        else:
            self.sampler = None

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
            self.input_signal_extractor = UniversalExtractor(
                paths=self.data_config["inputs"],
                in_memory=self.in_memory
            )
        else:
            self.input_signal_extractor = None

        # Initialize Target Signal Extractor
        if target_signal_extractor is not None:
            self.target_signal_extractor = target_signal_extractor
        elif self.data_config["targets"]:
            self.target_signal_extractor = UniversalExtractor(
                paths=self.data_config["targets"],
                in_memory=self.in_memory
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
        if (transforms is None) != (deterministic_transforms is None):
            raise ValueError(
                "Both 'transforms' and 'deterministic_transforms' must be provided, or neither (to use defaults)."
            )

        if transforms is not None:
            assert deterministic_transforms is not None  # guaranteed by check above
            self.transforms = Compose(transforms)
            self.deterministic_transforms = Compose(deterministic_transforms)
        else:
            # Auto-create from DataConfig
            self.transforms = Compose(
                create_default_transforms(self.data_config, deterministic=False)
            )
            self.deterministic_transforms = Compose(
                create_default_transforms(self.data_config, deterministic=True)
            )

    def _get_exclude_intervals(self) -> dict[str, InterLap]:
        """Loads excluded intervals from files specified in config."""
        return get_exclude_intervals(
            self.genome_config["exclude_intervals"],
        )

    def _initialize_sampler(self) -> Sampler:
        """Creates the data sampler based on configuration."""
        if self.sampler_config is None:
            raise ValueError("Cannot initialize sampler without sampler_config")

        return create_sampler(
            self.sampler_config,
            self.genome_config["chrom_sizes"],
            folds=self.folds,
            exclude_intervals=self.exclude_intervals,
            fasta_path=self.genome_config["fasta_path"],
            seed=self.seed,
            prepare_cache=self.prepare_cache,
        )

    def __len__(self) -> int:
        """Returns the total number of samples available."""
        if self.sampler is None:
            return 0
        return len(self.sampler)

    def get_raw_targets(
        self,
        query: str | tuple[str, int, int] | list[object] | Interval,
        crop_to_output_len: bool = True
    ) -> torch.Tensor:
        """
        Retrieves raw target signals for a specific interval, bypassing transforms.
        
        This is useful for getting true observed counts without binning or log transforms
        that might be applied in the standard pipeline.
        
        Args:
            query: Interval object, string "chr:start-end", or tuple (chr, start, end).
            crop_to_output_len: Whether to crop the extracted signal to the output length.
            
        Returns:
            torch.Tensor: The raw target signal.
        """
        interval = resolve_interval(query)
        
        if self.target_signal_extractor is None:
            raise RuntimeError("Target signal extractor is not initialized.")

        # Extract (C, Input_Len)
        raw_target = self.target_signal_extractor.extract(interval) 
        
        if crop_to_output_len:
            crop_start = (self.data_config["input_len"] - self.data_config["output_len"]) // 2
            crop_end = crop_start + self.data_config["output_len"]
            
            # Check if we need to crop
            if raw_target.shape[-1] > self.data_config["output_len"]:
                raw_target = raw_target[..., crop_start:crop_end]
                
        return raw_target

    def get_interval(self, query: str | tuple[str, int, int] | list[object] | Interval) -> dict[str, Any]:
        """
        Retrieves inputs and targets for a specific interval, side-stepping the sampler.
        Resolves query into an Interval object first.

        Args:
            query: Interval object, string "chr:start-end", or tuple (chr, start, end).

        Returns:
            dict: Same structure as __getitem__.
        """
        interval = resolve_interval(query)
        return self._get_interval(interval)

    def _get_interval(self, interval: Interval) -> dict[str, Any]:
        """
        Internal method to extract data for a resolved interval.
        
        This method performs the core logic of fetching data from extractors
        and applying transformations.
        
        Note: The 'interval' object is passed to transforms, which may modify it in-place
        (e.g., Jitter updates the start/end coordinates). The returned dictionary contains
        the string representation of this potentially modified interval.
        
        Args:
            interval: The genomic interval to retrieve.
            
        Returns:
            dict: Containing 'inputs', 'targets', and 'intervals' string.
        """
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

        # Apply transforms (self.transforms and self.deterministic_transforms
        # are always Compose objects, possibly wrapping an empty list)
        if self.is_train:
            inputs, targets, interval = self.transforms(
                inputs, targets, interval
            )
        else:
            inputs, targets, interval = self.deterministic_transforms(
                inputs, targets, interval
            )

        return {
            "inputs": inputs,
            "targets": targets,
            "intervals": str(interval),
        }

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
                               Note: If random transforms (like Jitter) are active, this string
                               reflects the transformed coordinates, not the original sampler coordinates.
                - 'interval_source': str — class name of the sub-sampler that produced this
                                     interval (e.g. ``"IntervalSampler"``, ``"ComplexityMatchedSampler"``).
                                     Requires the sampler to implement ``get_interval_source()``
                                     (e.g. :class:`MultiSampler` / :class:`PeakSampler`); defaults to
                                     ``"unknown"`` for samplers that do not support source labelling.
        
        Raises:
            TypeError: If no sampler is configured for this dataset.
        """
        if self.sampler is None:
            raise TypeError("Dataset has no sampler configured. Use get_interval() for specific queries.")
        interval = self.sampler[idx]
        result = self._get_interval(interval)
        result["interval_source"] = self.sampler.get_interval_source(idx)
        return result

    def __getitems__(self, indices: list[int]) -> list[dict[str, object]]:
        """Batch retrieval optimization (optional)."""
        # Batch optimization if needed, for now loop
        return [self.__getitem__(idx) for idx in indices]

    def _subset(self, sampler: Sampler | None, is_train: bool = True) -> "CerberusDataset":
        """
        Internal method to create a new dataset instance using the provided sampler.
        
        This is used by split_folds to create train/val/test datasets that share
        underlying resources (extractors, caches) with the parent dataset but
        iterate over different subsets of intervals.
        
        Args:
            sampler: The subset sampler to use.
            is_train: Whether the subset is for training.
            
        Returns:
            CerberusDataset: A new dataset instance sharing resources.
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
            transforms=self.transforms.transforms,
            deterministic_transforms=self.deterministic_transforms.transforms,
            in_memory=self.in_memory,
            is_train=is_train,
            seed=self.seed,
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
        if self.sampler is None:
            return (
                self._subset(None, is_train=True),
                self._subset(None, is_train=False),
                self._subset(None, is_train=False),
            )

        train_sampler, val_sampler, test_sampler = self.sampler.split_folds(
            test_fold=test_fold, val_fold=val_fold
        )

        return (
            self._subset(train_sampler, is_train=True),
            self._subset(val_sampler, is_train=False),
            self._subset(test_sampler, is_train=False),
        )

    def resample(self, seed: int | None = None) -> None:
        """
        Trigger resampling if the underlying sampler supports it.
        
        This is useful for MultiSampler to re-draw indices based on scaling factors
        at the beginning of each epoch.

        Args:
            seed: Random seed for reproducibility.
        """
        if self.sampler is not None:
            self.sampler.resample(seed=seed)
