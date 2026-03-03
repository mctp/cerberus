import random
import pytorch_lightning as pl
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from .dataset import CerberusDataset
from .signal import UniversalExtractor
from .config import (
    GenomeConfig,
    DataConfig,
    SamplerConfig,
    validate_genome_config,
    validate_data_config,
    validate_sampler_config,
    validate_data_and_sampler_compatibility,
)

logger = logging.getLogger(__name__)


class CerberusDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Cerberus.
    
    Manages data loading, splitting, and setup for training, validation, and testing.
    Handles efficient data loading with multi-processing and shared memory.
    """
    def __init__(
        self,
        genome_config: GenomeConfig,
        data_config: DataConfig,
        sampler_config: SamplerConfig,
        test_fold: int | None = None,
        val_fold: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str | None = None,
        seed: int | None = None,
        drop_last: bool = False,
    ):
        """
        Args:
            genome_config: Genome configuration dictionary.
            data_config: Data configuration dictionary.
            sampler_config: Sampler configuration dictionary.
            test_fold: Fold index to use for testing.
            val_fold: Fold index to use for validation.
            pin_memory: Whether to pin memory in DataLoaders (recommended for GPU training).
            persistent_workers: Whether to use persistent workers in DataLoaders.
            multiprocessing_context: Context name for multiprocessing (e.g., 'spawn', 'fork').
            seed: Optional random seed for sampler initialization.
            drop_last: Whether to drop the last incomplete batch in training.
        """
        super().__init__()
        self.genome_config = validate_genome_config(genome_config)
        self.data_config = validate_data_config(data_config)
        self.sampler_config = validate_sampler_config(sampler_config)
        validate_data_and_sampler_compatibility(self.data_config, self.sampler_config)
        
        # Runtime settings (configured via setup)
        self.batch_size = 1
        self.val_batch_size = 1
        self.num_workers = 0
        self.in_memory = False

        # Resolve fold indices: argument > config
        if test_fold is None:
            test_fold = self.genome_config["fold_args"]["test_fold"]
        if val_fold is None:
            val_fold = self.genome_config["fold_args"]["val_fold"]
        
        self.test_fold = test_fold
        self.val_fold = val_fold

        # Disable pin_memory on MPS devices as it is currently not supported
        if pin_memory and torch.backends.mps.is_available():
            pin_memory = False
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        self.seed = seed
        self.drop_last = drop_last

        self.train_dataset: CerberusDataset | None = None
        self.val_dataset: CerberusDataset | None = None
        self.test_dataset: CerberusDataset | None = None
        self._is_initialized = False

    @staticmethod
    def _worker_init_fn(worker_id):
        """
        Internal worker initialization function for DataLoader.
        
        CRITICAL for reproducibility and data diversity:
        1. Ensures each worker has a different random seed derived from the torch seed.
        2. Without this, if workers use numpy.random (e.g. in Samplers or custom transforms),
           they would all share the same initial state and produce identical augmentations/samples.
        3. Ties numpy random state to torch's random state management.
        """
        # Seed numpy/random based on torch seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def setup(
        self, 
        stage: str | None = None, 
        batch_size: int | None = None, 
        val_batch_size: int | None = None,
        num_workers: int | None = None,
        in_memory: bool | None = None,
    ):
        """
        Sets up the datasets for the specified stage.
        
        Initializes the full dataset and splits it into train/val/test sets based on the configured folds.
        Allows updating runtime parameters (batch_size, num_workers).

        Args:
            stage: Stage name (e.g., 'fit', 'test'). Optional.
            batch_size: Batch size override.
            val_batch_size: Validation/Test batch size override.
            num_workers: Number of workers override.
            in_memory: Whether to load data into memory.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        
        if val_batch_size is not None:
            self.val_batch_size = val_batch_size
        elif batch_size is not None:
            self.val_batch_size = batch_size

        if num_workers is not None:
            self.num_workers = num_workers
        if in_memory is not None:
            self.in_memory = in_memory

        if self._is_initialized:
            return

        logger.info(f"Setting up DataModule (test_fold={self.test_fold}, val_fold={self.val_fold})...")

        # Initialize full dataset
        full_dataset = CerberusDataset(
            genome_config=self.genome_config,
            data_config=self.data_config,
            sampler_config=self.sampler_config,
            in_memory=self.in_memory,
            seed=self.seed,
        )

        # Split into folds
        self.train_dataset, self.val_dataset, self.test_dataset = full_dataset.split_folds(
            test_fold=self.test_fold, val_fold=self.val_fold
        )
        logger.info(f"DataModule setup complete. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        self._is_initialized = True

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        if not self._is_initialized:
            raise RuntimeError("DataModule not setup")
        assert self.train_dataset is not None

        # When reloading dataloaders every epoch, we perform resampling here.
        # This requires Trainer(reload_dataloaders_every_n_epochs=1).
        if self.trainer:
            try:
                rank = self.trainer.global_rank
                epoch = self.trainer.current_epoch
                # Ensure unique seed per rank to maximize data coverage in DDP
                world_size = self.trainer.world_size if self.trainer else 1
                base_seed = self.seed if self.seed is not None else 0
                seed = base_seed + (epoch * world_size) + rank
                self.train_dataset.resample(seed=seed)
            except (AttributeError, RuntimeError) as exc:
                logger.warning("Could not resample train dataset: %s", exc)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            multiprocessing_context=self.multiprocessing_context,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        if not self._is_initialized:
            raise RuntimeError("DataModule not setup")
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            multiprocessing_context=self.multiprocessing_context,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test DataLoader."""
        if not self._is_initialized:
            raise RuntimeError("DataModule not setup")
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            multiprocessing_context=self.multiprocessing_context,
        )

    def compute_median_counts(self, n_samples: int = 2000) -> float:
        """
        Compute the median total signal count per peak from the training fold.

        Samples up to n_samples intervals from the training dataset, sums raw target
        signal over all channels and positions to get total counts per peak, and
        returns the median multiplied by target_scale. The result represents the
        effective count magnitude that the loss function sees during training.

        Used to compute a data-adaptive count loss weight (alpha) for BPNet-style
        models via compute_counts_loss_weight(). See docs/internal/adaptive_counts_loss_weight.md.

        Args:
            n_samples: Number of intervals to sample. If the training dataset has
                fewer intervals, all are used.

        Returns:
            Median total raw counts per peak scaled by data_config["target_scale"].

        Raises:
            RuntimeError: If setup() has not been called yet.
        """
        if not self._is_initialized or self.train_dataset is None:
            raise RuntimeError(
                "DataModule must be setup before computing statistics. "
                "Call datamodule.setup() first."
            )
        dataset = self.train_dataset
        if dataset.sampler is None:
            raise RuntimeError("train_dataset has no sampler; cannot compute median counts.")
        n = len(dataset)
        indices = random.sample(range(n), min(n_samples, n))

        # Use a short-lived temporary extractor so the dataset's own handles are
        # never opened in the main process.  On Linux, DataLoader workers are
        # forked and inherit open file descriptors; workers sharing the same
        # kernel open-file description interleave seeks and corrupt bigtools
        # B-tree reads (BadData / Unexpected isleaf panics).  The tmp_extractor
        # is garbage-collected when this method returns, leaving the dataset
        # extractor at _bigwig_files=None so each worker opens its own fd.
        tmp_extractor = UniversalExtractor(
            paths=dataset.data_config["targets"],
            in_memory=False,
        )
        output_len = dataset.data_config["output_len"]
        input_len = dataset.data_config["input_len"]
        crop_start = (input_len - output_len) // 2
        crop_end = crop_start + output_len

        counts = []
        for i in indices:
            interval = dataset.sampler[i]
            raw = tmp_extractor.extract(interval)   # (C, input_len), no transforms
            if raw.shape[-1] > output_len:
                raw = raw[..., crop_start:crop_end]
            counts.append(float(raw.sum()))
        raw_median = float(np.median(counts))
        target_scale = self.data_config["target_scale"]
        scaled_median = raw_median * target_scale
        logger.info(
            f"Computed median_counts={scaled_median:.1f} "
            f"(raw={raw_median:.1f} × target_scale={target_scale}) "
            f"from {len(indices)} training intervals."
        )
        return scaled_median
