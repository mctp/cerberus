import random
from pathlib import Path
import pytorch_lightning as pl
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from .cache import get_default_cache_dir, resolve_cache_dir, load_prepare_cache, save_prepare_cache
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
        seed: int = 42,
        drop_last: bool = False,
        cache_dir: Path | str | None = None,
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
            seed: Random seed for sampler initialization and prepare_data
                caching. Must be the same across all DDP ranks.
            drop_last: Whether to drop the last incomplete batch in training.
            cache_dir: Base directory for prepare_data() cache. Defaults to
                $XDG_CACHE_HOME/cerberus or ~/.cache/cerberus.
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
        self.cache_dir = Path(cache_dir) if cache_dir is not None else get_default_cache_dir()

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

    def _resolve_cache_dir(self) -> Path | None:
        """
        Computes the deterministic cache subdirectory for the current config.

        Returns None if the sampler type does not benefit from caching.
        """
        sampler_type = self.sampler_config["sampler_type"]
        # NOTE: Every sampler type that uses ComplexityMatchedSampler must be
        # listed here, otherwise its metrics won't be cached to disk and will
        # be recomputed from scratch on every run.
        if sampler_type not in ("peak", "complexity_matched", "negative_peak"):
            return None
        return resolve_cache_dir(
            self.cache_dir,
            fasta_path=self.genome_config["fasta_path"],
            sampler_config=self.sampler_config,
            seed=self.seed,
            chrom_sizes=self.genome_config["chrom_sizes"],
        )

    def prepare_data(self) -> None:
        """
        Pre-computes complexity metrics and caches them to disk.

        Called by Lightning on rank 0 only, before setup() runs on all ranks.
        Creates a temporary sampler to trigger metric computation, then
        serializes the resulting metrics_cache to disk so that all ranks can
        load it in setup() without redundant FASTA reads.

        No-op if the sampler type doesn't need caching or if a valid cache
        already exists.
        """
        cache_dir = self._resolve_cache_dir()
        if cache_dir is None:
            return
        if (cache_dir / "ready").exists():
            logger.info(f"prepare_data cache already exists at {cache_dir}")
            return

        logger.info("prepare_data: computing complexity metrics for caching...")

        # Create a temporary dataset to trigger sampler initialization and
        # metric computation. This opens FASTA/BigWig files on rank 0 only.
        tmp_dataset = CerberusDataset(
            genome_config=self.genome_config,
            data_config=self.data_config,
            sampler_config=self.sampler_config,
            seed=self.seed,
        )

        # Extract metrics_cache from the sampler
        sampler = tmp_dataset.sampler
        sampler_type = self.sampler_config["sampler_type"]
        # NOTE: Keep in sync with _resolve_cache_dir — every sampler type
        # listed there must have a branch here to extract its metrics_cache.
        if sampler_type == "complexity_matched":
            metrics_cache = sampler.metrics_cache  # type: ignore[union-attr]
        elif sampler_type == "peak":
            if sampler.negatives is not None:  # type: ignore[union-attr]
                metrics_cache = sampler.negatives.metrics_cache  # type: ignore[union-attr]
            else:
                logger.info("prepare_data: peak sampler has no negatives, nothing to cache")
                return
        elif sampler_type == "negative_peak":
            metrics_cache = sampler.negatives.metrics_cache  # type: ignore[union-attr]
        else:
            return

        save_prepare_cache(cache_dir, metrics_cache)
        logger.info(f"prepare_data: cached {len(metrics_cache)} entries to {cache_dir}")

    def _load_prepare_cache(self) -> dict[str, np.ndarray] | None:
        """
        Loads the prepare_data cache from disk if available.

        Returns None if the sampler type doesn't need caching or no cache exists.
        """
        cache_dir = self._resolve_cache_dir()
        if cache_dir is None:
            return None
        return load_prepare_cache(cache_dir)

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

        # Load cached complexity metrics if available (populated by prepare_data)
        prepare_cache = self._load_prepare_cache()

        # Initialize full dataset
        full_dataset = CerberusDataset(
            genome_config=self.genome_config,
            data_config=self.data_config,
            sampler_config=self.sampler_config,
            in_memory=self.in_memory,
            seed=self.seed,
            prepare_cache=prepare_cache,
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
                seed = self.seed + (epoch * world_size) + rank
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
