import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional
from .dataset import CerberusDataset
from .config import GenomeConfig, DataConfig, SamplerConfig

class CerberusDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Cerberus.
    
    Manages data loading, splitting, and setup for training, validation, and testing.
    Handles efficient data loading with multi-processing and shared memory.
    """
    def __init__(
        self,
        genome_config: dict | GenomeConfig,
        data_config: dict | DataConfig,
        sampler_config: dict | SamplerConfig,
        batch_size: int = 32,
        num_workers: int = 4,
        test_fold: int = 0,
        val_fold: int = 1,
        pin_memory: bool = True,
    ):
        """
        Args:
            genome_config: Genome configuration dictionary.
            data_config: Data configuration dictionary.
            sampler_config: Sampler configuration dictionary.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes for data loading.
            test_fold: Fold index to use for testing.
            val_fold: Fold index to use for validation.
            pin_memory: Whether to pin memory in DataLoaders (recommended for GPU training).
        """
        super().__init__()
        self.genome_config = genome_config
        self.data_config = data_config
        self.sampler_config = sampler_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.pin_memory = pin_memory

        self.train_dataset: Optional[CerberusDataset] = None
        self.val_dataset: Optional[CerberusDataset] = None
        self.test_dataset: Optional[CerberusDataset] = None
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

    def setup(self, stage: Optional[str] = None):
        """
        Sets up the datasets for the specified stage.
        
        Initializes the full dataset and splits it into train/val/test sets based on the configured folds.
        
        Args:
            stage: Stage name (e.g., 'fit', 'test'). Optional.
        """
        if self._is_initialized:
            return

        # Initialize full dataset
        full_dataset = CerberusDataset(
            genome_config=self.genome_config,
            data_config=self.data_config,
            sampler_config=self.sampler_config,
        )

        # Split into folds
        self.train_dataset, self.val_dataset, self.test_dataset = full_dataset.split_folds(
            test_fold=self.test_fold, val_fold=self.val_fold
        )
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
                seed = int(epoch + rank)
                self.train_dataset.resample(seed=seed)
            except (AttributeError, RuntimeError):
                # Trainer not fully initialized or in testing
                pass

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        if not self._is_initialized:
            raise RuntimeError("DataModule not setup")
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the test DataLoader."""
        if not self._is_initialized:
            raise RuntimeError("DataModule not setup")
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
        )
