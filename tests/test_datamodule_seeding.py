
import unittest
from unittest.mock import MagicMock
from typing import cast
from cerberus.datamodule import CerberusDataModule

class TestDataModuleSeeding(unittest.TestCase):
    def setUp(self):
        # minimal valid configs
        self.genome_config = {
            "name": "hg38",
            "fasta_path": "pyproject.toml",
            "allowed_chroms": ["chr1"],
            "chrom_sizes": {"chr1": 1000},
            "exclude_intervals": {},
            "fold_type": "chrom_partition",
            "fold_args": {"k": 5, "test_fold": 0, "val_fold": 1}
        } # type: ignore
        self.data_config = {
            "inputs": {},
            "targets": {},
            "input_len": 100,
            "output_len": 100,
            "output_bin_size": 1,
            "encoding": "ACGT",
            "max_jitter": 0,
            "log_transform": False,
            "reverse_complement": False,
        "target_scale": 1.0,
            "count_pseudocount": 1.0,
            "use_sequence": True
        } # type: ignore
        self.sampler_config = {
            "sampler_type": "random",
            "padded_size": 100,
            "sampler_args": {"num_intervals": 10}
        } # type: ignore
        
        # Create DataModule
        self.dm = CerberusDataModule(
            genome_config=self.genome_config, # type: ignore
            data_config=self.data_config, # type: ignore
            sampler_config=self.sampler_config, # type: ignore
            seed=123 # Configured seed
        )
        
        # Mock trainer
        self.trainer_mock = MagicMock()
        self.trainer_mock.global_rank = 0
        self.trainer_mock.current_epoch = 0
        self.trainer_mock.world_size = 1
        self.dm.trainer = self.trainer_mock # type: ignore
        
        # Mock dataset creation to avoid loading actual files
        self.train_ds_mock = MagicMock()
        self.train_ds_mock.__len__.return_value = 100
        self.dm.train_dataset = self.train_ds_mock # type: ignore
        
        self.val_ds_mock = MagicMock()
        self.val_ds_mock.__len__.return_value = 100
        self.dm.val_dataset = self.val_ds_mock # type: ignore
        
        self.test_ds_mock = MagicMock()
        self.test_ds_mock.__len__.return_value = 100
        self.dm.test_dataset = self.test_ds_mock # type: ignore
        
        self.dm._is_initialized = True

    def test_train_dataloader_uses_configured_seed(self):
        """Test that user configured seed is used as base."""
        # epoch=0, rank=0, seed=123
        self.dm.train_dataloader()
        self.train_ds_mock.resample.assert_called_with(seed=123)

    def test_train_dataloader_changes_with_epoch(self):
        """Test seed changes with epoch."""
        # epoch=1, rank=0, seed=123
        self.trainer_mock.current_epoch = 1
        self.dm.train_dataloader()
        self.train_ds_mock.resample.assert_called_with(seed=124)

    def test_train_dataloader_changes_with_rank(self):
        """Test seed changes with rank."""
        # epoch=0, rank=1, world_size=2, seed=123
        self.trainer_mock.current_epoch = 0
        self.trainer_mock.global_rank = 1
        self.trainer_mock.world_size = 2
        
        self.dm.train_dataloader()
        self.train_ds_mock.resample.assert_called_with(seed=124)
        
    def test_train_dataloader_complex_formula(self):
        """Test the full formula: base + epoch*world + rank"""
        # epoch=5, rank=3, world_size=4, seed=100
        self.dm.seed = 100
        self.trainer_mock.current_epoch = 5
        self.trainer_mock.global_rank = 3
        self.trainer_mock.world_size = 4
        
        self.dm.train_dataloader()
        self.train_ds_mock.resample.assert_called_with(seed=123)

    def test_train_dataloader_seed_zero(self):
        """Test behavior when seed is 0."""
        self.dm.seed = 0
        # epoch=0, rank=0. Expected: 0 + 0 + 0 = 0
        self.dm.train_dataloader()
        self.train_ds_mock.resample.assert_called_with(seed=0)

        # epoch=1. Expected: 0 + 1*1 + 0 = 1
        self.trainer_mock.current_epoch = 1
        self.dm.train_dataloader()
        self.train_ds_mock.resample.assert_called_with(seed=1)

    def test_val_dataloader_no_resample(self):
        """Test that validation dataloader does NOT call resample."""
        self.dm.val_dataloader()
        self.val_ds_mock.resample.assert_not_called()

if __name__ == '__main__':
    unittest.main()
