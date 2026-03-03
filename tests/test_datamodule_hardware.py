import torch
import pytest
from unittest.mock import MagicMock, patch
from cerberus.datamodule import CerberusDataModule

def test_datamodule_hardware_settings():
    """
    Verify hardware-specific settings in CerberusDataModule:
    1. pin_memory disabled on MPS.
    2. persistent_workers usage.
    3. multiprocessing_context configuration.
    """
    
    # Mock validators
    with patch('cerberus.datamodule.validate_genome_config') as mock_genome, \
         patch('cerberus.datamodule.validate_data_config') as mock_data, \
         patch('cerberus.datamodule.validate_sampler_config') as mock_sampler, \
         patch('cerberus.datamodule.validate_data_and_sampler_compatibility') as mock_compat, \
         patch('cerberus.datamodule.CerberusDataset') as mock_dataset_cls:
        
        mock_genome.return_value = {"fold_args": {"test_fold": 0, "val_fold": 1}}
        mock_data.return_value = {}
        mock_sampler.return_value = {"sampler_type": "random"}
        
        # Mock dataset instance for split_folds
        mock_dataset_instance = MagicMock()
        mock_dataset_cls.return_value = mock_dataset_instance
        
        # split_folds returns tuple of 3 datasets
        train_ds = MagicMock()
        train_ds.__len__.return_value = 10
        val_ds = MagicMock()
        val_ds.__len__.return_value = 10
        test_ds = MagicMock()
        test_ds.__len__.return_value = 10
        
        mock_dataset_instance.split_folds.return_value = (train_ds, val_ds, test_ds)
        
        # --- Test pin_memory on MPS ---
        mps_available = torch.backends.mps.is_available()
        
        dm = CerberusDataModule(
            genome_config={}, # type: ignore
            data_config={}, # type: ignore
            sampler_config={"sampler_type": "random"}, # type: ignore
            pin_memory=True
        )
        
        if mps_available:
            assert dm.pin_memory is False, "pin_memory should be disabled on MPS"
        else:
            assert dm.pin_memory is True, "pin_memory should remain True on non-MPS"

        dm_explicit = CerberusDataModule(
            genome_config={}, # type: ignore
            data_config={}, # type: ignore
            sampler_config={"sampler_type": "random"}, # type: ignore
            pin_memory=False
        )
        assert dm_explicit.pin_memory is False, "pin_memory should remain False when explicitly set"
        
        # --- Test persistent_workers ---
        
        # Case 1: Default (persistent_workers=True), num_workers=0 (default)
        dm.setup() # Initialize datasets
        val_loader = dm.val_dataloader()
        assert val_loader.persistent_workers is False, "persistent_workers must be False if num_workers=0"
        
        # Case 2: Default (persistent_workers=True), num_workers=2
        dm.setup(num_workers=2) # Update num_workers
        val_loader = dm.val_dataloader()
        assert val_loader.persistent_workers is True, "persistent_workers should be True in val_loader when enabled and num_workers > 0"
        
        train_loader = dm.train_dataloader()
        assert train_loader.persistent_workers is True, "persistent_workers should be True in train_loader when enabled and num_workers > 0"
        
        # Case 3: Explicitly disabled, num_workers=2
        dm_disabled = CerberusDataModule(
            genome_config={}, # type: ignore
            data_config={}, # type: ignore
            sampler_config={"sampler_type": "random"}, # type: ignore
            persistent_workers=False
        )
        dm_disabled.setup(num_workers=2)
        val_loader = dm_disabled.val_dataloader()
        assert val_loader.persistent_workers is False, "persistent_workers should be False when explicitly disabled"
        
        train_loader = dm_disabled.train_dataloader()
        assert train_loader.persistent_workers is False, "persistent_workers should be False in train_loader when explicitly disabled"
        
        # --- Test multiprocessing_context ---
        
        # Case 1: Default (None)
        dm_ctx = CerberusDataModule(
            genome_config={}, # type: ignore
            data_config={}, # type: ignore
            sampler_config={"sampler_type": "random"} # type: ignore
        )
        dm_ctx.setup(num_workers=2)
        val_loader = dm_ctx.val_dataloader()
        assert val_loader.multiprocessing_context is None, "Default multiprocessing_context should be None"
        
        # Case 2: Explicit 'spawn'
        dm_spawn = CerberusDataModule(
            genome_config={}, # type: ignore
            data_config={}, # type: ignore
            sampler_config={"sampler_type": "random"}, # type: ignore
            multiprocessing_context='spawn'
        )
        dm_spawn.setup(num_workers=2)
        val_loader = dm_spawn.val_dataloader()
        if val_loader.multiprocessing_context is not None:
             ctx_repr = str(val_loader.multiprocessing_context)
             assert 'spawn' in ctx_repr.lower() or 'Spawn' in type(val_loader.multiprocessing_context).__name__, \
                 f"Context should be spawn, got {val_loader.multiprocessing_context}"

if __name__ == "__main__":
    try:
        test_datamodule_hardware_settings()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        import sys
        sys.exit(1)
