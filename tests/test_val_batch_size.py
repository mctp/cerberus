from unittest.mock import MagicMock, patch

import pytest

from cerberus.config import DataConfig, GenomeConfig, SamplerConfig
from cerberus.datamodule import CerberusDataModule


@pytest.fixture
def mock_dataset_cls():
    with (
        patch("cerberus.datamodule.CerberusDataset") as mock,
        patch.object(CerberusDataModule, "_validate_paths"),
    ):
        yield mock


def test_val_batch_size_configuration(
    mock_dataset_cls,
):
    # Setup mocks
    mock_instance = MagicMock()
    mock_dataset_cls.return_value = mock_instance
    train_ds = MagicMock()
    val_ds = MagicMock()
    test_ds = MagicMock()
    # Mocks for len so dataloader init works
    train_ds.__len__.return_value = 10
    val_ds.__len__.return_value = 10
    test_ds.__len__.return_value = 10

    mock_instance.split_folds.return_value = (train_ds, val_ds, test_ds)

    genome_config = GenomeConfig.model_construct(
        name="mock_genome",
        fasta_path="mock.fa",
        chrom_sizes={"chr1": 1000},
        allowed_chroms=["chr1"],
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 5, "test_fold": 0, "val_fold": 1},
    )
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=100,
        max_jitter=0,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )
    sampler_config = SamplerConfig.model_construct(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )

    dm = CerberusDataModule(genome_config, data_config, sampler_config)

    # 1. Test default behavior (no val_batch_size provided)
    dm.setup(batch_size=16)
    assert dm.batch_size == 16
    assert dm.val_batch_size == 16

    assert dm.train_dataloader().batch_size == 16
    assert dm.val_dataloader().batch_size == 16
    assert dm.test_dataloader().batch_size == 16

    # Reset
    dm._is_initialized = False

    # 2. Test explicit val_batch_size
    dm.setup(batch_size=16, val_batch_size=32)
    assert dm.batch_size == 16
    assert dm.val_batch_size == 32

    assert dm.train_dataloader().batch_size == 16
    assert dm.val_dataloader().batch_size == 32
    assert dm.test_dataloader().batch_size == 32

    # Reset
    dm._is_initialized = False

    # 3. Test setup without changing default (batch_size None)
    dm.setup(batch_size=8)
    assert dm.batch_size == 8
    assert dm.val_batch_size == 8

    # Reset
    dm._is_initialized = False

    dm.setup(batch_size=10, val_batch_size=20)
    assert dm.batch_size == 10
    assert dm.val_batch_size == 20
