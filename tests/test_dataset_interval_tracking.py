import pytest
import torch

from cerberus.config import DataConfig, GenomeConfig, SamplerConfig
from cerberus.dataset import CerberusDataset
from cerberus.transform import Jitter, ReverseComplement


# Mocking helpers
class MockSequenceExtractor:
    def extract(self, interval):
        # Return dummy sequence tensor (4, len)
        length = interval.end - interval.start
        return torch.zeros(4, length)


@pytest.fixture
def mock_dataset(tmp_path):
    # Minimal config to instantiate Dataset
    # We will override components manually to test specific transforms

    genome_path = tmp_path / "genome.fa"
    genome_path.touch()

    genome_config = GenomeConfig.model_construct(
        name="test",
        fasta_path=genome_path,
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 1000},
        exclude_intervals={},
        fold_type="chrom_partition",
        fold_args={"k": 2, "test_fold": None, "val_fold": None},
    )

    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=100,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    (tmp_path / "dummy.bed").write_text("chr1\t100\t200\n")

    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=100,
        sampler_args={"intervals_path": tmp_path / "dummy.bed"},
    )

    ds = CerberusDataset(
        genome_config,
        data_config,
        sampler_config,
        sequence_extractor=MockSequenceExtractor(),
    )
    return ds


def test_interval_string_jitter(mock_dataset):
    # Manually set transforms
    mock_dataset.transforms.transforms = [Jitter(input_len=50, max_jitter=0)]

    assert len(mock_dataset) == 1

    item = mock_dataset[0]
    interval_str = item["intervals"]

    # Expected: chr1:125-175(+)
    assert interval_str == "chr1:125-175(+)"


def test_interval_string_rc(mock_dataset):
    # Config: RC probability 1.0
    mock_dataset.transforms.transforms = [ReverseComplement(probability=1.0)]

    item = mock_dataset[0]
    interval_str = item["intervals"]

    # Expected: chr1:100-200(-)
    assert interval_str == "chr1:100-200(-)"


def test_interval_string_jitter_and_rc(mock_dataset):
    # Chain: Jitter -> RC
    mock_dataset.transforms.transforms = [
        Jitter(input_len=50, max_jitter=0),  # -> 125-175(+)
        ReverseComplement(probability=1.0),  # -> 125-175(-)
    ]

    item = mock_dataset[0]
    interval_str = item["intervals"]

    assert interval_str == "chr1:125-175(-)"
