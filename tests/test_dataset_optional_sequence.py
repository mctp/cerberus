import pytest
import torch

from cerberus.config import DataConfig, SamplerConfig
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_config


@pytest.fixture
def genome_setup(tmp_path):
    genome_path = tmp_path / "genome.fa"
    genome_path.write_text(">chr1\nACGTACGTACGTACGT")

    # Create .fai file
    fai_path = tmp_path / "genome.fa.fai"
    fai_path.write_text("chr1\t16\t6\t16\t17\n")

    # Create peaks
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t0\t10\n")

    # Create bigwig inputs
    inputs = tmp_path / "input.bw"
    inputs.touch()

    return genome_path, peaks, inputs


def test_dataset_no_sequence_init(genome_setup):
    genome_path, peaks, input_path = genome_setup

    # Create dummy bigwig class or mock to avoid reading invalid file
    class MockSignalExtractor:
        def extract(self, interval):
            return torch.ones(1, interval.end - interval.start)

    genome_config = create_genome_config(
        name="test", fasta_path=genome_path, species="human", allowed_chroms=["chr1"]
    )

    data_config = DataConfig.model_construct(
        inputs={"track": input_path},
        targets={},
        input_len=4,
        output_len=4,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=False,
    )

    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=4,
        sampler_args={"intervals_path": peaks},
    )

    # Initialize with mock signal extractor to bypass file reading
    ds = CerberusDataset(
        genome_config,
        data_config,
        sampler_config,
        input_signal_extractor=MockSignalExtractor(),
        sequence_extractor=None,
    )

    assert ds.sequence_extractor is None

    # Check item
    sample = ds[0]
    # Inputs should only have 1 channel (from mock signal)
    assert sample["inputs"].shape == (1, 4)
    assert torch.all(sample["inputs"] == 1.0)


def test_dataset_no_sequence_default_extractor(genome_setup):
    genome_path, peaks, input_path = genome_setup

    genome_config = create_genome_config(
        name="test", fasta_path=genome_path, species="human", allowed_chroms=["chr1"]
    )

    # Case: No sequence, No signals -> Error
    data_config_no_inputs = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=4,
        output_len=4,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=False,
    )

    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=4,
        sampler_args={"intervals_path": peaks},
    )

    with pytest.raises(ValueError, match="No input sources provided"):
        CerberusDataset(genome_config, data_config_no_inputs, sampler_config)


def test_dataset_split_preserves_no_sequence(genome_setup):
    genome_path, peaks, input_path = genome_setup

    class MockSignalExtractor:
        def extract(self, interval):
            return torch.ones(1, interval.end - interval.start)

    genome_config = create_genome_config(
        name="test", fasta_path=genome_path, species="human", allowed_chroms=["chr1"]
    )

    data_config = DataConfig.model_construct(
        inputs={"track": input_path},
        targets={},
        input_len=4,
        output_len=4,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=False,
    )

    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=4,
        sampler_args={"intervals_path": peaks},
    )

    ds = CerberusDataset(
        genome_config,
        data_config,
        sampler_config,
        input_signal_extractor=MockSignalExtractor(),
    )

    assert ds.sequence_extractor is None

    # Split folds
    train, val, test = ds.split_folds()

    assert train.sequence_extractor is None
    assert val.sequence_extractor is None
    assert test.sequence_extractor is None

    assert train.data_config.use_sequence is False


def test_dataset_use_sequence_true(genome_setup):
    genome_path, peaks, input_path = genome_setup

    genome_config = create_genome_config(
        name="test", fasta_path=genome_path, species="human", allowed_chroms=["chr1"]
    )

    # use_sequence defaults to True or explicit True
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=4,
        output_len=4,
        output_bin_size=1,
        encoding="ACGT",
        max_jitter=0,
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )

    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=4,
        sampler_args={"intervals_path": peaks},
    )

    # Should initialize sequence extractor
    ds = CerberusDataset(genome_config, data_config, sampler_config)
    assert ds.sequence_extractor is not None

    sample = ds[0]
    assert sample["inputs"].shape == (4, 4)
