import pytest
import torch
from cerberus.dataset import CerberusDataset
from cerberus.interval import Interval
from cerberus.genome import create_genome_config
from cerberus.config import DataConfig, SamplerConfig, IntervalSamplerArgs


class MockTargetExtractor:
    """Returns a deterministic signal based on interval coordinates for testing."""

    def extract(self, interval: Interval) -> torch.Tensor:
        length = interval.end - interval.start
        # Return ascending values [1, 2, ..., length] so we can verify no transforms applied
        return torch.arange(1, length + 1, dtype=torch.float32).unsqueeze(0)  # (1, L)


@pytest.fixture
def dataset_with_targets(tmp_path):
    """Create a CerberusDataset with a mock target extractor and transforms that modify targets."""
    genome = tmp_path / "genome.fa"
    genome.write_text(">chr1\n" + "A" * 2000 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text("chr1\t2000\t6\t2000\t2001\n")

    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t200\n")

    genome_config = create_genome_config(
        name="test", fasta_path=genome, species="human",
        allowed_chroms=["chr1"], exclude_intervals={},
    )

    # input_len > output_len to test cropping
    # log_transform=True and output_bin_size=2 to ensure transforms modify targets
    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=100,
        output_len=50,
        output_bin_size=2,
        encoding="ACGT",
        max_jitter=0,
        log_transform=True,
        reverse_complement=False,
        target_scale=10.0,
        use_sequence=True,
    )

    sampler_config = SamplerConfig.model_construct(
        sampler_type="interval",
        padded_size=100,
        sampler_args=IntervalSamplerArgs.model_construct(intervals_path=peaks),
    )

    ds = CerberusDataset(
        genome_config, data_config, sampler_config,
        target_signal_extractor=MockTargetExtractor(),
        sequence_extractor=None, sampler=None, exclude_intervals={},
    )
    return ds


class TestGetRawTargets:
    def test_bypasses_transforms(self, dataset_with_targets):
        """get_raw_targets should return untransformed signal -- no binning, log, or scaling."""
        ds = dataset_with_targets
        interval = Interval("chr1", 100, 200)  # 100bp, matches input_len

        raw = ds.get_raw_targets(interval, crop_to_output_len=True)

        # MockTargetExtractor returns [1..100] for a 100bp interval
        # After crop to output_len=50: center crop from idx 25 to 75 -> [26..75]
        assert raw.shape == (1, 50)
        expected = torch.arange(26, 76, dtype=torch.float32)
        assert torch.allclose(raw[0], expected)

    def test_get_interval_applies_transforms(self, dataset_with_targets):
        """Verify that get_interval DOES transform (to confirm get_raw_targets is different)."""
        ds = dataset_with_targets
        interval = Interval("chr1", 100, 200)

        data = ds.get_interval(interval)
        targets = data["targets"]

        # With output_bin_size=2, output_len=50 -> 25 bins after binning
        # With target_scale=10.0 and log_transform=True, values are modified
        assert targets.shape[-1] == 25  # binned to 25 positions
        # Values should NOT be the raw ascending sequence
        raw = ds.get_raw_targets(interval, crop_to_output_len=True)
        assert not torch.allclose(targets, raw[..., :25])

    def test_no_crop(self, dataset_with_targets):
        """crop_to_output_len=False should return the full extracted signal."""
        ds = dataset_with_targets
        interval = Interval("chr1", 100, 200)  # 100bp

        raw = ds.get_raw_targets(interval, crop_to_output_len=False)

        assert raw.shape == (1, 100)
        expected = torch.arange(1, 101, dtype=torch.float32)
        assert torch.allclose(raw[0], expected)

    def test_crop_when_input_equals_output(self, tmp_path):
        """When input_len == output_len, crop is a no-op."""
        genome = tmp_path / "genome.fa"
        genome.write_text(">chr1\n" + "A" * 2000 + "\n")
        fai = tmp_path / "genome.fa.fai"
        fai.write_text("chr1\t2000\t6\t2000\t2001\n")

        peaks = tmp_path / "peaks.bed"
        peaks.write_text("chr1\t100\t200\n")

        genome_config = create_genome_config(
            name="test", fasta_path=genome, species="human",
            allowed_chroms=["chr1"], exclude_intervals={},
        )

        data_config = DataConfig.model_construct(
            inputs={},
            targets={},
            input_len=50,
            output_len=50,
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
            padded_size=50,
            sampler_args=IntervalSamplerArgs.model_construct(intervals_path=peaks),
        )

        ds = CerberusDataset(
            genome_config, data_config, sampler_config,
            target_signal_extractor=MockTargetExtractor(),
            sequence_extractor=None, sampler=None, exclude_intervals={},
        )

        raw_cropped = ds.get_raw_targets(Interval("chr1", 100, 150), crop_to_output_len=True)
        raw_full = ds.get_raw_targets(Interval("chr1", 100, 150), crop_to_output_len=False)
        assert torch.allclose(raw_cropped, raw_full)

    def test_no_target_extractor_raises(self, tmp_path):
        """Should raise RuntimeError if target signal extractor is not initialized."""
        genome = tmp_path / "genome.fa"
        genome.write_text(">chr1\n" + "A" * 2000 + "\n")
        fai = tmp_path / "genome.fa.fai"
        fai.write_text("chr1\t2000\t6\t2000\t2001\n")

        peaks = tmp_path / "peaks.bed"
        peaks.write_text("chr1\t100\t200\n")

        genome_config = create_genome_config(
            name="test", fasta_path=genome, species="human",
            allowed_chroms=["chr1"], exclude_intervals={},
        )

        data_config = DataConfig.model_construct(
            inputs={},
            targets={},
            input_len=50,
            output_len=50,
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
            padded_size=50,
            sampler_args=IntervalSamplerArgs.model_construct(intervals_path=peaks),
        )

        ds = CerberusDataset(
            genome_config, data_config, sampler_config,
            target_signal_extractor=None,
            sequence_extractor=None, sampler=None, exclude_intervals={},
        )

        with pytest.raises(RuntimeError, match="Target signal extractor is not initialized"):
            ds.get_raw_targets(Interval("chr1", 100, 150))

    def test_accepts_string_query(self, dataset_with_targets):
        """Should accept 'chr:start-end' string format."""
        raw = dataset_with_targets.get_raw_targets("chr1:100-200", crop_to_output_len=True)
        assert raw.shape == (1, 50)

    def test_accepts_tuple_query(self, dataset_with_targets):
        """Should accept (chrom, start, end) tuple format."""
        raw = dataset_with_targets.get_raw_targets(("chr1", 100, 200), crop_to_output_len=True)
        assert raw.shape == (1, 50)

    def test_accepts_interval_query(self, dataset_with_targets):
        """Should accept Interval object."""
        raw = dataset_with_targets.get_raw_targets(Interval("chr1", 100, 200), crop_to_output_len=True)
        assert raw.shape == (1, 50)
