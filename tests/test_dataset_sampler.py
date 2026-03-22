import pytest

from cerberus.config import DataConfig, SamplerConfig
from cerberus.dataset import CerberusDataset
from cerberus.genome import create_genome_config
from cerberus.samplers import IntervalSampler


def test_dataset_instantiates_interval_sampler(tmp_path):
    # 1. Setup Files
    genome = tmp_path / "genome.fa"
    # Write valid content so SequenceExtractor works
    genome.write_text(">chr1\n" + "N" * 1000 + "\n")
    fai = tmp_path / "genome.fa.fai"
    fai.write_text("chr1\t1000\t6\t1000\t1001\n")

    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t200\nchr1\t500\t600\n")

    blacklist = tmp_path / "blacklist.bed"
    blacklist.write_text("chr1\t150\t250\n") # Overlaps first peak

    # 2. Configs
    genome_config = create_genome_config(
        name="test_genome",
        fasta_path=genome,
        species="human",
        allowed_chroms=["chr1"],
        exclude_intervals={"blacklist": blacklist}
    )

    data_config = DataConfig.model_construct(
        inputs={},
        targets={},
        input_len=200,
        output_len=200,
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
        padded_size=200,
        sampler_args={"intervals_path": peaks},
    )

    # 3. Instantiate
    ds = CerberusDataset(genome_config, data_config, sampler_config, sequence_extractor=None, sampler=None, exclude_intervals=None)

    # 4. Verify Sampler
    assert isinstance(ds.sampler, IntervalSampler)
    assert ds.sampler.padded_size == 200
    assert ds.sampler.chrom_sizes == genome_config.chrom_sizes
    # Check exclude intervals exists for chr1
    assert ds.sampler.exclude_intervals is not None
    assert "chr1" in ds.sampler.exclude_intervals
    assert len(ds.sampler.exclude_intervals["chr1"]) > 0

    # 5. Verify Content
    # First peak (100-200). Padded 200 -> Center 150. Start 50, End 250.
    # Blacklist (150-250). Overlaps.
    # Should be excluded.

    # Second peak (500-600). Center 550. Start 450, End 650.
    # Valid.

    assert len(ds) == 1

    # Check sampler directly for interval logic
    interval = ds.sampler[0]
    assert interval.chrom == "chr1"
    # Peak 500-600, center 550. Padded 200 -> start 450, end 650.
    assert interval.start == 450
    assert interval.end == 650

    # Check dataset return values
    out = ds[0]
    seq = out["inputs"]
    assert seq.shape == (4, 200)

def test_dataset_invalid_sampler(tmp_path):
    genome = tmp_path / "genome.fa"
    genome.touch()
    (tmp_path / "genome.fa.fai").touch()

    genome_config = create_genome_config(name="test", fasta_path=genome, species="human")
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
    sampler_config = SamplerConfig.model_construct(
        sampler_type="unknown",
        padded_size=100,
        sampler_args=None,
    )

    with pytest.raises(ValueError, match="Unsupported sampler type"):
        CerberusDataset(genome_config, data_config, sampler_config, sequence_extractor=None, sampler=None, exclude_intervals=None)
