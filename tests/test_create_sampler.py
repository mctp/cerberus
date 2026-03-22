import pytest

from cerberus.config import SamplerConfig
from cerberus.samplers import create_sampler


def test_create_sampler_unknown_type():
    """Unknown sampler type raises ValueError from create_sampler."""
    cfg = SamplerConfig(
        sampler_type="unknown_type",
        padded_size=100,
        sampler_args={"stride": 50},
    )
    with pytest.raises(ValueError, match="Unsupported sampler type: unknown_type"):
        create_sampler(cfg, chrom_sizes={}, exclude_intervals={}, folds=[])


def test_create_sampler_random(tmp_path):
    cfg = SamplerConfig(
        sampler_type="random",
        padded_size=100,
        sampler_args={"num_intervals": 10},
    )
    chrom_sizes = {"chr1": 1000}
    sampler = create_sampler(
        cfg, chrom_sizes=chrom_sizes, exclude_intervals={}, folds=[]
    )
    assert len(sampler) == 10
