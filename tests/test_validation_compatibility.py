import pytest
from pathlib import Path
from pydantic import ValidationError
from cerberus.config import (
    CerberusConfig,
    DataConfig,
    SamplerConfig,
    ModelConfig,
    TrainConfig,
    GenomeConfig,
    FoldArgs,
    SlidingWindowSamplerArgs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _genome_config(tmp_path: Path) -> GenomeConfig:
    fasta = tmp_path / "genome.fa"
    fasta.touch()
    return GenomeConfig(
        name="test",
        fasta_path=fasta,
        exclude_intervals={},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 10000},
        fold_type="chrom_partition",
        fold_args=FoldArgs(k=2),
    )


def _train_config() -> TrainConfig:
    return TrainConfig(
        batch_size=32,
        max_epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=5,
        optimizer="adam",
        filter_bias_and_bn=True,
        scheduler_type="cosine",
        scheduler_args={"T_max": 10},
        reload_dataloaders_every_n_epochs=1,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="m",
        model_cls="torch.nn.Linear",
        loss_cls="cerberus.loss.MSEMultinomialLoss",
        loss_args={},
        metrics_cls="torchmetrics.MetricCollection",
        metrics_args={},
        model_args={},
        pretrained=[],
    )


def _data_config(input_len: int = 100, max_jitter: int = 50) -> DataConfig:
    return DataConfig(
        inputs={},
        targets={},
        input_len=input_len,
        output_len=50,
        max_jitter=max_jitter,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        target_scale=1.0,
        use_sequence=True,
    )


def _sampler_config(padded_size: int) -> SamplerConfig:
    return SamplerConfig(
        sampler_type="sliding_window",
        padded_size=padded_size,
        sampler_args=SlidingWindowSamplerArgs(stride=50),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_data_and_sampler_compatibility_valid(tmp_path):
    """padded_size == input_len + 2 * max_jitter should pass."""
    CerberusConfig(
        train_config=_train_config(),
        genome_config=_genome_config(tmp_path),
        data_config=_data_config(input_len=100, max_jitter=50),
        sampler_config=_sampler_config(padded_size=200),
        model_config=_model_config(),
    )


def test_data_and_sampler_compatibility_exact_boundary(tmp_path):
    """padded_size exactly at boundary should pass."""
    CerberusConfig(
        train_config=_train_config(),
        genome_config=_genome_config(tmp_path),
        data_config=_data_config(input_len=100, max_jitter=50),
        sampler_config=_sampler_config(padded_size=200),
        model_config=_model_config(),
    )


def test_data_and_sampler_compatibility_invalid(tmp_path):
    """padded_size < input_len + 2 * max_jitter should raise."""
    with pytest.raises(ValidationError, match=r"Sampler padded_size \(199\) is smaller than required size"):
        CerberusConfig(
            train_config=_train_config(),
            genome_config=_genome_config(tmp_path),
            data_config=_data_config(input_len=100, max_jitter=50),
            sampler_config=_sampler_config(padded_size=199),
            model_config=_model_config(),
        )


def test_data_and_sampler_compatibility_large_jitter(tmp_path):
    """Large jitter: required = 1000 + 2*500 = 2000, padded_size=1500 should fail."""
    with pytest.raises(ValidationError, match=r"Sampler padded_size \(1500\) is smaller than required size"):
        CerberusConfig(
            train_config=_train_config(),
            genome_config=_genome_config(tmp_path),
            data_config=_data_config(input_len=1000, max_jitter=500),
            sampler_config=_sampler_config(padded_size=1500),
            model_config=_model_config(),
        )
