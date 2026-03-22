from pathlib import Path

import pytest
import torch.nn as nn
from pydantic import ValidationError

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    GenomeConfig,
    ModelConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.output import get_log_count_params

# ---------------------------------------------------------------------------
# Helpers — minimal valid configs for cross-validation tests
# ---------------------------------------------------------------------------

def _genome_config(tmp_path: Path) -> dict:
    """Return a minimal valid GenomeConfig dict (files must exist)."""
    fasta = tmp_path / "genome.fa"
    fasta.touch()
    return {
        "name": "test",
        "fasta_path": str(fasta),
        "exclude_intervals": {},
        "allowed_chroms": ["chr1"],
        "chrom_sizes": {"chr1": 10000},
        "fold_type": "chrom_partition",
        "fold_args": {"k": 2},
    }

def _train_config_dict() -> dict:
    return {
        "batch_size": 32,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "adam",
        "filter_bias_and_bn": True,
        "scheduler_type": "cosine",
        "scheduler_args": {"T_max": 10},
        "reload_dataloaders_every_n_epochs": 1,
        "adam_eps": 1e-8,
        "gradient_clip_val": None,
    }

def _model_config_dict(**overrides) -> dict:
    base = {
        "name": "m",
        "model_cls": "torch.nn.Linear",
        "loss_cls": "cerberus.loss.MSEMultinomialLoss",
        "loss_args": {},
        "metrics_cls": "torchmetrics.MetricCollection",
        "metrics_args": {},
        "model_args": {},
        "pretrained": [],
    }
    base.update(overrides)
    return base

# --- Data Config Tests ---

def test_validate_data_config_rc_without_sequence():
    """reverse_complement=True with use_sequence=False should raise."""
    with pytest.raises(ValidationError, match="reverse_complement"):
        DataConfig(
            inputs={},
            targets={},
            input_len=100,
            output_len=100,
            max_jitter=0,
            output_bin_size=1,
            encoding="ACGT",
            log_transform=False,
            reverse_complement=True,
            use_sequence=False,
            target_scale=1.0,
        )

# --- Sampler Config Tests ---

def test_validate_sampler_config_sliding_window():
    cfg = SamplerConfig(
        sampler_type="sliding_window",
        padded_size=1000,
        sampler_args={"stride": 50},
    )
    assert cfg.sampler_type == "sliding_window"

def test_validate_sampler_config_accepts_plain_dict():
    """sampler_args is now a plain dict — empty dict is accepted at config level."""
    cfg = SamplerConfig(
        sampler_type="interval",
        padded_size=1000,
        sampler_args={},
    )
    assert cfg.sampler_args == {}

# --- Model Config Tests ---

class DummyLoss(nn.Module):
    def forward(self, x, y): return 0

def test_validate_model_config_valid():
    cfg = ModelConfig(
        name="test_model",
        model_cls="torch.nn.Linear",
        loss_cls="tests.test_config_validation.DummyLoss",
        loss_args={},
        metrics_cls="torchmetrics.MetricCollection",
        metrics_args={},
        model_args={
            "input_channels": ["A", "C", "G", "T"],
            "output_channels": ["signal"],
            "output_type": "signal",
        },
        pretrained=[],
    )
    assert cfg.name == "test_model"
    assert cfg.model_args["input_channels"] == ["A", "C", "G", "T"]


# --- Compatibility Tests (via CerberusConfig cross-validation) ---

def test_data_and_sampler_compatibility_valid(tmp_path):
    """padded_size == input_len + 2*max_jitter should pass."""
    CerberusConfig(
        train_config=TrainConfig(**_train_config_dict()),
        genome_config=GenomeConfig(**_genome_config(tmp_path)),
        data_config=DataConfig(
            inputs={}, targets={},
            input_len=100, output_len=100, max_jitter=10,
            output_bin_size=1, encoding="ACGT", log_transform=False,
            reverse_complement=False, target_scale=1.0, use_sequence=True,
        ),
        sampler_config=SamplerConfig(
            sampler_type="sliding_window", padded_size=120,
            sampler_args={"stride": 50},
        ),
        model_config=ModelConfig(**_model_config_dict()),
    )

def test_data_and_sampler_compatibility_invalid(tmp_path):
    """padded_size < input_len + 2*max_jitter should raise."""
    with pytest.raises(ValidationError, match="Sampler padded_size"):
        CerberusConfig(
            train_config=TrainConfig(**_train_config_dict()),
            genome_config=GenomeConfig(**_genome_config(tmp_path)),
            data_config=DataConfig(
                inputs={}, targets={},
                input_len=100, output_len=100, max_jitter=10,
                output_bin_size=1, encoding="ACGT", log_transform=False,
                reverse_complement=False, target_scale=1.0, use_sequence=True,
            ),
            sampler_config=SamplerConfig(
                sampler_type="sliding_window", padded_size=119,
                sampler_args={"stride": 50},
            ),
            model_config=ModelConfig(**_model_config_dict()),
        )

def test_data_and_model_compatibility_valid(tmp_path):
    cons = tmp_path / "cons.bw"
    cons.touch()
    tgt = tmp_path / "tgt.bw"
    tgt.touch()
    CerberusConfig(
        train_config=TrainConfig(**_train_config_dict()),
        genome_config=GenomeConfig(**_genome_config(tmp_path)),
        data_config=DataConfig(
            inputs={"track1": cons},
            targets={"target1": tgt},
            input_len=100, output_len=100, max_jitter=0,
            output_bin_size=1, encoding="ACGT", log_transform=False,
            reverse_complement=False, target_scale=1.0, use_sequence=True,
        ),
        sampler_config=SamplerConfig(
            sampler_type="sliding_window", padded_size=100,
            sampler_args={"stride": 50},
        ),
        model_config=ModelConfig(**_model_config_dict(
            model_args={
                "input_channels": ["track1", "A", "C", "G", "T"],
                "output_channels": ["target1"],
                "output_type": "signal",
            },
        )),
    )

def test_data_and_model_compatibility_invalid_targets(tmp_path):
    tgt = tmp_path / "tgt.bw"
    tgt.touch()
    with pytest.raises(ValidationError, match="Model output channels"):
        CerberusConfig(
            train_config=TrainConfig(**_train_config_dict()),
            genome_config=GenomeConfig(**_genome_config(tmp_path)),
            data_config=DataConfig(
                inputs={}, targets={"target1": tgt},
                input_len=100, output_len=100, max_jitter=0,
                output_bin_size=1, encoding="ACGT", log_transform=False,
                reverse_complement=False, target_scale=1.0, use_sequence=True,
            ),
            sampler_config=SamplerConfig(
                sampler_type="sliding_window", padded_size=100,
                sampler_args={"stride": 50},
            ),
            model_config=ModelConfig(**_model_config_dict(
                model_args={
                    "input_channels": ["A"],
                    "output_channels": ["target2"],  # Mismatch
                    "output_type": "signal",
                },
            )),
        )

def test_data_and_model_compatibility_invalid_inputs(tmp_path):
    track = tmp_path / "track.bw"
    track.touch()
    with pytest.raises(ValidationError, match="Data inputs .* are not in model input channels"):
        CerberusConfig(
            train_config=TrainConfig(**_train_config_dict()),
            genome_config=GenomeConfig(**_genome_config(tmp_path)),
            data_config=DataConfig(
                inputs={"track1": track}, targets={},
                input_len=100, output_len=100, max_jitter=0,
                output_bin_size=1, encoding="ACGT", log_transform=False,
                reverse_complement=False, target_scale=1.0, use_sequence=True,
            ),
            sampler_config=SamplerConfig(
                sampler_type="sliding_window", padded_size=100,
                sampler_args={"stride": 50},
            ),
            model_config=ModelConfig(**_model_config_dict(
                model_args={
                    "input_channels": ["A"],  # Missing track1
                    "output_type": "signal",
                },
            )),
        )

# --- ModelConfig.count_pseudocount (first-class field) ---

def test_model_config_count_pseudocount_default():
    """count_pseudocount defaults to 0.0."""
    cfg = ModelConfig(**_model_config_dict())
    assert cfg.count_pseudocount == 0.0

def test_model_config_count_pseudocount_set():
    """count_pseudocount can be set explicitly."""
    cfg = ModelConfig(**_model_config_dict(count_pseudocount=150.0))
    assert cfg.count_pseudocount == 150.0

def test_model_config_count_pseudocount_negative_rejected():
    """Negative count_pseudocount should be rejected."""
    with pytest.raises(ValidationError, match="count_pseudocount"):
        ModelConfig(**_model_config_dict(count_pseudocount=-5.0))

def test_model_config_count_pseudocount_zero():
    """count_pseudocount=0.0 is valid (for Poisson/NB losses)."""
    cfg = ModelConfig(**_model_config_dict(count_pseudocount=0.0))
    assert cfg.count_pseudocount == 0.0
    assert isinstance(cfg.count_pseudocount, float)

def test_model_config_count_pseudocount_coerced_to_float():
    """Integer count_pseudocount is coerced to float."""
    cfg = ModelConfig(**_model_config_dict(count_pseudocount=100))
    assert cfg.count_pseudocount == 100.0
    assert isinstance(cfg.count_pseudocount, float)

# --- get_log_count_params ---

def _model_config_with_loss(loss_cls: str, loss_args: dict | None = None, count_pseudocount: float = 0.0):
    """Return a ModelConfig with the given loss class."""
    return ModelConfig(
        name="m",
        model_cls="torch.nn.Linear",
        loss_cls=loss_cls,
        loss_args=loss_args or {},
        metrics_cls="torchmetrics.MetricCollection",
        metrics_args={},
        model_args={},
        pretrained=[],
        count_pseudocount=count_pseudocount,
    )

def test_get_log_count_params_mse():
    """MSE loss returns uses_pseudocount=True and reads count_pseudocount from model_config."""
    conf = _model_config_with_loss(
        "cerberus.loss.MSEMultinomialLoss",
        count_pseudocount=50.0,
    )
    includes, pseudocount = get_log_count_params(conf)
    assert includes is True
    assert pseudocount == 50.0

def test_get_log_count_params_coupled_mse():
    """CoupledMSEMultinomialLoss inherits uses_count_pseudocount=True."""
    conf = _model_config_with_loss(
        "cerberus.loss.CoupledMSEMultinomialLoss",
        count_pseudocount=25.0,
    )
    includes, pseudocount = get_log_count_params(conf)
    assert includes is True
    assert pseudocount == 25.0

def test_get_log_count_params_poisson():
    """Poisson loss returns uses_pseudocount=False and pseudocount=0.0."""
    conf = _model_config_with_loss(
        "cerberus.loss.PoissonMultinomialLoss",
        count_pseudocount=99.0,  # present but ignored by get_log_count_params
    )
    includes, pseudocount = get_log_count_params(conf)
    assert includes is False
    assert pseudocount == 0.0

def test_get_log_count_params_dalmatian():
    """DalmatianLoss has uses_count_pseudocount=True."""
    conf = _model_config_with_loss(
        "cerberus.loss.DalmatianLoss",
        loss_args={"base_loss_cls": "cerberus.loss.MSEMultinomialLoss"},
        count_pseudocount=100.0,
    )
    includes, pseudocount = get_log_count_params(conf)
    assert includes is True
    assert pseudocount == 100.0

def test_get_log_count_params_negative_binomial():
    """NegativeBinomialMultinomialLoss inherits uses_count_pseudocount=False."""
    conf = _model_config_with_loss(
        "cerberus.loss.NegativeBinomialMultinomialLoss",
        count_pseudocount=1.0,
    )
    includes, pseudocount = get_log_count_params(conf)
    assert includes is False
    assert pseudocount == 0.0
