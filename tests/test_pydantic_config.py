"""
Comprehensive regression tests for the Pydantic config system.

Covers:
1. Model construction — valid data, attribute access, frozen immutability
2. Validation — missing fields, invalid types, negative values, extra keys
3. Sampler args — plain dict sampler_args on SamplerConfig
4. Pseudocount on ModelConfig — first-class field, default, negative
5. Serialization round-trip — model_dump, model_validate, YAML
6. CerberusConfig cross-validation — padded_size, channel mismatch
7. model_copy — immutable updates, nested
8. Backward compatibility — parse_hparams_config legacy migration
9. model_config_ alias on CerberusConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    GenomeConfig,
    ModelConfig,
    PretrainedConfig,
    SamplerConfig,
    TrainConfig,
    parse_hparams_config,
)


# ---------------------------------------------------------------------------
# Helpers — construct configs that skip file-existence validators
# ---------------------------------------------------------------------------


def _fold_args(**overrides: Any) -> dict[str, Any]:
    kw: dict[str, Any] = dict(k=5, test_fold=None, val_fold=None)
    kw.update(overrides)
    return kw


def _genome_config(**overrides: Any) -> GenomeConfig:
    kw: dict[str, Any] = dict(
        name="hg38",
        fasta_path=Path("/fake/genome.fa"),
        exclude_intervals={"bl": Path("/fake/bl.bed")},
        allowed_chroms=["chr1"],
        chrom_sizes={"chr1": 1_000_000},
        fold_type="chrom_partition",
        fold_args=_fold_args(),
    )
    kw.update(overrides)
    return GenomeConfig.model_construct(_fields_set=None, **kw)


def _data_config(**overrides: Any) -> DataConfig:
    kw: dict[str, Any] = dict(
        inputs={"seq": Path("/fake/input.bw")},
        targets={"out": Path("/fake/target.bw")},
        input_len=1000,
        output_len=500,
        max_jitter=50,
        output_bin_size=1,
        encoding="ACGT",
        log_transform=False,
        reverse_complement=False,
        use_sequence=True,
        target_scale=1.0,
    )
    kw.update(overrides)
    return DataConfig.model_construct(_fields_set=None, **kw)


def _train_config(**overrides: Any) -> TrainConfig:
    kw: dict[str, Any] = dict(
        batch_size=32,
        max_epochs=100,
        learning_rate=0.001,
        weight_decay=0.01,
        patience=5,
        optimizer="adamw",
        scheduler_type="default",
        scheduler_args={},
        filter_bias_and_bn=True,
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )
    kw.update(overrides)
    return TrainConfig.model_construct(_fields_set=None, **kw)


def _model_config(**overrides: Any) -> ModelConfig:
    kw: dict[str, Any] = dict(
        name="test_model",
        model_cls="cerberus.models.bpnet.BPNet",
        loss_cls="cerberus.models.bpnet.BPNetLoss",
        loss_args={},
        metrics_cls="cerberus.models.bpnet.BPNetMetricCollection",
        metrics_args={},
        model_args={
            "input_channels": ["seq"],
            "output_channels": ["out"],
            "output_type": "signal",
        },
        pretrained=[],
        count_pseudocount=0.0,
    )
    kw.update(overrides)
    return ModelConfig.model_construct(_fields_set=None, **kw)


def _sampler_config(**overrides: Any) -> SamplerConfig:
    kw: dict[str, Any] = dict(
        sampler_type="random",
        padded_size=2048,
        sampler_args={"num_intervals": 100},
    )
    kw.update(overrides)
    return SamplerConfig.model_construct(_fields_set=None, **kw)


def _cerberus_config(**overrides: Any) -> CerberusConfig:
    kw: dict[str, Any] = dict(
        train_config=_train_config(),
        genome_config=_genome_config(),
        data_config=_data_config(),
        sampler_config=_sampler_config(),
        model_config_=_model_config(),
    )
    kw.update(overrides)
    return CerberusConfig.model_construct(_fields_set=None, **kw)


def _full_config_dict(tmp_path: Path) -> dict[str, Any]:
    """Return a dict suitable for CerberusConfig.model_validate with real tmp files."""
    (tmp_path / "genome.fa").touch()
    (tmp_path / "exclude.bed").touch()
    (tmp_path / "input.bw").touch()
    (tmp_path / "target.bw").touch()
    (tmp_path / "peaks.bed").touch()

    return {
        "train_config": {
            "batch_size": 32,
            "max_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "patience": 5,
            "optimizer": "adamw",
            "scheduler_type": "default",
            "scheduler_args": {},
            "filter_bias_and_bn": True,
            "reload_dataloaders_every_n_epochs": 0,
            "adam_eps": 1e-8,
            "gradient_clip_val": None,
        },
        "genome_config": {
            "name": "hg38",
            "fasta_path": str(tmp_path / "genome.fa"),
            "exclude_intervals": {"bl": str(tmp_path / "exclude.bed")},
            "allowed_chroms": ["chr1"],
            "chrom_sizes": {"chr1": 1_000_000},
            "fold_type": "chrom_partition",
            "fold_args": {"k": 5},
        },
        "data_config": {
            "inputs": {"seq": str(tmp_path / "input.bw")},
            "targets": {"out": str(tmp_path / "target.bw")},
            "input_len": 1000,
            "output_len": 500,
            "max_jitter": 50,
            "output_bin_size": 1,
            "encoding": "ACGT",
            "log_transform": False,
            "reverse_complement": False,
            "use_sequence": True,
            "target_scale": 1.0,
        },
        "sampler_config": {
            "sampler_type": "interval",
            "padded_size": 2048,
            "sampler_args": {"intervals_path": str(tmp_path / "peaks.bed")},
        },
        "model_config": {
            "name": "test_model",
            "model_cls": "cerberus.models.bpnet.BPNet",
            "loss_cls": "cerberus.models.bpnet.BPNetLoss",
            "loss_args": {},
            "metrics_cls": "cerberus.models.bpnet.BPNetMetricCollection",
            "metrics_args": {},
            "model_args": {
                "input_channels": ["seq"],
                "output_channels": ["out"],
                "output_type": "signal",
            },
            "pretrained": [],
            "count_pseudocount": 0.0,
        },
    }


# ===========================================================================
# 1. Model Construction
# ===========================================================================


class TestModelConstruction:
    """Construct each config type with valid data and verify basics."""

    def test_fold_args_construction(self):
        fa = {"k": 5, "test_fold": 0, "val_fold": 1}
        assert fa["k"] == 5
        assert fa["test_fold"] == 0
        assert fa["val_fold"] == 1

    def test_train_config_construction(self):
        tc = _train_config()
        assert tc.batch_size == 32
        assert tc.max_epochs == 100
        assert tc.learning_rate == 0.001
        assert tc.optimizer == "adamw"

    def test_genome_config_construction(self):
        gc = _genome_config()
        assert gc.name == "hg38"
        assert gc.fasta_path == Path("/fake/genome.fa")
        assert gc.allowed_chroms == ["chr1"]

    def test_data_config_construction(self):
        dc = _data_config()
        assert dc.input_len == 1000
        assert dc.output_len == 500
        assert dc.encoding == "ACGT"
        assert dc.target_scale == 1.0

    def test_sampler_config_construction(self):
        sc = _sampler_config()
        assert sc.sampler_type == "random"
        assert sc.padded_size == 2048

    def test_model_config_construction(self):
        mc = _model_config()
        assert mc.name == "test_model"
        assert mc.model_cls == "cerberus.models.bpnet.BPNet"
        assert mc.pretrained == []
        assert mc.count_pseudocount == 0.0

    def test_pretrained_config_construction(self):
        pc = PretrainedConfig(
            weights_path="/fake/weights.pt",
            source=None,
            target=None,
            freeze=False,
        )
        assert pc.weights_path == "/fake/weights.pt"
        assert pc.source is None
        assert pc.freeze is False

    def test_cerberus_config_construction(self):
        cc = _cerberus_config()
        assert cc.train_config.batch_size == 32
        assert cc.data_config.input_len == 1000
        assert cc.model_config_.name == "test_model"

    def test_frozen_train_config_raises_on_assignment(self):
        tc = TrainConfig(
            batch_size=32, max_epochs=100, learning_rate=0.001,
            weight_decay=0.01, patience=5, optimizer="adamw",
            scheduler_type="default", scheduler_args={},
            filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8, gradient_clip_val=None,
        )
        with pytest.raises(ValidationError):
            tc.batch_size = 64  # type: ignore[misc]

    def test_frozen_model_config_raises_on_assignment(self):
        mc = _model_config()
        with pytest.raises(ValidationError):
            mc.name = "other"  # type: ignore[misc]

    def test_frozen_sampler_config_raises_on_assignment(self):
        sc = _sampler_config()
        with pytest.raises(ValidationError):
            sc.padded_size = 4096  # type: ignore[misc]


# ===========================================================================
# 2. Validation
# ===========================================================================


class TestValidation:
    """Missing required fields, invalid types, negative values, extra keys."""

    def test_train_config_missing_required_raises(self):
        with pytest.raises(ValidationError):
            TrainConfig(batch_size=32)  # type: ignore[call-arg]

    def test_train_config_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            TrainConfig(
                batch_size="not_an_int",  # type: ignore[arg-type]
                max_epochs=100, learning_rate=0.001,
                weight_decay=0.01, patience=5, optimizer="adamw",
                scheduler_type="default", scheduler_args={},
                filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0,
                adam_eps=1e-8, gradient_clip_val=None,
            )

    def test_train_config_negative_batch_size_raises(self):
        with pytest.raises(ValidationError):
            TrainConfig(
                batch_size=-1, max_epochs=100, learning_rate=0.001,
                weight_decay=0.01, patience=5, optimizer="adamw",
                scheduler_type="default", scheduler_args={},
                filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0,
                adam_eps=1e-8, gradient_clip_val=None,
            )

    def test_train_config_zero_learning_rate_raises(self):
        with pytest.raises(ValidationError):
            TrainConfig(
                batch_size=32, max_epochs=100, learning_rate=0.0,
                weight_decay=0.01, patience=5, optimizer="adamw",
                scheduler_type="default", scheduler_args={},
                filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0,
                adam_eps=1e-8, gradient_clip_val=None,
            )

    def test_train_config_extra_key_raises(self):
        with pytest.raises(ValidationError):
            TrainConfig(
                batch_size=32, max_epochs=100, learning_rate=0.001,
                weight_decay=0.01, patience=5, optimizer="adamw",
                scheduler_type="default", scheduler_args={},
                filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0,
                adam_eps=1e-8, gradient_clip_val=None,
                unknown_key="oops",  # type: ignore[call-arg]
            )

    def test_sampler_config_negative_padded_size_raises(self):
        with pytest.raises(ValidationError):
            SamplerConfig(
                sampler_type="random",
                padded_size=-100,
                sampler_args={"num_intervals": 10},
            )

    def test_sampler_config_zero_padded_size_raises(self):
        with pytest.raises(ValidationError):
            SamplerConfig(
                sampler_type="random",
                padded_size=0,
                sampler_args={"num_intervals": 10},
            )

    def test_data_config_negative_input_len_raises(self, mock_files):
        with pytest.raises(ValidationError):
            DataConfig(
                inputs={"seq": mock_files["input"]},
                targets={"out": mock_files["target"]},
                input_len=-100,
                output_len=500,
                max_jitter=0,
                output_bin_size=1,
                encoding="ACGT",
                log_transform=False,
                reverse_complement=False,
                use_sequence=True,
                target_scale=1.0,
            )

    def test_data_config_negative_max_jitter_raises(self, mock_files):
        with pytest.raises(ValidationError):
            DataConfig(
                inputs={"seq": mock_files["input"]},
                targets={"out": mock_files["target"]},
                input_len=100,
                output_len=50,
                max_jitter=-10,
                output_bin_size=1,
                encoding="ACGT",
                log_transform=False,
                reverse_complement=False,
                use_sequence=True,
                target_scale=1.0,
            )

    def test_pretrained_config_extra_key_raises(self):
        with pytest.raises(ValidationError):
            PretrainedConfig(
                weights_path="/fake/w.pt",
                source=None,
                target=None,
                freeze=False,
                extra_field="bad",  # type: ignore[call-arg]
            )

    def test_model_config_extra_key_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(
                name="m",
                model_cls="a.B",
                loss_cls="a.C",
                loss_args={},
                metrics_cls="a.D",
                metrics_args={},
                model_args={},
                extra_field=True,  # type: ignore[call-arg]
            )


# ===========================================================================
# 3. Sampler Args (plain dict)
# ===========================================================================


class TestSamplerArgs:
    """sampler_args is now a plain dict[str, Any] on SamplerConfig."""

    def test_random_sampler_args_constructs(self):
        sc = SamplerConfig(
            sampler_type="random",
            padded_size=1024,
            sampler_args={"num_intervals": 50},
        )
        assert isinstance(sc.sampler_args, dict)
        assert sc.sampler_args["num_intervals"] == 50

    def test_sliding_window_args_constructs(self):
        sc = SamplerConfig(
            sampler_type="sliding_window",
            padded_size=1024,
            sampler_args={"stride": 256},
        )
        assert isinstance(sc.sampler_args, dict)
        assert sc.sampler_args["stride"] == 256

    def test_interval_sampler_args_from_dict(self, tmp_path):
        """sampler_args as a plain dict with intervals_path."""
        bed = tmp_path / "intervals.bed"
        bed.touch()
        sc = SamplerConfig(
            sampler_type="interval",
            padded_size=1024,
            sampler_args={"intervals_path": str(bed)},
        )
        assert isinstance(sc.sampler_args, dict)

    def test_peak_sampler_args_from_dict(self, tmp_path):
        bed = tmp_path / "peaks.bed"
        bed.touch()
        sc = SamplerConfig(
            sampler_type="peak",
            padded_size=2048,
            sampler_args={"intervals_path": str(bed)},
        )
        assert isinstance(sc.sampler_args, dict)

    def test_negative_peak_sampler_args_from_dict(self, tmp_path):
        bed = tmp_path / "neg_peaks.bed"
        bed.touch()
        sc = SamplerConfig(
            sampler_type="negative_peak",
            padded_size=2048,
            sampler_args={"intervals_path": str(bed)},
        )
        assert isinstance(sc.sampler_args, dict)

    def test_random_sampler_args_from_dict(self):
        """sampler_args as a plain dict for 'random' type."""
        sc = SamplerConfig(
            sampler_type="random",
            padded_size=1024,
            sampler_args={"num_intervals": 100},
        )
        assert isinstance(sc.sampler_args, dict)
        assert sc.sampler_args["num_intervals"] == 100

    def test_sliding_window_args_from_dict(self):
        sc = SamplerConfig(
            sampler_type="sliding_window",
            padded_size=1024,
            sampler_args={"stride": 128},
        )
        assert isinstance(sc.sampler_args, dict)

    def test_complexity_matched_args_constructs(self):
        target = SamplerConfig.model_construct(
            _fields_set=None,
            sampler_type="random",
            padded_size=1024,
            sampler_args={"num_intervals": 50},
        )
        candidate = SamplerConfig.model_construct(
            _fields_set=None,
            sampler_type="random",
            padded_size=1024,
            sampler_args={"num_intervals": 200},
        )
        sc = SamplerConfig(
            sampler_type="complexity_matched",
            padded_size=1024,
            sampler_args={
                "target_sampler": target,
                "candidate_sampler": candidate,
                "bins": 20,
                "candidate_ratio": 5.0,
                "metrics": ["gc_content"],
            },
        )
        assert isinstance(sc.sampler_args, dict)
        assert sc.sampler_args["bins"] == 20


# ===========================================================================
# 4. Pseudocount on ModelConfig
# ===========================================================================


class TestModelConfigPseudocount:
    """count_pseudocount as a first-class field on ModelConfig."""

    def test_is_first_class_field(self):
        mc = _model_config()
        assert hasattr(mc, "count_pseudocount")

    def test_default_is_zero(self):
        mc = ModelConfig(
            name="m",
            model_cls="a.B",
            loss_cls="a.C",
            loss_args={},
            metrics_cls="a.D",
            metrics_args={},
            model_args={},
        )
        assert mc.count_pseudocount == 0.0

    def test_custom_value(self):
        mc = ModelConfig(
            name="m",
            model_cls="a.B",
            loss_cls="a.C",
            loss_args={},
            metrics_cls="a.D",
            metrics_args={},
            model_args={},
            count_pseudocount=100.0,
        )
        assert mc.count_pseudocount == 100.0

    def test_negative_value_raises(self):
        with pytest.raises(ValidationError):
            ModelConfig(
                name="m",
                model_cls="a.B",
                loss_cls="a.C",
                loss_args={},
                metrics_cls="a.D",
                metrics_args={},
                model_args={},
                count_pseudocount=-1.0,
            )


# ===========================================================================
# 5. Serialization Round-Trip
# ===========================================================================


class TestSerializationRoundTrip:
    """model_dump, model_validate, YAML round-trip."""

    def test_model_dump_converts_path_to_str_json_mode(self):
        dc = _data_config()
        dumped = dc.model_dump(mode="json")
        assert isinstance(dumped["inputs"]["seq"], str)
        assert isinstance(dumped["targets"]["out"], str)

    def test_train_config_round_trip(self):
        tc = TrainConfig(
            batch_size=32, max_epochs=100, learning_rate=0.001,
            weight_decay=0.01, patience=5, optimizer="adamw",
            scheduler_type="default", scheduler_args={},
            filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8, gradient_clip_val=None,
        )
        dumped = tc.model_dump()
        restored = TrainConfig.model_validate(dumped)
        assert restored == tc

    def test_model_config_round_trip(self):
        mc = ModelConfig(
            name="m",
            model_cls="a.B",
            loss_cls="a.C",
            loss_args={"alpha": 0.5},
            metrics_cls="a.D",
            metrics_args={},
            model_args={"layers": 8},
            count_pseudocount=42.0,
        )
        dumped = mc.model_dump()
        restored = ModelConfig.model_validate(dumped)
        assert restored == mc

    def test_pretrained_config_round_trip(self):
        pc = PretrainedConfig(
            weights_path="/w.pt", source="encoder", target="encoder", freeze=True
        )
        dumped = pc.model_dump()
        restored = PretrainedConfig.model_validate(dumped)
        assert restored == pc

    def test_cerberus_config_round_trip(self, tmp_path):
        """Full CerberusConfig: dump -> validate round-trip."""
        d = _full_config_dict(tmp_path)
        config = CerberusConfig.model_validate(d)
        dumped = config.model_dump()
        restored = CerberusConfig.model_validate(dumped)
        assert restored.train_config == config.train_config
        assert restored.model_config_.name == config.model_config_.name
        assert restored.data_config.input_len == config.data_config.input_len

    def test_yaml_round_trip(self, tmp_path):
        """YAML dump -> load -> model_validate."""
        d = _full_config_dict(tmp_path)
        config = CerberusConfig.model_validate(d)
        dumped = config.model_dump(mode="json")

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dumped, f)

        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)

        restored = CerberusConfig.model_validate(loaded)
        assert restored.train_config.batch_size == config.train_config.batch_size
        assert restored.model_config_.count_pseudocount == config.model_config_.count_pseudocount


# ===========================================================================
# 6. CerberusConfig Cross-Validation
# ===========================================================================


class TestCerberusConfigCrossValidation:
    """padded_size vs input_len+jitter, channel mismatch, valid config."""

    def test_valid_config_constructs(self, tmp_path):
        d = _full_config_dict(tmp_path)
        config = CerberusConfig.model_validate(d)
        assert isinstance(config, CerberusConfig)

    def test_padded_size_too_small_raises(self, tmp_path):
        """padded_size < input_len + 2*max_jitter should raise."""
        d = _full_config_dict(tmp_path)
        # input_len=1000, max_jitter=50, required=1100; set padded_size=500
        d["sampler_config"]["padded_size"] = 500
        with pytest.raises(ValidationError, match="padded_size"):
            CerberusConfig.model_validate(d)

    def test_output_channel_mismatch_raises(self, tmp_path):
        """Model output channels != data target channels should raise."""
        d = _full_config_dict(tmp_path)
        d["model_config"]["model_args"]["output_channels"] = ["wrong_channel"]
        with pytest.raises(ValidationError, match="output channels"):
            CerberusConfig.model_validate(d)

    def test_input_channel_mismatch_raises(self, tmp_path):
        """Data input channels not a subset of model input channels should raise."""
        d = _full_config_dict(tmp_path)
        d["model_config"]["model_args"]["input_channels"] = ["other"]
        with pytest.raises(ValidationError, match="input"):
            CerberusConfig.model_validate(d)

    def test_exact_padded_size_is_valid(self, tmp_path):
        """padded_size == input_len + 2*max_jitter is just enough."""
        d = _full_config_dict(tmp_path)
        # input_len=1000, max_jitter=50 -> required=1100
        d["sampler_config"]["padded_size"] = 1100
        config = CerberusConfig.model_validate(d)
        assert config.sampler_config.padded_size == 1100


# ===========================================================================
# 7. model_copy (Immutable Updates)
# ===========================================================================


class TestModelCopy:
    """model_copy produces new instances; originals unchanged."""

    def test_fold_args_copy_update(self):
        fa = {"k": 5, "test_fold": 0, "val_fold": 1}
        fa2 = {**fa, "k": 10}
        assert fa2["k"] == 10
        assert fa["k"] == 5  # original unchanged

    def test_train_config_copy_update(self):
        tc = TrainConfig(
            batch_size=32, max_epochs=100, learning_rate=0.001,
            weight_decay=0.01, patience=5, optimizer="adamw",
            scheduler_type="default", scheduler_args={},
            filter_bias_and_bn=True, reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8, gradient_clip_val=None,
        )
        tc2 = tc.model_copy(update={"batch_size": 64})
        assert tc2.batch_size == 64
        assert tc.batch_size == 32

    def test_model_config_copy_update_pseudocount(self):
        mc = ModelConfig(
            name="m",
            model_cls="a.B",
            loss_cls="a.C",
            loss_args={},
            metrics_cls="a.D",
            metrics_args={},
            model_args={},
            count_pseudocount=0.0,
        )
        mc2 = mc.model_copy(update={"count_pseudocount": 50.0})
        assert mc2.count_pseudocount == 50.0
        assert mc.count_pseudocount == 0.0

    def test_nested_fold_args_copy(self):
        gc = _genome_config()
        new_fa = {"k": 10, "test_fold": 2, "val_fold": 3}
        gc2 = gc.model_copy(update={"fold_args": new_fa})
        assert gc2.fold_args["k"] == 10
        assert gc.fold_args["k"] == 5  # original unchanged

    def test_cerberus_config_copy_update_sub_config(self):
        cc = _cerberus_config()
        new_tc = _train_config(batch_size=128)
        cc2 = cc.model_copy(update={"train_config": new_tc})
        assert cc2.train_config.batch_size == 128
        assert cc.train_config.batch_size == 32


# ===========================================================================
# 8. Backward Compatibility (parse_hparams_config)
# ===========================================================================


class TestBackwardCompatibility:
    """Legacy config migration in parse_hparams_config."""

    def _write_hparams(self, tmp_path, config_dict):
        hparams_path = tmp_path / "hparams.yaml"
        with open(hparams_path, "w") as f:
            yaml.dump(config_dict, f)
        return hparams_path

    def test_legacy_count_pseudocount_migrates_to_model_config(self, tmp_path):
        """count_pseudocount in data_config migrates to model_config."""
        d = _full_config_dict(tmp_path)
        # Add legacy pseudocount to data_config
        d["data_config"]["count_pseudocount"] = 50.0
        # Remove from model_config to trigger migration
        del d["model_config"]["count_pseudocount"]

        hparams_path = self._write_hparams(tmp_path, d)
        config = parse_hparams_config(hparams_path)
        # 50.0 * target_scale(1.0) = 50.0
        assert config.model_config_.count_pseudocount == pytest.approx(50.0)

    def test_legacy_pseudocount_scaled_by_target_scale(self, tmp_path):
        """Legacy pseudocount is multiplied by target_scale during migration."""
        d = _full_config_dict(tmp_path)
        d["data_config"]["count_pseudocount"] = 50.0
        d["data_config"]["target_scale"] = 2.0
        del d["model_config"]["count_pseudocount"]

        hparams_path = self._write_hparams(tmp_path, d)
        config = parse_hparams_config(hparams_path)
        # 50.0 * 2.0 = 100.0
        assert config.model_config_.count_pseudocount == pytest.approx(100.0)

    def test_missing_pretrained_defaults_to_empty_list(self, tmp_path):
        """Old YAML without 'pretrained' in model_config gets [] backfilled."""
        d = _full_config_dict(tmp_path)
        del d["model_config"]["pretrained"]

        hparams_path = self._write_hparams(tmp_path, d)
        config = parse_hparams_config(hparams_path)
        assert config.model_config_.pretrained == []

    def test_extra_lightning_keys_ignored(self, tmp_path):
        """Extra top-level keys (e.g. from Lightning) are silently ignored."""
        d = _full_config_dict(tmp_path)
        d["lr_find_results"] = {"lr": 0.001}
        d["callbacks"] = {"checkpoint": {}}

        hparams_path = self._write_hparams(tmp_path, d)
        config = parse_hparams_config(hparams_path)
        assert isinstance(config, CerberusConfig)

    def test_missing_required_section_raises(self, tmp_path):
        """Missing required top-level key raises ValueError."""
        d = _full_config_dict(tmp_path)
        del d["train_config"]

        hparams_path = self._write_hparams(tmp_path, d)
        with pytest.raises(ValueError, match="missing required"):
            parse_hparams_config(hparams_path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_hparams_config("/nonexistent/hparams.yaml")


# ===========================================================================
# 9. model_config_ alias on CerberusConfig
# ===========================================================================


class TestModelConfigAlias:
    """model_config_ is the Python attribute; 'model_config' is the serialization key."""

    def test_accessible_via_attribute(self):
        cc = _cerberus_config()
        assert cc.model_config_.name == "test_model"

    def test_serializes_as_model_config_with_by_alias(self):
        cc = _cerberus_config()
        dumped = cc.model_dump(by_alias=True)
        assert "model_config" in dumped
        assert "model_config_" not in dumped

    def test_deserializes_from_model_config_key(self, tmp_path):
        """CerberusConfig.model_validate accepts 'model_config' key in dict."""
        d = _full_config_dict(tmp_path)
        assert "model_config" in d
        config = CerberusConfig.model_validate(d)
        assert config.model_config_.name == "test_model"

    def test_by_alias_round_trip(self, tmp_path):
        """Dump with by_alias=True, then validate from the dumped dict."""
        d = _full_config_dict(tmp_path)
        config = CerberusConfig.model_validate(d)
        dumped = config.model_dump(by_alias=True)
        # model_dump(by_alias=True) uses "model_config" in the dict
        restored = CerberusConfig.model_validate(dumped)
        assert restored.model_config_.name == config.model_config_.name

    def test_populate_by_name_allows_model_config_(self, tmp_path):
        """populate_by_name=True allows using 'model_config_' key as well."""
        d = _full_config_dict(tmp_path)
        # Rename model_config -> model_config_ (the field name)
        d["model_config_"] = d.pop("model_config")
        config = CerberusConfig.model_validate(d)
        assert config.model_config_.name == "test_model"
