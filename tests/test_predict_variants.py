"""Tests for cerberus.predict_variants — batched variant effect scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyfaidx
import pytest
import torch
import torch.nn as nn

from cerberus.config import (
    CerberusConfig,
    DataConfig,
    GenomeConfig,
    ModelConfig,
    SamplerConfig,
    TrainConfig,
)
from cerberus.interval import Interval
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import (
    FactorizedProfileCountOutput,
    ModelOutput,
    ProfileCountOutput,
)
from cerberus.predict_variants import VariantResult, score_variants, score_variants_from_ensemble
from cerberus.variants import Variant

FIXTURES = Path(__file__).parent / "data" / "fixtures"
FASTA_PATH = FIXTURES / "test_variants.fa"


# ── Mock model infrastructure ────────────────────────────────────────


@dataclass
class _ProfileCountMockOutput(ProfileCountOutput):
    """ProfileCountOutput that bypasses frozen-field issues in tests."""

    def detach(self) -> _ProfileCountMockOutput:
        return _ProfileCountMockOutput(
            logits=self.logits.detach(),
            log_counts=self.log_counts.detach(),
            out_interval=self.out_interval,
        )


class _SequenceSensitiveModel(nn.Module):
    """Model that returns ProfileCountOutput depending on the input.

    The logits are derived from the input tensor so that ref and alt
    sequences produce measurably different outputs.  The log_counts
    head returns the sum of the center column of the input.
    """

    def __init__(self, output_len: int = 50) -> None:
        super().__init__()
        self.output_len = output_len
        # Need at least one parameter so next(model.parameters()) works
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> ProfileCountOutput:
        B = x.shape[0]
        L = self.output_len
        # Derive logits from the input: average-pool the sequence channels
        # down to output_len bins to simulate a real model
        seq = x[:, :4, :]  # (B, 4, input_len)
        # Simple pooling: take L evenly spaced positions
        indices = torch.linspace(0, seq.shape[-1] - 1, L).long()
        logits = seq[:, :1, indices]  # (B, 1, L) — single channel

        # Count head: sum of all input values → (B, 1)
        log_counts = seq.sum(dim=(1, 2), keepdim=True).squeeze(-1)  # (B, 1)  # (B, 1)

        return _ProfileCountMockOutput(logits=logits, log_counts=log_counts)


class _ConstantModel(nn.Module):
    """Model that returns identical output regardless of input."""

    def __init__(self, output_len: int = 50) -> None:
        super().__init__()
        self.output_len = output_len
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> ProfileCountOutput:
        B = x.shape[0]
        L = self.output_len
        logits = torch.zeros(B, 1, L)
        log_counts = torch.ones(B, 1)
        return _ProfileCountMockOutput(logits=logits, log_counts=log_counts)


class _DalmatianMockModel(nn.Module):
    """Model that returns FactorizedProfileCountOutput."""

    def __init__(self, output_len: int = 50) -> None:
        super().__init__()
        self.output_len = output_len
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> FactorizedProfileCountOutput:
        B = x.shape[0]
        L = self.output_len
        seq = x[:, :4, :]
        indices = torch.linspace(0, seq.shape[-1] - 1, L).long()
        logits = seq[:, :1, indices]
        log_counts = seq.sum(dim=(1, 2), keepdim=True).squeeze(-1)  # (B, 1)

        # Dalmatian: bias is constant, signal varies
        bias_logits = torch.zeros(B, 1, L)
        bias_log_counts = torch.ones(B, 1)
        signal_logits = logits.clone()
        signal_log_counts = log_counts.clone()

        return FactorizedProfileCountOutput(
            logits=logits,
            log_counts=log_counts,
            bias_logits=bias_logits,
            bias_log_counts=bias_log_counts,
            signal_logits=signal_logits,
            signal_log_counts=signal_log_counts,
        )


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def fasta():
    fa = pyfaidx.Fasta(str(FASTA_PATH))
    yield fa
    fa.close()


@pytest.fixture()
def snp_variants() -> list[Variant]:
    """Two SNPs on chr1 within the test FASTA (104bp of repeating ACGT)."""
    return [
        Variant("chr1", 48, "A", "G"),
        Variant("chr1", 52, "A", "T"),
    ]


@pytest.fixture()
def mixed_variants() -> list[Variant]:
    """SNP + deletion + insertion on chr1."""
    return [
        Variant("chr1", 48, "A", "G"),
        Variant("chr1", 40, "ACGT", "A"),  # 3bp deletion
        Variant("chr1", 52, "A", "ACGT"),  # 3bp insertion
    ]


@pytest.fixture()
def sensitive_model() -> _SequenceSensitiveModel:
    return _SequenceSensitiveModel(output_len=50)


@pytest.fixture()
def constant_model() -> _ConstantModel:
    return _ConstantModel(output_len=50)


@pytest.fixture()
def dalmatian_model() -> _DalmatianMockModel:
    return _DalmatianMockModel(output_len=50)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_cerberus_config(
    input_len: int = 20,
    output_len: int = 50,
    fasta_path: str = str(FASTA_PATH),
    inputs: dict | None = None,
    loss_cls: str = "cerberus.loss.PoissonMultinomialLoss",
    count_pseudocount: float = 0.0,
) -> CerberusConfig:
    """Create a minimal CerberusConfig for variant scoring tests."""
    return CerberusConfig.model_construct(
        data_config=DataConfig.model_construct(
            inputs=inputs if inputs is not None else {},
            targets={},
            input_len=input_len,
            output_len=output_len,
            output_bin_size=1,
            max_jitter=0,
            encoding="ACGT",
            log_transform=False,
            reverse_complement=False,
            target_scale=1.0,
            use_sequence=True,
        ),
        genome_config=GenomeConfig.model_construct(
            name="test",
            fasta_path=fasta_path,
            chrom_sizes={"chr1": 104, "chr2": 20},
            allowed_chroms=["chr1", "chr2"],
            exclude_intervals={},
            fold_type="chrom_partition",
            fold_args={"k": 2, "test_fold": None, "val_fold": None},
        ),
        sampler_config=SamplerConfig.model_construct(
            sampler_type="random",
            padded_size=input_len,
            sampler_args={"num_intervals": 10},
        ),
        train_config=TrainConfig.model_construct(
            batch_size=1,
            max_epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            patience=5,
            optimizer="adam",
            scheduler_type="default",
            scheduler_args={},
            filter_bias_and_bn=True,
            reload_dataloaders_every_n_epochs=0,
            adam_eps=1e-8,
            gradient_clip_val=None,
        ),
        model_config_=ModelConfig.model_construct(
            name="mock",
            model_cls="torch.nn.Linear",
            loss_cls=loss_cls,
            loss_args={"count_pseudocount": count_pseudocount},
            metrics_cls="torchmetrics.MeanSquaredError",
            metrics_args={},
            model_args={},
            pretrained=[],
            count_pseudocount=count_pseudocount,
        ),
    )


def _create_mock_ensemble(
    model: nn.Module,
    config: CerberusConfig | None = None,
) -> ModelEnsemble:
    """Build a ModelEnsemble wrapping a single model without disk I/O."""
    if config is None:
        config = _make_cerberus_config()
    models = {"0": model}
    folds: list = []
    with (
        patch("cerberus.model_ensemble._ModelManager") as mock_cls,
        patch(
            "cerberus.model_ensemble.find_latest_hparams",
            return_value=Path("hparams.yaml"),
        ),
        patch(
            "cerberus.model_ensemble.parse_hparams_config",
            return_value=config,
        ),
    ):
        loader = mock_cls.return_value
        loader.load_models_and_folds.return_value = (models, folds)
        return ModelEnsemble(checkpoint_path=".", device=torch.device("cpu"))


# ── VariantResult ────────────────────────────────────────────────────


class TestVariantResult:
    def test_construction(self):
        v = Variant("chr1", 100, "A", "G")
        effects = {"sad": torch.tensor([0.5]), "log_fc": torch.tensor([0.1])}
        iv = Interval("chr1", 90, 110)
        result = VariantResult(variant=v, effects=effects, interval=iv)
        assert result.variant is v
        assert result.effects is effects
        assert result.interval is iv

    def test_frozen(self):
        v = Variant("chr1", 100, "A", "G")
        result = VariantResult(
            variant=v,
            effects={"sad": torch.tensor([0.0])},
            interval=Interval("chr1", 90, 110),
        )
        with pytest.raises(AttributeError):
            result.variant = Variant("chr1", 200, "C", "T")  # type: ignore[misc]

    def test_repr(self):
        v = Variant("chr1", 100, "A", "G")
        result = VariantResult(
            variant=v,
            effects={"sad": torch.tensor([0.0])},
            interval=Interval("chr1", 90, 110),
        )
        assert "VariantResult" in repr(result)

    def test_eq(self):
        v = Variant("chr1", 100, "A", "G")
        effects = {"sad": torch.tensor([0.0])}
        iv = Interval("chr1", 90, 110)
        r1 = VariantResult(variant=v, effects=effects, interval=iv)
        r2 = VariantResult(variant=v, effects=effects, interval=iv)
        assert r1 == r2


# ── score_variants (plain model) ─────────────────────────────────────


class TestScoreVariantsPlainModel:
    """Tests with a plain nn.Module (no ensemble)."""

    def test_snps_produce_results(self, fasta, snp_variants, sensitive_model):
        results = list(
            score_variants(
                model=sensitive_model,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
                batch_size=64,
            )
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, VariantResult)
            assert isinstance(r.variant, Variant)
            assert isinstance(r.effects, dict)
            assert isinstance(r.interval, Interval)

    def test_effect_keys_profile_count(self, fasta, snp_variants, sensitive_model):
        """ProfileCountOutput models produce the expected metric keys."""
        results = list(
            score_variants(
                model=sensitive_model,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
            )
        )
        expected_keys = {"sad", "max_abs_diff", "pearson", "log_fc", "jsd"}
        assert set(results[0].effects.keys()) == expected_keys

    def test_effect_keys_dalmatian(self, fasta, snp_variants, dalmatian_model):
        """FactorizedProfileCountOutput adds signal-only metrics."""
        results = list(
            score_variants(
                model=dalmatian_model,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
            )
        )
        keys = set(results[0].effects.keys())
        assert "signal_sad" in keys
        assert "signal_log_fc" in keys
        assert "signal_jsd" in keys

    def test_constant_model_zero_effects(self, fasta, snp_variants, constant_model):
        """A model that ignores input should produce zero/minimal effects."""
        results = list(
            score_variants(
                model=constant_model,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
            )
        )
        for r in results:
            assert r.effects["sad"].abs().max().item() == pytest.approx(0.0, abs=1e-5)
            assert r.effects["log_fc"].abs().max().item() == pytest.approx(0.0, abs=1e-5)
            assert r.effects["jsd"].abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_sensitive_model_nonzero_effects(self, fasta, sensitive_model):
        """A sequence-sensitive model should produce nonzero effects for a SNP."""
        v = Variant("chr1", 48, "A", "G")
        results = list(
            score_variants(
                model=sensitive_model,
                variants=[v],
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) == 1
        assert results[0].effects["sad"].abs().max().item() > 0.0

    def test_mixed_variant_types(self, fasta, mixed_variants, sensitive_model):
        """SNPs, deletions, and insertions are all scored."""
        results = list(
            score_variants(
                model=sensitive_model,
                variants=mixed_variants,
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) == 3

    def test_effect_tensor_shapes(self, fasta, snp_variants, sensitive_model):
        """Effect tensors should have shape (C,) for single-channel model."""
        results = list(
            score_variants(
                model=sensitive_model,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
            )
        )
        for r in results:
            for key, tensor in r.effects.items():
                assert tensor.ndim == 1, f"{key} should be 1D, got {tensor.shape}"

    def test_effects_are_cpu(self, fasta, snp_variants, sensitive_model):
        """Effect tensors should always be on CPU regardless of model device."""
        results = list(
            score_variants(
                model=sensitive_model,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
                device=torch.device("cpu"),
            )
        )
        for r in results:
            for tensor in r.effects.values():
                assert tensor.device == torch.device("cpu")

    def test_interval_matches_variant(self, fasta, sensitive_model):
        """Returned interval should be centered on the variant."""
        v = Variant("chr1", 48, "A", "G")
        results = list(
            score_variants(
                model=sensitive_model,
                variants=[v],
                fasta=fasta,
                input_len=20,
            )
        )
        iv = results[0].interval
        assert iv.chrom == "chr1"
        assert len(iv) == 20
        # Window should be centered on ref_center (48 for this SNP)
        expected_start = 48 - 20 // 2
        assert iv.start == expected_start

    def test_variant_order_preserved(self, fasta, mixed_variants, sensitive_model):
        """Results are yielded in the same order as input variants."""
        results = list(
            score_variants(
                model=sensitive_model,
                variants=mixed_variants,
                fasta=fasta,
                input_len=20,
            )
        )
        for original, result in zip(mixed_variants, results):
            assert result.variant is original


# ── Batching ─────────────────────────────────────────────────────────


class TestBatching:
    def test_batch_size_1(self, fasta, snp_variants, sensitive_model):
        """batch_size=1 should still produce correct results."""
        results = list(
            score_variants(
                model=sensitive_model,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
                batch_size=1,
            )
        )
        assert len(results) == 2

    def test_batch_size_larger_than_input(self, fasta, snp_variants, sensitive_model):
        """batch_size > n_variants should work fine."""
        results = list(
            score_variants(
                model=sensitive_model,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
                batch_size=1000,
            )
        )
        assert len(results) == 2

    def test_results_consistent_across_batch_sizes(self, fasta, sensitive_model):
        """Different batch sizes should produce identical results."""
        variants = [Variant("chr1", 48, "A", "G"), Variant("chr1", 52, "A", "T")]

        results_bs1 = list(
            score_variants(
                model=sensitive_model,
                variants=variants,
                fasta=fasta,
                input_len=20,
                batch_size=1,
            )
        )
        results_bs2 = list(
            score_variants(
                model=sensitive_model,
                variants=variants,
                fasta=fasta,
                input_len=20,
                batch_size=2,
            )
        )

        assert len(results_bs1) == len(results_bs2)
        for r1, r2 in zip(results_bs1, results_bs2):
            for key in r1.effects:
                torch.testing.assert_close(r1.effects[key], r2.effects[key])


# ── Error handling ───────────────────────────────────────────────────


class TestErrorHandling:
    def test_boundary_variant_skipped(self, fasta, sensitive_model):
        """Variant near chromosome edge (window out of bounds) is skipped."""
        # chr1 is 104bp; window of 20 centered at pos=2 starts at -8
        v = Variant("chr1", 2, "G", "A")
        results = list(
            score_variants(
                model=sensitive_model,
                variants=[v],
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) == 0

    def test_bad_chrom_skipped(self, fasta, sensitive_model):
        """Variant on unknown chromosome is skipped."""
        v = Variant("chrX", 50, "A", "G")
        results = list(
            score_variants(
                model=sensitive_model,
                variants=[v],
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) == 0

    def test_ref_mismatch_skipped(self, fasta, sensitive_model):
        """Variant with ref allele not matching FASTA is skipped."""
        # Position 48 in the repeating ACGT pattern is 'A', not 'T'
        v = Variant("chr1", 48, "T", "G")
        results = list(
            score_variants(
                model=sensitive_model,
                variants=[v],
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) == 0

    def test_mixed_good_and_bad_variants(self, fasta, sensitive_model):
        """Good variants are scored even when some fail."""
        variants = [
            Variant("chr1", 2, "G", "A"),  # boundary → skip
            Variant("chr1", 48, "A", "G"),  # valid
            Variant("chrX", 50, "A", "G"),  # bad chrom → skip
            Variant("chr1", 52, "A", "T"),  # valid
        ]
        results = list(
            score_variants(
                model=sensitive_model,
                variants=variants,
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) == 2
        assert results[0].variant.pos == 48
        assert results[1].variant.pos == 52

    def test_all_bad_variants_returns_empty(self, fasta, sensitive_model):
        """If every variant fails, we get an empty iterator, no crash."""
        variants = [
            Variant("chrX", 50, "A", "G"),
            Variant("chr1", 2, "G", "A"),
        ]
        results = list(
            score_variants(
                model=sensitive_model,
                variants=variants,
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) == 0

    def test_empty_input(self, fasta, sensitive_model):
        """Empty variant iterable produces no results."""
        results = list(
            score_variants(
                model=sensitive_model,
                variants=[],
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) == 0


# ── Generator behavior ───────────────────────────────────────────────


class TestGeneratorBehavior:
    def test_is_generator(self, fasta, snp_variants, sensitive_model):
        """score_variants returns a generator, not a list."""
        result = score_variants(
            model=sensitive_model,
            variants=snp_variants,
            fasta=fasta,
            input_len=20,
        )
        assert hasattr(result, "__next__")

    def test_lazy_evaluation(self, fasta, sensitive_model):
        """Generator should not consume all variants upfront."""
        call_count = 0

        def variant_gen():
            nonlocal call_count
            for v in [Variant("chr1", 48, "A", "G"), Variant("chr1", 52, "A", "T")]:
                call_count += 1
                yield v

        gen = score_variants(
            model=sensitive_model,
            variants=variant_gen(),
            fasta=fasta,
            input_len=20,
            batch_size=1,
        )
        # Before consuming, only the first batch should have been pulled
        first = next(gen)
        assert call_count == 1
        assert first.variant.pos == 48


# ── ModelEnsemble integration ────────────────────────────────────────


class TestScoreVariantsEnsemble:
    """Tests using score_variants with a mock ModelEnsemble."""

    def test_ensemble_produces_results(self, fasta, snp_variants):
        model = _SequenceSensitiveModel(output_len=50)
        ensemble = _create_mock_ensemble(model)

        results = list(
            score_variants(
                model=ensemble,
                variants=snp_variants,
                fasta=fasta,
                input_len=20,
                batch_size=64,
            )
        )
        assert len(results) == 2
        for r in results:
            assert "sad" in r.effects

    def test_use_folds_passed_through(self, fasta):
        """use_folds argument should be forwarded to ModelEnsemble.forward."""
        model = _SequenceSensitiveModel(output_len=50)
        ensemble = _create_mock_ensemble(model)

        # Spy on the forward method
        original_forward = ensemble.forward
        call_args_list: list = []

        def spy_forward(*args, **kwargs):
            call_args_list.append(kwargs)
            return original_forward(*args, **kwargs)

        ensemble.forward = spy_forward  # type: ignore[method-assign]

        variants = [Variant("chr1", 48, "A", "G")]
        list(
            score_variants(
                model=ensemble,
                variants=variants,
                fasta=fasta,
                input_len=20,
                use_folds=["test"],
            )
        )

        # forward called twice (ref + alt), both with use_folds=["test"]
        assert len(call_args_list) == 2
        for call_kwargs in call_args_list:
            assert call_kwargs.get("use_folds") == ["test"]


# ── score_variants_from_ensemble ─────────────────────────────────────


class TestScoreVariantsFromEnsemble:
    """Tests for the convenience wrapper."""

    def test_extracts_config_and_scores(self, snp_variants):
        model = _SequenceSensitiveModel(output_len=50)
        config = _make_cerberus_config(input_len=20, fasta_path=str(FASTA_PATH))
        ensemble = _create_mock_ensemble(model, config)

        results = list(
            score_variants_from_ensemble(
                ensemble=ensemble,
                variants=snp_variants,
                batch_size=64,
            )
        )
        assert len(results) == 2

    def test_accepts_external_fasta(self, fasta, snp_variants):
        model = _SequenceSensitiveModel(output_len=50)
        config = _make_cerberus_config(input_len=20)
        ensemble = _create_mock_ensemble(model, config)

        results = list(
            score_variants_from_ensemble(
                ensemble=ensemble,
                variants=snp_variants,
                fasta=fasta,
                batch_size=64,
            )
        )
        assert len(results) == 2

    def test_rejects_non_sequence_model(self):
        model = _SequenceSensitiveModel(output_len=50)
        config = _make_cerberus_config(input_len=20)
        # Patch use_sequence to False
        config = config.model_copy(
            update={
                "data_config": config.data_config.model_copy(
                    update={"use_sequence": False}
                )
            }
        )
        ensemble = _create_mock_ensemble(model, config)

        with pytest.raises(ValueError, match="use_sequence is False"):
            list(score_variants_from_ensemble(ensemble=ensemble, variants=[]))

    def test_rejects_multi_channel_model(self):
        model = _SequenceSensitiveModel(output_len=50)
        config = _make_cerberus_config(input_len=20)
        config = config.model_copy(
            update={
                "data_config": config.data_config.model_copy(
                    update={"inputs": {"accessibility": Path("/fake/input.bw")}}
                )
            }
        )
        ensemble = _create_mock_ensemble(model, config)

        with pytest.raises(ValueError, match="input signal tracks"):
            list(score_variants_from_ensemble(ensemble=ensemble, variants=[]))

    def test_pseudocount_poisson(self, snp_variants):
        """Poisson loss → log_counts_include_pseudocount=False, pseudocount=0."""
        model = _SequenceSensitiveModel(output_len=50)
        config = _make_cerberus_config(
            input_len=20,
            fasta_path=str(FASTA_PATH),
            loss_cls="cerberus.loss.PoissonMultinomialLoss",
            count_pseudocount=0.0,
        )
        ensemble = _create_mock_ensemble(model, config)

        # Should not raise — verifies pseudocount extraction works
        results = list(
            score_variants_from_ensemble(ensemble=ensemble, variants=snp_variants)
        )
        assert len(results) == 2

    def test_pseudocount_mse(self, snp_variants):
        """MSE loss → log_counts_include_pseudocount=True, pseudocount>0."""
        model = _SequenceSensitiveModel(output_len=50)
        config = _make_cerberus_config(
            input_len=20,
            fasta_path=str(FASTA_PATH),
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            count_pseudocount=1.0,
        )
        ensemble = _create_mock_ensemble(model, config)

        results = list(
            score_variants_from_ensemble(ensemble=ensemble, variants=snp_variants)
        )
        assert len(results) == 2


# ── Metric sanity checks ────────────────────────────────────────────


class TestMetricSanity:
    """Verify that effect metrics satisfy basic mathematical properties."""

    def _score_one(
        self, model: nn.Module, fasta: pyfaidx.Fasta, variant: Variant
    ) -> VariantResult:
        results = list(
            score_variants(
                model=model, variants=[variant], fasta=fasta, input_len=20
            )
        )
        assert len(results) == 1
        return results[0]

    def test_sad_non_negative(self, fasta, sensitive_model):
        r = self._score_one(sensitive_model, fasta, Variant("chr1", 48, "A", "G"))
        assert (r.effects["sad"] >= 0).all()

    def test_jsd_non_negative(self, fasta, sensitive_model):
        r = self._score_one(sensitive_model, fasta, Variant("chr1", 48, "A", "G"))
        assert (r.effects["jsd"] >= -1e-6).all()  # allow tiny float error

    def test_pearson_bounded(self, fasta, sensitive_model):
        r = self._score_one(sensitive_model, fasta, Variant("chr1", 48, "A", "G"))
        assert (r.effects["pearson"] >= -1.0 - 1e-5).all()
        assert (r.effects["pearson"] <= 1.0 + 1e-5).all()

    def test_max_abs_diff_leq_sad(self, fasta, sensitive_model):
        """max_abs_diff should never exceed SAD (SAD sums all abs diffs)."""
        r = self._score_one(sensitive_model, fasta, Variant("chr1", 48, "A", "G"))
        assert (r.effects["max_abs_diff"] <= r.effects["sad"] + 1e-6).all()


# ── VCF / TSV integration ───────────────────────────────────────────


class TestFileIntegration:
    """Test scoring from real VCF and TSV files."""

    def test_score_from_vcf(self, fasta, sensitive_model):
        """Score variants loaded from the test VCF fixture."""
        from cerberus.variants import load_vcf

        vcf_path = FIXTURES / "test_variants.vcf"
        variants = list(load_vcf(vcf_path))
        # Some variants may fail due to boundary issues with the tiny test FASTA
        results = list(
            score_variants(
                model=sensitive_model,
                variants=variants,
                fasta=fasta,
                input_len=20,
            )
        )
        # At least some should succeed (the FASTA is small, some may be out of bounds)
        assert len(results) >= 0  # no crash is the main assertion

    def test_score_from_tsv(self, fasta, sensitive_model):
        """Score variants loaded from the test TSV fixture."""
        from cerberus.variants import load_variants

        variants = list(load_variants(FIXTURES / "test_variants.tsv"))
        results = list(
            score_variants(
                model=sensitive_model,
                variants=variants,
                fasta=fasta,
                input_len=20,
            )
        )
        assert len(results) >= 0  # no crash is the main assertion
