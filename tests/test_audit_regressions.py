"""Regression tests for code quality audit issues #2 and #3."""

from unittest.mock import MagicMock

import pytest
import torch

from cerberus.loss import (
    CoupledNegativeBinomialMultinomialLoss,
    CoupledPoissonMultinomialLoss,
    NegativeBinomialMultinomialLoss,
    PoissonMultinomialLoss,
)
from cerberus.output import ProfileCountOutput, ProfileLogRates
from cerberus.signal import UniversalExtractor

# ---------------------------------------------------------------------------
# Issue #2: .bed.gz suffix detection
# ---------------------------------------------------------------------------

class TestUniversalExtractorRouting:
    """Verify that UniversalExtractor routes files by extension correctly via the registry."""

    def _get_channel_cls_names(self, paths: dict[str, str]) -> dict[str, str]:
        """Instantiate UniversalExtractor with mocked extractors and return cls name per channel."""
        mock_bw = MagicMock()
        mock_bw.__name__ = "SignalExtractor"
        mock_bb = MagicMock()
        mock_bb.__name__ = "BigBedMaskExtractor"
        mock_bed = MagicMock()
        mock_bed.__name__ = "BedMaskExtractor"
        mock_bw_mem = MagicMock()
        mock_bw_mem.__name__ = "InMemorySignalExtractor"
        mock_bb_mem = MagicMock()
        mock_bb_mem.__name__ = "InMemoryBigBedMaskExtractor"

        from cerberus.signal import _EXTRACTOR_REGISTRY
        saved = dict(_EXTRACTOR_REGISTRY)
        try:
            _EXTRACTOR_REGISTRY.clear()
            _EXTRACTOR_REGISTRY['.bw'] = (mock_bw, mock_bw_mem)
            _EXTRACTOR_REGISTRY['.bigwig'] = (mock_bw, mock_bw_mem)
            _EXTRACTOR_REGISTRY['.bb'] = (mock_bb, mock_bb_mem)
            _EXTRACTOR_REGISTRY['.bigbed'] = (mock_bb, mock_bb_mem)
            _EXTRACTOR_REGISTRY['.bed'] = (mock_bed, None)
            _EXTRACTOR_REGISTRY['.bed.gz'] = (mock_bed, None)
            ext = UniversalExtractor(paths)  # type: ignore[arg-type]
        finally:
            _EXTRACTOR_REGISTRY.clear()
            _EXTRACTOR_REGISTRY.update(saved)
        return {name: ext._channel_to_cls[name].__name__ for name in ext.channels}

    def test_bed_gz_routed_to_bed(self):
        """Regression: .bed.gz files must go to the BED bucket, not BigWig."""
        routing = self._get_channel_cls_names({"peaks": "/data/peaks.bed.gz"})
        assert routing["peaks"] == "BedMaskExtractor"

    def test_plain_bed_routed_to_bed(self):
        routing = self._get_channel_cls_names({"peaks": "/data/peaks.bed"})
        assert routing["peaks"] == "BedMaskExtractor"

    def test_bigwig_routed_to_bw(self):
        routing = self._get_channel_cls_names({"signal": "/data/signal.bw"})
        assert routing["signal"] == "SignalExtractor"

    def test_bigbed_routed_to_bb(self):
        routing = self._get_channel_cls_names({"mask": "/data/mask.bb"})
        assert routing["mask"] == "BigBedMaskExtractor"

    def test_non_bed_gz_not_routed_to_bed(self):
        """Regression: arbitrary .gz files must NOT be misrouted to BED."""
        # .gz without .bed prefix should raise since no extractor is registered for .gz
        with pytest.raises(ValueError, match="No extractor registered"):
            self._get_channel_cls_names({"data": "/data/signal.foo.gz"})

    def test_mixed_routing(self):
        routing = self._get_channel_cls_names({
            "bw_track": "/data/a.bw",
            "bed_track": "/data/b.bed.gz",
            "bb_track": "/data/c.bigbed",
            "plain_bed": "/data/d.bed",
        })
        assert routing["bw_track"] == "SignalExtractor"
        assert routing["bed_track"] == "BedMaskExtractor"
        assert routing["bb_track"] == "BigBedMaskExtractor"
        assert routing["plain_bed"] == "BedMaskExtractor"


# ---------------------------------------------------------------------------
# Issue #3: Poisson/NB losses must accept count_pseudocount kwarg
# ---------------------------------------------------------------------------

class TestLossPseudocountCompat:
    """Regression: all loss classes must accept count_pseudocount without crashing."""

    B, C, L = 2, 3, 64

    def _make_profile_count_inputs(self):
        logits = torch.randn(self.B, self.C, self.L)
        log_counts = torch.randn(self.B, self.C)
        targets = torch.randint(0, 10, (self.B, self.C, self.L)).float()
        outputs = ProfileCountOutput(logits=logits, log_counts=log_counts)
        return outputs, targets

    def _make_log_rates_inputs(self):
        log_rates = torch.randn(self.B, self.C, self.L)
        targets = torch.randint(0, 10, (self.B, self.C, self.L)).float()
        outputs = ProfileLogRates(log_rates=log_rates)
        return outputs, targets

    def test_poisson_accepts_count_pseudocount(self):
        """PoissonMultinomialLoss must not crash when given count_pseudocount."""
        loss_fn = PoissonMultinomialLoss(count_pseudocount=0.5, count_per_channel=True)
        outputs, targets = self._make_profile_count_inputs()
        loss = loss_fn(outputs, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_nb_accepts_count_pseudocount(self):
        """NegativeBinomialMultinomialLoss must not crash when given count_pseudocount."""
        loss_fn = NegativeBinomialMultinomialLoss(
            total_count=10.0, count_pseudocount=0.5, count_per_channel=True,
        )
        outputs, targets = self._make_profile_count_inputs()
        loss = loss_fn(outputs, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_coupled_poisson_accepts_count_pseudocount(self):
        loss_fn = CoupledPoissonMultinomialLoss(count_pseudocount=0.5)
        outputs, targets = self._make_log_rates_inputs()
        loss = loss_fn(outputs, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_coupled_nb_accepts_count_pseudocount(self):
        loss_fn = CoupledNegativeBinomialMultinomialLoss(
            total_count=10.0, count_pseudocount=0.5,
        )
        outputs, targets = self._make_log_rates_inputs()
        loss = loss_fn(outputs, targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_propagate_pseudocount_all_loss_classes(self):
        """Simulate what propagate_pseudocount does: inject count_pseudocount into kwargs."""
        loss_classes = [
            PoissonMultinomialLoss,
            NegativeBinomialMultinomialLoss,
            CoupledPoissonMultinomialLoss,
            CoupledNegativeBinomialMultinomialLoss,
        ]
        kwargs_base = {"count_pseudocount": 2.0}
        for cls in loss_classes:
            extra = {"total_count": 10.0} if "NegativeBinomial" in cls.__name__ else {}
            instance = cls(**kwargs_base, **extra)
            assert instance is not None, f"{cls.__name__} failed to instantiate"
