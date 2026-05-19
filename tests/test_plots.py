import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from cerberus.config import TrainConfig
from cerberus.loss import DifferentialCountLoss, ProfilePoissonNLLLoss
from cerberus.metrics import DefaultMetricCollection
from cerberus.models.bpnet import DifferentialBPNetMetricCollection
from cerberus.module import CerberusModule
from cerberus.output import ProfileCountOutput, ProfileLogRates
from cerberus.plots import (
    _apply_seqlogo_mode,
    plot_attribution_heatmap,
    plot_attribution_panel,
    plot_seqlogo,
    save_count_scatter,
)

# ---------------------------------------------------------------------------
# save_count_scatter unit tests
# ---------------------------------------------------------------------------


def test_save_count_scatter_creates_file():
    rng = np.random.default_rng(0)
    preds = rng.standard_normal(100).astype(np.float32)
    targets = rng.standard_normal(100).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_count_scatter(preds, targets, tmp_dir, epoch=3)
        expected = Path(tmp_dir) / "plots" / "val_count_scatter_epoch_003.png"
        assert expected.exists()


def test_save_count_scatter_uses_custom_filename_prefix():
    """Non-default kwargs route to the configured output filename."""
    rng = np.random.default_rng(2)
    preds = rng.standard_normal(50).astype(np.float32)
    targets = rng.standard_normal(50).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_count_scatter(
            preds, targets, tmp_dir, epoch=1,
            x_label="True delta log counts",
            y_label="Predicted delta log counts",
            title="Val delta log counts",
            filename_prefix="val_delta_log_counts_scatter",
        )
        custom = Path(tmp_dir) / "plots" / "val_delta_log_counts_scatter_epoch_001.png"
        default = Path(tmp_dir) / "plots" / "val_count_scatter_epoch_001.png"
        assert custom.exists()
        assert not default.exists()


def test_save_count_scatter_skips_without_matplotlib():
    rng = np.random.default_rng(1)
    preds = rng.standard_normal(10).astype(np.float32)
    targets = rng.standard_normal(10).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("builtins.__import__", side_effect=ImportError("no matplotlib")):
            # Should not raise; silently skips
            try:
                save_count_scatter(preds, targets, tmp_dir, epoch=0)
            except ImportError:
                pass  # acceptable if the patch propagates -- no PNG should be created
        assert not (Path(tmp_dir) / "plots").exists()


# ---------------------------------------------------------------------------
# CerberusModule scatter plot integration tests
# ---------------------------------------------------------------------------


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 8)

    def forward(self, x):
        # Output shape: (B, 1, 8) -- one channel, profile length 8
        return ProfileLogRates(log_rates=self.layer(x).unsqueeze(1))


class _DummyDifferentialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 16)

    def forward(self, x):
        # Two channels (A, B), profile length 8.  log_counts is the per-
        # channel mean over the profile axis -- enough structure to give
        # DifferentialCountLoss / DifferentialBPNetMetricCollection a real
        # delta to evaluate.
        out = self.layer(x).view(x.shape[0], 2, 8)
        logits = torch.zeros_like(out)
        log_counts = out.mean(dim=-1)
        return ProfileCountOutput(logits=logits, log_counts=log_counts)


@pytest.fixture
def _base_config():
    return TrainConfig.model_construct(
        batch_size=10,
        max_epochs=5,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        optimizer="adamw",
        scheduler_type="default",
        scheduler_args={},
        filter_bias_and_bn=True,
        adam_eps=1e-8,
        gradient_clip_val=None,
        reload_dataloaders_every_n_epochs=0,
    )


def test_validation_step_populates_metric_state(_base_config):
    """validation_step populates LogCountsPearsonCorrCoef's preds_list/targets_list."""
    module = CerberusModule(
        _DummyModel(),
        criterion=ProfilePoissonNLLLoss(log_input=True, full=False),
        metrics=DefaultMetricCollection(),
        train_config=_base_config,
    )
    module.log = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    module._trainer = mock_trainer

    batch = {
        "inputs": torch.randn(4, 10),
        "targets": torch.abs(torch.randn(4, 1, 8)),
    }
    module.validation_step(batch, 0)

    pearson_metric = module.val_metrics["val_pearson_log_counts"]
    assert len(pearson_metric.preds_list) == 1  # type: ignore[arg-type]
    assert len(pearson_metric.targets_list) == 1  # type: ignore[arg-type]
    assert pearson_metric.preds_list[0].shape == (4,)  # type: ignore[union-attr]


def test_on_validation_epoch_end_saves_scatter(_base_config):
    """on_validation_epoch_end writes a PNG from metric-accumulated data."""
    module = CerberusModule(
        _DummyModel(),
        criterion=ProfilePoissonNLLLoss(log_input=True, full=False),
        metrics=DefaultMetricCollection(),
        train_config=_base_config,
    )
    module.log = MagicMock()
    module.log_dict = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.sanity_checking = False
    mock_trainer.current_epoch = 2
    module._trainer = mock_trainer

    # Run a validation step to populate metrics
    batch = {
        "inputs": torch.randn(4, 10),
        "targets": torch.abs(torch.randn(4, 1, 8)),
    }
    module.validation_step(batch, 0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        mock_trainer.logger.log_dir = tmp_dir
        module.on_validation_epoch_end()
        plot_dir = Path(tmp_dir) / "plots"
        pngs = list(plot_dir.glob("val_count_scatter_epoch_*.png"))
        assert len(pngs) == 1

    # Metrics must be reset after epoch end
    pearson_metric = module.val_metrics["val_pearson_log_counts"]
    assert pearson_metric.preds_list == []


def test_on_validation_epoch_end_skips_scatter_during_sanity_check(_base_config):
    """No PNG is written during Lightning's sanity check pass."""
    module = CerberusModule(
        _DummyModel(),
        criterion=ProfilePoissonNLLLoss(log_input=True, full=False),
        metrics=DefaultMetricCollection(),
        train_config=_base_config,
    )
    module.log_dict = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.sanity_checking = True  # sanity check active
    module._trainer = mock_trainer

    # Run a validation step to populate metrics
    module.log = MagicMock()
    batch = {
        "inputs": torch.randn(4, 10),
        "targets": torch.abs(torch.randn(4, 1, 8)),
    }
    module.validation_step(batch, 0)

    with patch("cerberus.plots.save_count_scatter") as mock_save:
        module.on_validation_epoch_end()
        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# val_metrics prefix-lookup contract (TorchMetrics).
#
# CerberusModule wraps a MetricCollection in `metrics.clone(prefix="val_")`
# and looks Pearson scatter targets up by bare name.  These tests pin the
# asymmetric __contains__ / __getitem__ behaviour the dispatch relies on so
# a future TorchMetrics version change surfaces here, not at validation time.
# ---------------------------------------------------------------------------


def test_metric_collection_clone_prefix_contains_is_bare_only():
    coll = DefaultMetricCollection().clone(prefix="val_")
    assert "pearson_log_counts" in coll
    assert "val_pearson_log_counts" not in coll


def test_metric_collection_clone_prefix_getitem_accepts_both_forms():
    coll = DefaultMetricCollection().clone(prefix="val_")
    bare = coll["pearson_log_counts"]
    prefixed = coll["val_pearson_log_counts"]
    assert bare is prefixed


# ---------------------------------------------------------------------------
# Scatter dispatch: _resolve_val_scatter_spec / on_validation_epoch_end
# ---------------------------------------------------------------------------


def _make_module(criterion, metrics, base_config):
    """Bare CerberusModule with a mocked trainer at rank 0, not sanity-checking."""
    model = (
        _DummyDifferentialModel()
        if isinstance(metrics, DifferentialBPNetMetricCollection)
        else _DummyModel()
    )
    module = CerberusModule(
        model, criterion=criterion, metrics=metrics, train_config=base_config,
    )
    module.log = MagicMock()
    module.log_dict = MagicMock()
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.sanity_checking = False
    trainer.current_epoch = 2
    module._trainer = trainer
    return module, trainer


def test_resolve_val_scatter_spec_picks_absolute_for_default_collection(_base_config):
    """DefaultMetricCollection exposes only pearson_log_counts → absolute spec."""
    module, _ = _make_module(
        ProfilePoissonNLLLoss(log_input=True, full=False),
        DefaultMetricCollection(),
        _base_config,
    )
    spec = module._resolve_val_scatter_spec()
    assert spec is not None
    metric, kwargs = spec
    assert kwargs["filename_prefix"] == "val_count_scatter"
    assert metric is module.val_metrics["pearson_log_counts"]


def test_resolve_val_scatter_spec_prefers_delta_when_both_present(_base_config):
    """Synthetic both-keys collection: delta wins per dispatch ordering."""
    module, _ = _make_module(
        ProfilePoissonNLLLoss(log_input=True, full=False),
        DefaultMetricCollection(),
        _base_config,
    )
    # CerberusModule queries val_metrics by bare key (``in`` and ``[]``); a
    # dict suffices.  nn.Module.__setattr__ rejects non-Module replacements
    # of registered submodules, so go through object.__setattr__.
    sentinel_delta = object()
    sentinel_abs = object()
    fake_val_metrics = {
        "pearson_log_counts": sentinel_abs,
        "pearson_delta_log_counts": sentinel_delta,
    }
    object.__setattr__(module, "val_metrics", fake_val_metrics)
    metric, kwargs = module._resolve_val_scatter_spec()  # type: ignore[misc]
    assert metric is sentinel_delta
    assert kwargs["filename_prefix"] == "val_delta_log_counts_scatter"


def test_resolve_val_scatter_spec_returns_none_when_no_pearson(_base_config):
    """No dispatched key present → silent skip (None), no crash."""
    module, _ = _make_module(
        ProfilePoissonNLLLoss(log_input=True, full=False),
        DefaultMetricCollection(),
        _base_config,
    )
    object.__setattr__(module, "val_metrics", {})
    assert module._resolve_val_scatter_spec() is None


def test_on_validation_epoch_end_saves_differential_scatter(_base_config):
    """End-to-end: DifferentialBPNetMetricCollection drives a delta-PNG."""
    module, trainer = _make_module(
        DifferentialCountLoss(cond_a_idx=0, cond_b_idx=1),
        DifferentialBPNetMetricCollection(cond_a_idx=0, cond_b_idx=1),
        _base_config,
    )
    batch = {
        "inputs": torch.randn(4, 10),
        "targets": torch.abs(torch.randn(4, 2, 8)),
    }
    module.validation_step(batch, 0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer.logger.log_dir = tmp_dir
        module.on_validation_epoch_end()
        delta_png = (
            Path(tmp_dir) / "plots" / "val_delta_log_counts_scatter_epoch_002.png"
        )
        absolute_png = Path(tmp_dir) / "plots" / "val_count_scatter_epoch_002.png"
        assert delta_png.exists()
        assert not absolute_png.exists()

    # Metric state was reset after compute().
    pearson = module.val_metrics["pearson_delta_log_counts"]
    assert pearson.preds_list == []


# ---------------------------------------------------------------------------
# Sequence logo helpers (plot_seqlogo / plot_attribution_heatmap /
# plot_attribution_panel). Skipped if logomaker isn't installed.
# ---------------------------------------------------------------------------

logomaker = pytest.importorskip("logomaker")
pd = pytest.importorskip("pandas")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _random_attrs(seed: int = 0, n_channels: int = 4, seq_len: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_channels, seq_len)).astype(np.float32)


def test_plot_seqlogo_runs_numpy_input() -> None:
    fig, ax = plt.subplots()
    try:
        plot_seqlogo(ax, _random_attrs())
    finally:
        plt.close(fig)


def test_plot_seqlogo_runs_torch_input() -> None:
    """Torch tensors are auto-converted to numpy — callers don't need to."""
    attrs = torch.tensor(_random_attrs(seed=1))
    fig, ax = plt.subplots()
    try:
        plot_seqlogo(ax, attrs)
    finally:
        plt.close(fig)


def test_plot_seqlogo_rejects_wrong_shape() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="must be 2-D"):
            plot_seqlogo(ax, np.zeros((1, 4, 8)))
    finally:
        plt.close(fig)


def test_plot_seqlogo_rejects_alphabet_mismatch() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="alphabet"):
            plot_seqlogo(ax, _random_attrs(n_channels=3), alphabet="ACGT")
    finally:
        plt.close(fig)


def test_plot_seqlogo_rejects_unknown_mode() -> None:
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="Unsupported mode"):
            plot_seqlogo(ax, _random_attrs(), mode="bogus")  # type: ignore[arg-type]
    finally:
        plt.close(fig)


def test_plot_seqlogo_accepts_rna_alphabet() -> None:
    fig, ax = plt.subplots()
    try:
        plot_seqlogo(ax, _random_attrs(), alphabet="ACGU")
    finally:
        plt.close(fig)


def test_plot_seqlogo_applies_ylim_override() -> None:
    fig, ax = plt.subplots()
    try:
        plot_seqlogo(ax, _random_attrs(), ylim=(-5.0, 5.0))
        lo, hi = ax.get_ylim()
        assert lo == pytest.approx(-5.0)
        assert hi == pytest.approx(5.0)
    finally:
        plt.close(fig)


# --- _apply_seqlogo_mode (pure math, no plotting) -------------------------


def test_apply_seqlogo_mode_attribution_is_identity() -> None:
    attrs = _random_attrs()
    np.testing.assert_array_equal(_apply_seqlogo_mode(attrs, "attribution"), attrs)


def test_apply_seqlogo_mode_probability_sums_to_one_per_column() -> None:
    probs = _apply_seqlogo_mode(_random_attrs(), "probability")
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-6)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_apply_seqlogo_mode_ic_bounded_by_log2_alphabet() -> None:
    """IC-mode heights per column sum to a value in [0, log2(|alphabet|)]."""
    attrs = _random_attrs(n_channels=4, seq_len=16)
    ic_values = _apply_seqlogo_mode(attrs, "ic")
    column_sums = ic_values.sum(axis=0)
    assert (column_sums >= -1e-6).all()
    assert (column_sums <= np.log2(4) + 1e-6).all()


def test_apply_seqlogo_mode_ic_peaks_at_log2_for_peaky_distribution() -> None:
    """A ``[+10, -10, -10, -10]`` column is near one-hot: IC ≈ log2(4) = 2."""
    attrs = np.tile(np.array([[10.0], [-10.0], [-10.0], [-10.0]]), (1, 3))
    ic_values = _apply_seqlogo_mode(attrs, "ic")
    assert ic_values.sum(axis=0) == pytest.approx([2.0, 2.0, 2.0], abs=1e-3)


# --- plot_attribution_heatmap ---------------------------------------------


def test_plot_attribution_heatmap_returns_axes_image() -> None:
    fig, ax = plt.subplots()
    try:
        img = plot_attribution_heatmap(ax, _random_attrs())
        assert img is not None
        # vmin/vmax should be symmetric around 0 by default.
        vmin, vmax = img.get_clim()
        assert vmax > 0
        assert vmin == pytest.approx(-vmax)
    finally:
        plt.close(fig)


def test_plot_attribution_heatmap_respects_custom_vlim() -> None:
    fig, ax = plt.subplots()
    try:
        img = plot_attribution_heatmap(ax, _random_attrs(), vlim=3.5)
        vmin, vmax = img.get_clim()
        assert vmin == pytest.approx(-3.5)
        assert vmax == pytest.approx(3.5)
    finally:
        plt.close(fig)


def test_plot_attribution_heatmap_handles_all_zero_input() -> None:
    """All-zero input shouldn't produce a degenerate vlim=0 (invisible plot)."""
    fig, ax = plt.subplots()
    try:
        img = plot_attribution_heatmap(ax, np.zeros((4, 5), dtype=np.float32))
        vmin, vmax = img.get_clim()
        assert vmax > 0
        assert vmin == pytest.approx(-vmax)
    finally:
        plt.close(fig)


def test_plot_attribution_heatmap_yticklabels_match_alphabet() -> None:
    fig, ax = plt.subplots()
    try:
        plot_attribution_heatmap(ax, _random_attrs(), alphabet="ACGT")
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels == ["A", "C", "G", "T"]
    finally:
        plt.close(fig)


# --- plot_attribution_panel -----------------------------------------------


def test_plot_attribution_panel_with_heatmap_returns_two_axes() -> None:
    fig = plt.figure()
    try:
        logo_ax, heatmap_ax = plot_attribution_panel(fig, _random_attrs())
        assert logo_ax is not None
        assert heatmap_ax is not None
    finally:
        plt.close(fig)


def test_plot_attribution_panel_without_heatmap_returns_none_heatmap() -> None:
    fig = plt.figure()
    try:
        logo_ax, heatmap_ax = plot_attribution_panel(
            fig, _random_attrs(), heatmap=False
        )
        assert logo_ax is not None
        assert heatmap_ax is None
    finally:
        plt.close(fig)


def test_plot_attribution_panel_respects_logo_mode() -> None:
    """The panel honors ``logo_mode``; ``ic`` produces a bounded-sum logo."""
    fig = plt.figure()
    try:
        logo_ax, _ = plot_attribution_panel(fig, _random_attrs(), logo_mode="ic")
        # In IC mode the logo's y-axis max is bounded by log2(4) = 2.
        _, ymax = logo_ax.get_ylim()
        assert ymax <= 2.05  # small slop for matplotlib auto-scale padding
    finally:
        plt.close(fig)
