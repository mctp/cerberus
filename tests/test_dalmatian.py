"""Tests for the Dalmatian architecture (end-to-end bias-factorized model)."""

import os

import pytest
import torch

from cerberus.loss import DalmatianLoss
from cerberus.models.biasnet import BiasNet
from cerberus.models.dalmatian import Dalmatian
from cerberus.output import (
    FactorizedProfileCountOutput,
    ProfileCountOutput,
    compute_total_log_counts,
    unbatch_modeloutput,
)

# --- Step 1: FactorizedProfileCountOutput tests ---


def test_dalmatian_output_is_profile_count_output():
    """FactorizedProfileCountOutput must be isinstance of ProfileCountOutput."""
    out = FactorizedProfileCountOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.randn(2, 1),
        bias_logits=torch.randn(2, 1, 100),
        bias_log_counts=torch.randn(2, 1),
        signal_logits=torch.randn(2, 1, 100),
        signal_log_counts=torch.randn(2, 1),
    )
    assert isinstance(out, ProfileCountOutput)


def test_dalmatian_output_detach():
    """detach() returns new FactorizedProfileCountOutput instance with all tensors detached."""
    out = FactorizedProfileCountOutput(
        logits=torch.randn(2, 1, 100, requires_grad=True),
        log_counts=torch.randn(2, 1, requires_grad=True),
        bias_logits=torch.randn(2, 1, 100, requires_grad=True),
        bias_log_counts=torch.randn(2, 1, requires_grad=True),
        signal_logits=torch.randn(2, 1, 100, requires_grad=True),
        signal_log_counts=torch.randn(2, 1, requires_grad=True),
    )
    det = out.detach()
    assert isinstance(det, FactorizedProfileCountOutput)
    assert not det.logits.requires_grad
    assert not det.log_counts.requires_grad
    assert not det.bias_logits.requires_grad
    assert not det.bias_log_counts.requires_grad
    assert not det.signal_logits.requires_grad
    assert not det.signal_log_counts.requires_grad


def test_dalmatian_output_unbatch():
    """unbatch_modeloutput works with FactorizedProfileCountOutput (all tensor fields split)."""
    out = FactorizedProfileCountOutput(
        logits=torch.randn(4, 1, 100),
        log_counts=torch.randn(4, 1),
        bias_logits=torch.randn(4, 1, 100),
        bias_log_counts=torch.randn(4, 1),
        signal_logits=torch.randn(4, 1, 100),
        signal_log_counts=torch.randn(4, 1),
    )
    items = unbatch_modeloutput(out, 4)
    assert len(items) == 4
    # All tensor fields should be present and correctly shaped
    for item in items:
        assert "bias_logits" in item
        assert "signal_logits" in item
        assert "bias_log_counts" in item
        assert "signal_log_counts" in item
        assert item["bias_logits"].shape == (1, 100)
        assert item["signal_log_counts"].shape == (1,)


def test_dalmatian_output_compute_total_log_counts():
    """compute_total_log_counts sees combined log_counts from FactorizedProfileCountOutput."""
    out = FactorizedProfileCountOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.tensor([[3.0], [4.0]]),
        bias_logits=torch.randn(2, 1, 100),
        bias_log_counts=torch.tensor([[2.0], [3.0]]),
        signal_logits=torch.randn(2, 1, 100),
        signal_log_counts=torch.tensor([[1.0], [2.0]]),
    )
    lc = compute_total_log_counts(out)
    # Should use combined log_counts, not decomposed
    assert torch.allclose(lc, torch.tensor([3.0, 4.0]))


# --- Step 2: Dalmatian model tests ---


def test_dalmatian_forward_shape():
    """Forward produces FactorizedProfileCountOutput with correct shapes."""
    model = Dalmatian(input_len=2112, output_len=1024)
    x = torch.randn(2, 4, 2112)
    out = model(x)
    assert isinstance(out, FactorizedProfileCountOutput)
    assert out.logits.shape == (2, 1, 1024)
    assert out.log_counts.shape == (2, 1)
    assert out.bias_logits.shape == (2, 1, 1024)
    assert out.bias_log_counts.shape == (2, 1)
    assert out.signal_logits.shape == (2, 1, 1024)
    assert out.signal_log_counts.shape == (2, 1)


def test_dalmatian_backward():
    """Gradients flow through both sub-models (via their respective loss paths)."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    out = model(x)
    # Combined loss (L_recon) trains signal_model only (bias is detached)
    # Bias-only loss (L_bias) trains bias_model
    loss = (
        out.logits.sum()
        + out.log_counts.sum()  # L_recon -> signal_model
        + out.bias_logits.sum()
        + out.bias_log_counts.sum()  # L_bias -> bias_model
    )
    loss.backward()
    for name, p in model.bias_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"bias_model.{name} has no gradient"
    for name, p in model.signal_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"signal_model.{name} has no gradient"


@pytest.mark.skipif(
    os.environ.get("RUN_VERY_SLOW_TESTS") is None,
    reason="Skipping very slow test (RUN_VERY_SLOW_TESTS not set)",
)
def test_dalmatian_gradient_detach():
    """L_recon gradients reach signal_model but NOT bias_model (gradient separation)."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    out = model(x)

    # Only use combined outputs (simulates L_recon path)
    loss = out.logits.sum() + out.log_counts.sum()
    loss.backward()

    # Signal model must receive gradients from L_recon
    for name, p in model.signal_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, (
                f"signal_model.{name} missing gradient from L_recon"
            )

    # Bias model must NOT receive gradients from L_recon (detached)
    for name, p in model.bias_model.named_parameters():
        if p.requires_grad:
            assert p.grad is None, (
                f"bias_model.{name} has gradient from L_recon (detach failed)"
            )


@pytest.mark.skipif(
    os.environ.get("RUN_VERY_SLOW_TESTS") is None,
    reason="Skipping very slow test (RUN_VERY_SLOW_TESTS not set)",
)
def test_dalmatian_bias_receives_gradient_from_bias_loss():
    """L_bias gradients reach bias_model but NOT signal_model."""
    model = Dalmatian()
    x = torch.randn(2, 4, 2112)
    out = model(x)

    # Only use bias outputs (simulates L_bias path)
    loss = out.bias_logits.sum() + out.bias_log_counts.sum()
    loss.backward()

    # Bias model must receive gradients from L_bias
    for name, p in model.bias_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"bias_model.{name} missing gradient from L_bias"

    # Signal model must NOT receive gradients from L_bias
    for name, p in model.signal_model.named_parameters():
        if p.requires_grad:
            assert p.grad is None, f"signal_model.{name} has gradient from L_bias"


def test_dalmatian_param_count():
    """Parameter count is in expected range for standard preset (default)."""
    model = Dalmatian()
    total = sum(p.numel() for p in model.parameters())
    bias_params = sum(p.numel() for p in model.bias_model.parameters())
    signal_params = sum(p.numel() for p in model.signal_model.parameters())
    assert 5_000 < bias_params < 20_000, f"Bias params {bias_params} out of range"
    assert 100_000 < signal_params < 300_000, (
        f"Signal params {signal_params} out of range"
    )
    assert total == bias_params + signal_params


def test_dalmatian_param_count_large():
    """Parameter count for large preset."""
    model = Dalmatian(signal_preset="large")
    signal_params = sum(p.numel() for p in model.signal_model.parameters())
    assert 1_000_000 < signal_params < 5_000_000, (
        f"Signal params {signal_params} out of range"
    )


def test_dalmatian_bias_input_crop():
    """BiasNet receives center-cropped input via auto-crop."""
    model = Dalmatian()
    # bias_input_len should be less than full input_len
    assert model.bias_input_len < model.input_len
    # bias_input_len = output_len + bias_shrinkage
    # stem: (11-1)+(11-1)=20, tower: 5*1*(9-1)=40, head: 45-1=44 → shrinkage=104
    assert model.bias_input_len == 1024 + 104  # 1128
    # Signal model uses full input
    assert model.signal_model.input_len == 2112


def test_dalmatian_shrinkage_validation():
    """Dalmatian rejects signal configs that don't produce exact output_len."""
    with pytest.raises(ValueError, match="SignalNet shrinkage"):
        Dalmatian(
            input_len=2112,
            output_len=1024,
            signal_args={"dilations": [1, 1, 2, 4]},  # too little shrinkage
        )


# --- Step 3: DalmatianLoss tests ---


def _make_factorized_output(batch_size=4, channels=1, length=100):
    """Helper to create a FactorizedProfileCountOutput for loss tests."""
    return FactorizedProfileCountOutput(
        logits=torch.randn(batch_size, channels, length, requires_grad=True),
        log_counts=torch.randn(batch_size, channels, requires_grad=True),
        bias_logits=torch.randn(batch_size, channels, length, requires_grad=True),
        bias_log_counts=torch.randn(batch_size, channels, requires_grad=True),
        signal_logits=torch.randn(batch_size, channels, length, requires_grad=True),
        signal_log_counts=torch.randn(batch_size, channels, requires_grad=True),
    )


def test_dalmatian_loss_instantiation():
    """DalmatianLoss instantiates with nested base loss from class string."""
    loss = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_weight": 1.0, "profile_weight": 1.0},
    )
    from cerberus.loss import MSEMultinomialLoss

    assert isinstance(loss.base_loss, MSEMultinomialLoss)


def test_dalmatian_loss_forward_mixed_batch():
    """Loss computes both terms with mixed peak/background batch."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    interval_source = [
        "IntervalSampler",
        "ComplexityMatchedSampler",
        "IntervalSampler",
        "ComplexityMatchedSampler",
    ]

    loss = loss_fn(output, target, interval_source=interval_source)
    assert loss.shape == ()
    assert loss.requires_grad


def test_dalmatian_loss_all_peaks():
    """When all examples are peaks, bias term is zero."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    interval_source = ["IntervalSampler"] * 4

    loss_all_peak = loss_fn(output, target, interval_source=interval_source)

    # Compare with just reconstruction loss
    recon_only = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=0.0,
    )
    loss_recon = recon_only(output, target, interval_source=interval_source)
    assert torch.allclose(loss_all_peak, loss_recon)


def test_dalmatian_loss_all_background():
    """When all examples are background, both terms contribute."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    interval_source = ["ComplexityMatchedSampler"] * 4

    loss = loss_fn(output, target, interval_source=interval_source)
    assert loss.shape == ()
    assert loss.requires_grad


def test_dalmatian_loss_backward():
    """Gradients flow through combined and bias fields."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    interval_source = [
        "IntervalSampler",
        "ComplexityMatchedSampler",
        "IntervalSampler",
        "ComplexityMatchedSampler",
    ]

    loss = loss_fn(output, target, interval_source=interval_source)
    loss.backward()

    # Combined fields get gradients from l_recon
    assert output.logits.grad is not None
    assert output.log_counts.grad is not None
    # Bias fields get gradients from l_bias (non-peak examples exist)
    assert output.bias_logits.grad is not None
    assert output.bias_log_counts.grad is not None


def test_dalmatian_loss_pseudocount_forwarding():
    """count_pseudocount is forwarded to the base loss."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        count_pseudocount=5.0,
    )
    assert loss_fn.base_loss.count_pseudocount == 5.0  # type: ignore[union-attr]


def test_dalmatian_loss_pseudocount_base_args_override():
    """Explicit base_loss_args.count_pseudocount overrides top-level."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_pseudocount": 3.0},
        count_pseudocount=5.0,
    )
    # base_loss_args takes precedence (setdefault doesn't overwrite)
    assert loss_fn.base_loss.count_pseudocount == 3.0  # type: ignore[union-attr]


def test_dalmatian_loss_with_poisson_base():
    """DalmatianLoss works with PoissonMultinomialLoss as base."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.PoissonMultinomialLoss",
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1
    interval_source = [
        "IntervalSampler",
        "ComplexityMatchedSampler",
        "IntervalSampler",
        "ComplexityMatchedSampler",
    ]

    loss = loss_fn(output, target, interval_source=interval_source)
    assert loss.shape == ()
    assert loss.requires_grad


# --- Step 4: Loss **kwargs API regression tests ---


def test_existing_losses_accept_kwargs():
    """All existing losses silently ignore extra kwargs (batch context)."""
    from cerberus.loss import MSEMultinomialLoss, PoissonMultinomialLoss

    output = ProfileCountOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.randn(2, 1),
    )
    target = torch.rand(2, 1, 100).abs() + 0.1
    extra = {
        "interval_source": ["IntervalSampler", "ComplexityMatchedSampler"],
        "some_other_key": 42,
    }

    for loss_cls in [MSEMultinomialLoss, PoissonMultinomialLoss]:
        loss_fn = loss_cls()
        loss = loss_fn(output, target, **extra)
        assert loss.shape == ()


def test_dalmatian_loss_receives_interval_source_via_kwargs():
    """DalmatianLoss extracts interval_source from kwargs (as module.py passes it)."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1

    # Simulate what _shared_step does: batch_context = {k: v for k != inputs/targets}
    batch_context = {
        "interval_source": [
            "IntervalSampler",
            "ComplexityMatchedSampler",
            "IntervalSampler",
            "ComplexityMatchedSampler",
        ]
    }
    loss = loss_fn(output, target, **batch_context)
    assert loss.shape == ()


def test_dalmatian_loss_missing_interval_source_raises():
    """DalmatianLoss raises KeyError when interval_source is missing from kwargs."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
    )
    output = _make_factorized_output(batch_size=4)
    target = torch.rand(4, 1, 100).abs() + 0.1

    with pytest.raises(KeyError):
        loss_fn(output, target)


# --- Step 5: Config / import_class integration tests ---


def test_dalmatian_model_via_import_class():
    """Dalmatian can be instantiated via import_class (config pipeline)."""
    from cerberus.utils import import_class

    cls = import_class("cerberus.models.dalmatian.Dalmatian")
    model = cls(input_len=2112, output_len=1024)
    assert isinstance(model, Dalmatian)


def test_dalmatian_loss_via_import_class():
    """DalmatianLoss can be instantiated via import_class (config pipeline)."""
    from cerberus.utils import import_class

    cls = import_class("cerberus.loss.DalmatianLoss")
    loss = cls(base_loss_cls="cerberus.loss.MSEMultinomialLoss")
    assert isinstance(loss, DalmatianLoss)


def test_dalmatian_convenience_import():
    """Dalmatian is importable from cerberus.models shortcut."""
    from cerberus.models import Dalmatian as D

    assert D is Dalmatian


def test_dalmatian_bias_model_is_biasnet():
    """Dalmatian's bias_model is a BiasNet instance (not Pomeranian)."""
    model = Dalmatian()
    assert isinstance(model.bias_model, BiasNet)


def test_dalmatian_signal_preset_standard():
    """signal_preset='standard' creates a ~150K-param SignalNet (Pomeranian K9)."""
    model = Dalmatian(signal_preset="standard")
    signal_params = sum(p.numel() for p in model.signal_model.parameters())
    assert 100_000 < signal_params < 250_000, f"Standard preset: {signal_params} params"
    x = torch.randn(2, 4, 2112)
    out = model(x)
    assert out.logits.shape == (2, 1, 1024)


def test_dalmatian_signal_preset_large():
    """signal_preset='large' (default) creates a ~3.9M-param SignalNet."""
    model = Dalmatian(signal_preset="large")
    signal_params = sum(p.numel() for p in model.signal_model.parameters())
    assert 3_000_000 < signal_params < 5_000_000, (
        f"Large preset: {signal_params} params"
    )


def test_dalmatian_signal_preset_override():
    """Explicit signal_args filters overrides preset default."""
    model = Dalmatian(signal_preset="standard", signal_args={"filters": 128})
    signal_params = sum(p.numel() for p in model.signal_model.parameters())
    # f=128 with expansion=1 (from standard preset) should be between standard and large
    assert 400_000 < signal_params < 1_000_000, (
        f"Override preset: {signal_params} params"
    )


def test_dalmatian_signal_preset_invalid():
    """Invalid signal_preset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown signal_preset"):
        Dalmatian(signal_preset="tiny")


# --- Step 6: End-to-end integration and additional tests ---


def test_dalmatian_cerberus_module_training_step():
    """Full training step through CerberusModule with Dalmatian + DalmatianLoss."""
    import tempfile
    import warnings

    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset

    from cerberus.config import TrainConfig
    from cerberus.models.pomeranian import PomeranianMetricCollection
    from cerberus.module import CerberusModule

    class DalmatianDataset(Dataset):
        def __init__(self, n=8):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {
                "inputs": torch.randn(4, 2112),
                "targets": torch.rand(1, 1024).abs() + 0.1,
                "interval_source": "IntervalSampler"
                if idx % 2 == 1
                else "ComplexityMatchedSampler",
            }

    model = Dalmatian(input_len=2112, output_len=1024)
    loss = DalmatianLoss(base_loss_cls="cerberus.loss.MSEMultinomialLoss")
    metrics = PomeranianMetricCollection()

    train_config = TrainConfig.model_construct(
        batch_size=2,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer="adam",
        scheduler_type="default",
        filter_bias_and_bn=False,
        patience=5,
        scheduler_args={},
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )

    module = CerberusModule(model, loss, metrics, train_config=train_config)
    dataset = DalmatianDataset(n=8)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = pl.Trainer(
            max_epochs=1,
            logger=False,
            default_root_dir=tmp_dir,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            accelerator="auto",
            devices=1,
            limit_train_batches=2,
            limit_val_batches=1,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*does not have many workers.*")
            trainer.fit(
                module, train_dataloaders=dataloader, val_dataloaders=dataloader
            )


def test_pomeranian_metrics_with_factorized_output():
    """PomeranianMetricCollection works with FactorizedProfileCountOutput."""
    from cerberus.models.pomeranian import PomeranianMetricCollection

    metrics = PomeranianMetricCollection()
    output = FactorizedProfileCountOutput(
        logits=torch.randn(2, 1, 100),
        log_counts=torch.randn(2, 1),
        bias_logits=torch.randn(2, 1, 100),
        bias_log_counts=torch.randn(2, 1),
        signal_logits=torch.randn(2, 1, 100),
        signal_log_counts=torch.randn(2, 1),
    )
    target = torch.rand(2, 1, 100).abs() + 0.1
    metrics.update(output, target)
    result = metrics.compute()
    assert "pearson" in result
    assert "mse_log_counts" in result


def test_dalmatian_multi_channel():
    """Dalmatian works with multiple output channels (independent samples)."""
    model = Dalmatian(
        input_len=2112,
        output_len=1024,
        output_channels=["sample1", "sample2"],
    )
    x = torch.randn(2, 4, 2112)
    out = model(x)
    # Per-channel counts (predict_total_count=False): log_counts is (B, C)
    assert out.logits.shape == (2, 2, 1024)
    assert out.log_counts.shape == (2, 2)
    assert out.bias_logits.shape == (2, 2, 1024)
    assert out.bias_log_counts.shape == (2, 2)
    assert out.signal_logits.shape == (2, 2, 1024)
    assert out.signal_log_counts.shape == (2, 2)


def test_dalmatian_multi_channel_loss():
    """DalmatianLoss works with multi-channel outputs (per-channel counts)."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_per_channel": True},
    )
    output = _make_factorized_output(batch_size=4, channels=2, length=100)
    target = torch.rand(4, 2, 100).abs() + 0.1
    interval_source = [
        "IntervalSampler",
        "ComplexityMatchedSampler",
        "IntervalSampler",
        "ComplexityMatchedSampler",
    ]

    loss = loss_fn(output, target, interval_source=interval_source)
    assert loss.shape == ()
    assert loss.requires_grad


def test_dalmatian_state_dict_roundtrip(tmp_path):
    """Dalmatian state_dict saves and loads correctly."""
    model = Dalmatian(input_len=2112, output_len=1024)
    path = tmp_path / "dalmatian.pt"
    torch.save(model.state_dict(), path)

    model2 = Dalmatian(input_len=2112, output_len=1024)
    model2.load_state_dict(torch.load(path, weights_only=True))

    # Verify parameters match
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} mismatch after load"


# --- Step 7: End-to-end optimization test ---


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") is None,
    reason="Skipping slow tests (RUN_SLOW_TESTS not set)",
)
def test_dalmatian_optimization_reduces_loss():
    """Train Dalmatian for a few steps and verify loss decreases and parameters move."""
    torch.manual_seed(42)

    model = Dalmatian(input_len=2112, output_len=1024)
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        bias_weight=1.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed synthetic data: 8 examples, alternating peak/background
    # Use a non-trivial target pattern so there's something to learn
    torch.manual_seed(0)
    inputs = torch.randn(8, 4, 2112)
    # Target: a peaked signal (Gaussian bump) so the model has structure to fit
    positions = torch.arange(1024).float()
    bump = torch.exp(-0.5 * ((positions - 512) / 50) ** 2)
    targets = bump.unsqueeze(0).unsqueeze(0).expand(8, 1, -1) + 0.1
    interval_source = ["IntervalSampler", "ComplexityMatchedSampler"] * 4

    # Snapshot initial parameters
    init_params = {n: p.clone() for n, p in model.named_parameters()}

    # Collect losses over training steps
    losses = []
    n_steps = 20
    for _ in range(n_steps):
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets, interval_source=interval_source)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # 1. Loss should decrease: final loss < initial loss
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )

    # 2. Both sub-models' parameters should have moved
    bias_moved = False
    signal_moved = False
    for name, param in model.named_parameters():
        if not torch.equal(param, init_params[name]):
            if "bias_model" in name:
                bias_moved = True
            if "signal_model" in name:
                signal_moved = True
    assert bias_moved, "BiasNet parameters did not change during training"
    assert signal_moved, "SignalNet parameters did not change during training"

    # 3. Signal count head should produce reasonable values after training
    with torch.no_grad():
        out = model(inputs[:2])
    assert torch.isfinite(out.signal_log_counts).all(), (
        f"Signal log_counts not finite: {out.signal_log_counts}"
    )


# --- Step 8: Pretrained weight loading tests ---


from cerberus.config import PretrainedConfig
from cerberus.pretrained import load_pretrained_weights


def _pc(weights_path, source=None, target=None, freeze=False):
    """Helper to build PretrainedConfig for tests."""
    return PretrainedConfig.model_construct(
        weights_path=str(weights_path),
        source=source,
        target=target,
        freeze=freeze,
    )


def test_load_pretrained_biasnet_standalone(tmp_path):
    """Load pretrained BiasNet weights into a new BiasNet (whole model)."""
    model1 = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(model1.state_dict(), tmp_path / "biasnet.pt")

    model2 = BiasNet(input_len=1128, output_len=1024, filters=12)
    load_pretrained_weights(
        model2,
        [
            _pc(tmp_path / "biasnet.pt"),
        ],
    )

    for (n1, p1), (n2, p2) in zip(
        model1.named_parameters(), model2.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} mismatch after loading"


def test_load_pretrained_biasnet_into_dalmatian(tmp_path):
    """Load standalone BiasNet weights into Dalmatian's bias_model sub-module."""
    bias = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(bias.state_dict(), tmp_path / "biasnet.pt")

    dalmatian = Dalmatian(input_len=2112, output_len=1024)

    # Snapshot signal_model params before loading
    signal_before = {n: p.clone() for n, p in dalmatian.signal_model.named_parameters()}

    load_pretrained_weights(
        dalmatian,
        [
            _pc(tmp_path / "biasnet.pt", target="bias_model"),
        ],
    )

    # Bias model should match the saved weights
    for (n1, p1), (_n2, p2) in zip(
        bias.named_parameters(), dalmatian.bias_model.named_parameters(), strict=True
    ):
        assert torch.equal(p1, p2), f"bias_model.{n1} not loaded correctly"

    # Signal model should be unchanged
    for name, param in dalmatian.signal_model.named_parameters():
        assert torch.equal(param, signal_before[name]), (
            f"signal_model.{name} was modified"
        )


def test_load_dalmatian_bias_from_dalmatian_checkpoint(tmp_path):
    """Load bias_model from a full Dalmatian checkpoint using source prefix."""
    dalmatian1 = Dalmatian(input_len=2112, output_len=1024)
    torch.save(dalmatian1.state_dict(), tmp_path / "dalmatian.pt")

    dalmatian2 = Dalmatian(input_len=2112, output_len=1024)

    load_pretrained_weights(
        dalmatian2,
        [
            _pc(tmp_path / "dalmatian.pt", source="bias_model", target="bias_model"),
        ],
    )

    # bias_model should match
    for (n1, p1), (_n2, p2) in zip(
        dalmatian1.bias_model.named_parameters(),
        dalmatian2.bias_model.named_parameters(),
        strict=True,
    ):
        assert torch.equal(p1, p2), f"bias_model.{n1} mismatch"


def test_load_full_dalmatian_checkpoint(tmp_path):
    """Load a full Dalmatian checkpoint into a new Dalmatian (all weights)."""
    dalmatian1 = Dalmatian(input_len=2112, output_len=1024)
    torch.save(dalmatian1.state_dict(), tmp_path / "dalmatian.pt")

    dalmatian2 = Dalmatian(input_len=2112, output_len=1024)

    load_pretrained_weights(
        dalmatian2,
        [
            _pc(tmp_path / "dalmatian.pt"),
        ],
    )

    for (n1, p1), (_n2, p2) in zip(
        dalmatian1.named_parameters(), dalmatian2.named_parameters(), strict=True
    ):
        assert torch.equal(p1, p2), f"{n1} mismatch"


def test_load_multiple_submodules(tmp_path):
    """Load bias and signal sub-modules from separate .pt files."""
    bias = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(bias.state_dict(), tmp_path / "bias.pt")

    from cerberus.models.pomeranian import Pomeranian

    signal = Pomeranian(input_len=2112, output_len=1024, predict_total_count=False)
    torch.save(signal.state_dict(), tmp_path / "signal.pt")

    dalmatian = Dalmatian(input_len=2112, output_len=1024)
    load_pretrained_weights(
        dalmatian,
        [
            _pc(tmp_path / "bias.pt", target="bias_model", freeze=True),
            _pc(tmp_path / "signal.pt", target="signal_model"),
        ],
    )

    # Bias should match and be frozen
    for (n1, p1), (n2, p2) in zip(
        bias.named_parameters(), dalmatian.bias_model.named_parameters(), strict=True
    ):
        assert torch.equal(p1, p2), f"bias_model.{n1} mismatch"
        assert not p2.requires_grad, f"bias_model.{n2} should be frozen"

    # Signal should match and NOT be frozen
    for (n1, p1), (n2, p2) in zip(
        signal.named_parameters(),
        dalmatian.signal_model.named_parameters(),
        strict=True,
    ):
        assert torch.equal(p1, p2), f"signal_model.{n1} mismatch"
        assert p2.requires_grad, f"signal_model.{n2} should not be frozen"


def test_freeze_pretrained_biasnet(tmp_path):
    """Freezing sets requires_grad=False on all loaded parameters."""
    model = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(model.state_dict(), tmp_path / "biasnet.pt")

    model2 = BiasNet(input_len=1128, output_len=1024, filters=12)
    load_pretrained_weights(
        model2,
        [
            _pc(tmp_path / "biasnet.pt", freeze=True),
        ],
    )

    for name, p in model2.named_parameters():
        assert not p.requires_grad, f"{name} should be frozen"


def test_freeze_bias_leaves_signal_unfrozen(tmp_path):
    """Freezing bias_model in Dalmatian leaves signal_model trainable."""
    bias = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(bias.state_dict(), tmp_path / "biasnet.pt")

    dalmatian = Dalmatian(input_len=2112, output_len=1024)
    load_pretrained_weights(
        dalmatian,
        [
            _pc(tmp_path / "biasnet.pt", target="bias_model", freeze=True),
        ],
    )

    for name, p in dalmatian.bias_model.named_parameters():
        assert not p.requires_grad, f"bias_model.{name} should be frozen"

    for name, p in dalmatian.signal_model.named_parameters():
        assert p.requires_grad, f"signal_model.{name} should NOT be frozen"


def test_pretrained_architecture_mismatch_raises(tmp_path):
    """Architecture mismatch (different filters) raises RuntimeError."""
    model_f12 = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(model_f12.state_dict(), tmp_path / "f12.pt")

    model_f24 = BiasNet(input_len=1128, output_len=1024, filters=24)
    with pytest.raises(RuntimeError):
        load_pretrained_weights(
            model_f24,
            [
                _pc(tmp_path / "f12.pt"),
            ],
        )


def test_pretrained_source_prefix_not_found_raises(tmp_path):
    """Invalid source prefix raises ValueError."""
    model = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(model.state_dict(), tmp_path / "biasnet.pt")

    dalmatian = Dalmatian(input_len=2112, output_len=1024)
    with pytest.raises(ValueError, match="No keys found"):
        load_pretrained_weights(
            dalmatian,
            [
                _pc(
                    tmp_path / "biasnet.pt",
                    source="nonexistent_module",
                    target="bias_model",
                ),
            ],
        )


def test_pretrained_invalid_target_raises(tmp_path):
    """Invalid target submodule name raises AttributeError."""
    model = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(model.state_dict(), tmp_path / "biasnet.pt")

    dalmatian = Dalmatian(input_len=2112, output_len=1024)
    with pytest.raises(AttributeError):
        load_pretrained_weights(
            dalmatian,
            [
                _pc(tmp_path / "biasnet.pt", target="nonexistent_model"),
            ],
        )


def test_load_real_pretrained_biasnet():
    """Load actual trained BiasNet from tests/data/ (integration test)."""
    pt_path = "pretrained/biasnet/atac_k562.pt"
    if not os.path.exists(pt_path):
        pytest.skip("Pretrained BiasNet not found at pretrained/biasnet/atac_k562.pt")

    model = BiasNet(input_len=1128, output_len=1024, filters=12)
    load_pretrained_weights(
        model,
        [
            _pc(pt_path),
        ],
    )

    # Verify model produces output
    x = torch.randn(1, 4, 1128)
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (1, 1, 1024)


def test_load_real_pretrained_biasnet_into_dalmatian():
    """Load actual trained BiasNet into Dalmatian's bias_model (integration test)."""
    pt_path = "pretrained/biasnet/atac_k562.pt"
    if not os.path.exists(pt_path):
        pytest.skip("Pretrained BiasNet not found at pretrained/biasnet/atac_k562.pt")

    dalmatian = Dalmatian(input_len=2112, output_len=1024)
    load_pretrained_weights(
        dalmatian,
        [
            _pc(pt_path, target="bias_model", freeze=True),
        ],
    )

    # Verify frozen bias model produces output in combined model
    x = torch.randn(1, 4, 2112)
    with torch.no_grad():
        out = dalmatian(x)
    assert out.logits.shape == (1, 1, 1024)
    assert out.bias_logits.shape == (1, 1, 1024)

    # Verify bias is frozen
    for name, p in dalmatian.bias_model.named_parameters():
        assert not p.requires_grad, f"bias_model.{name} should be frozen"


def test_pretrained_empty_list_is_noop():
    """Empty pretrained list does nothing (no error)."""
    model = BiasNet(input_len=1128, output_len=1024, filters=12)
    params_before = {n: p.clone() for n, p in model.named_parameters()}

    load_pretrained_weights(model, [])

    for name, param in model.named_parameters():
        assert torch.equal(param, params_before[name])


# --- Step 9: bias_args / signal_args forwarding tests ---


def test_dalmatian_bias_args_forwarding():
    """bias_args dict is forwarded to BiasNet with native parameter names."""
    model = Dalmatian(bias_args={"filters": 24, "dropout": 0.2})
    # Verify the bias model was created with the specified filters
    bias_params = sum(p.numel() for p in model.bias_model.parameters())
    default_model = Dalmatian()
    default_bias_params = sum(p.numel() for p in default_model.bias_model.parameters())
    assert bias_params > default_bias_params  # 24 filters > 12 filters


def test_dalmatian_signal_args_forwarding():
    """signal_args dict is forwarded to Pomeranian with native parameter names."""
    model = Dalmatian(signal_args={"dropout": 0.3})
    x = torch.randn(1, 4, 2112)
    out = model(x)
    assert out.logits.shape == (1, 1, 1024)


def test_dalmatian_bias_args_none_uses_defaults():
    """bias_args=None uses BiasNet defaults (same as no override)."""
    model_default = Dalmatian()
    model_none = Dalmatian(bias_args=None)
    p1 = sum(p.numel() for p in model_default.bias_model.parameters())
    p2 = sum(p.numel() for p in model_none.bias_model.parameters())
    assert p1 == p2


# --- Step 10: compute_shrinkage() tests ---


def test_biasnet_compute_shrinkage_default():
    """BiasNet.compute_shrinkage() with defaults matches known value."""
    # stem: (11-1)+(11-1)=20, tower: 5*1*(9-1)=40, head: 45-1=44 → 104
    assert BiasNet.compute_shrinkage() == 104


def test_pomeranian_compute_shrinkage_default():
    """Pomeranian.compute_shrinkage() with defaults matches known value."""
    from cerberus.models.pomeranian import Pomeranian

    # stem: 20, tower: (1+1+2+4+8+16+32+64)*(9-1) = 128*8 = 1024, head: 44 → 1088
    assert Pomeranian.compute_shrinkage() == 1088


def test_bpnet_compute_shrinkage_default():
    """BPNet.compute_shrinkage() with defaults matches known value."""
    from cerberus.models.bpnet import BPNet

    # stem: 20, tower: sum(2^i*(3-1) for i=1..8) = 2*510 = 1020, head: 74 → 1114
    assert BPNet.compute_shrinkage() == 1114


def test_bpnet1024_compute_shrinkage():
    """BPNet1024 shrinkage matches its documented 1088."""
    from cerberus.models.bpnet import BPNet

    # stem: 20, tower: 2*510=1020, head: 48 → but BPNet1024 uses k=49, dil_k=3
    shrinkage = BPNet.compute_shrinkage(
        conv_kernel_size=21,
        n_dilated_layers=8,
        dil_kernel_size=3,
        profile_kernel_size=49,
    )
    assert shrinkage == 1088


def test_compute_shrinkage_matches_model_geometry():
    """compute_shrinkage() matches actual input_len - output_len for Dalmatian."""
    model = Dalmatian(input_len=2112, output_len=1024)
    # BiasNet: input_len = output_len + shrinkage
    assert model.bias_input_len == 1024 + BiasNet.compute_shrinkage()


# --- Step 11: shared_bias tests ---


def test_dalmatian_shared_bias_shapes():
    """shared_bias=True: bias has 1 channel, signal has N channels, combined has N."""
    model = Dalmatian(
        input_len=2112,
        output_len=1024,
        output_channels=["ct1", "ct2", "ct3"],
        shared_bias=True,
    )
    x = torch.randn(2, 4, 2112)
    out = model(x)
    # Combined output has N channels
    assert out.logits.shape == (2, 3, 1024)
    assert out.log_counts.shape == (2, 3)
    # Bias has 1 channel
    assert out.bias_logits.shape == (2, 1, 1024)
    assert out.bias_log_counts.shape == (2, 1)
    # Signal has N channels
    assert out.signal_logits.shape == (2, 3, 1024)
    assert out.signal_log_counts.shape == (2, 3)


def test_dalmatian_shared_bias_backward():
    """Gradients flow through both sub-models with shared_bias=True."""
    model = Dalmatian(
        output_channels=["ct1", "ct2"],
        shared_bias=True,
    )
    x = torch.randn(2, 4, 2112)
    out = model(x)
    loss = (
        out.logits.sum()
        + out.log_counts.sum()
        + out.bias_logits.sum()
        + out.bias_log_counts.sum()
    )
    loss.backward()
    for name, p in model.bias_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"bias_model.{name} has no gradient"
    for name, p in model.signal_model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"signal_model.{name} has no gradient"


def test_dalmatian_shared_bias_loss():
    """DalmatianLoss sums targets across channels when bias has fewer channels."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_per_channel": True},
        bias_weight=1.0,
    )
    # 1-channel bias, 3-channel signal/combined
    output = FactorizedProfileCountOutput(
        logits=torch.randn(4, 3, 100, requires_grad=True),
        log_counts=torch.randn(4, 3, requires_grad=True),
        bias_logits=torch.randn(4, 1, 100, requires_grad=True),
        bias_log_counts=torch.randn(4, 1, requires_grad=True),
        signal_logits=torch.randn(4, 3, 100, requires_grad=True),
        signal_log_counts=torch.randn(4, 3, requires_grad=True),
    )
    target = torch.rand(4, 3, 100).abs() + 0.1
    interval_source = [
        "IntervalSampler",
        "ComplexityMatchedSampler",
        "IntervalSampler",
        "ComplexityMatchedSampler",
    ]
    loss = loss_fn(output, target, interval_source=interval_source)
    assert loss.shape == ()
    assert loss.requires_grad
    # Verify gradients flow to bias
    loss.backward()
    assert output.bias_logits.grad is not None
    assert output.bias_log_counts.grad is not None


def test_dalmatian_shared_bias_loss_zero_weight():
    """bias_weight=0.0 skips L_bias entirely (no bias gradients from loss)."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_per_channel": True},
        bias_weight=0.0,
    )
    output = FactorizedProfileCountOutput(
        logits=torch.randn(4, 3, 100, requires_grad=True),
        log_counts=torch.randn(4, 3, requires_grad=True),
        bias_logits=torch.randn(4, 1, 100, requires_grad=True),
        bias_log_counts=torch.randn(4, 1, requires_grad=True),
        signal_logits=torch.randn(4, 3, 100, requires_grad=True),
        signal_log_counts=torch.randn(4, 3, requires_grad=True),
    )
    target = torch.rand(4, 3, 100).abs() + 0.1
    interval_source = ["ComplexityMatchedSampler"] * 4

    components = loss_fn.loss_components(
        output, target, interval_source=interval_source
    )
    loss = loss_fn(output, target, interval_source=interval_source)
    # Total loss should equal recon_loss (bias_weight=0)
    assert torch.allclose(loss, components["recon_loss"])


def test_dalmatian_shared_bias_broadcast_math():
    """Verify broadcasting and expand_as produce mathematically correct combined outputs.

    Profile: combined[:, c, :] == bias[:, 0, :] + signal[:, c, :]  for each channel c
    Counts:  combined[:, c]    == logsumexp(bias[:, 0], signal[:, c])  for each channel c
    """
    model = Dalmatian(
        output_channels=["ct1", "ct2", "ct3"],
        shared_bias=True,
    )
    x = torch.randn(2, 4, 2112)
    with torch.no_grad():
        out = model(x)

    n_channels = 3
    for c in range(n_channels):
        # Profile: each channel should be bias[:,0,:] + signal[:,c,:]
        expected_logits = out.bias_logits[:, 0, :].detach() + out.signal_logits[:, c, :]
        assert torch.allclose(out.logits[:, c, :], expected_logits, atol=1e-6), (
            f"Profile broadcast wrong for channel {c}"
        )

        # Counts: each channel should be logsumexp(bias[:,0], signal[:,c])
        stacked = torch.stack(
            [out.bias_log_counts[:, 0].detach(), out.signal_log_counts[:, c]], dim=-1
        )
        expected_counts = torch.logsumexp(stacked, dim=-1)
        assert torch.allclose(out.log_counts[:, c], expected_counts, atol=1e-6), (
            f"Count logsumexp wrong for channel {c}"
        )


def test_dalmatian_shared_bias_default_false():
    """Default shared_bias=False preserves existing single-channel behavior."""
    model = Dalmatian()
    assert not model.shared_bias
    x = torch.randn(1, 4, 2112)
    out = model(x)
    # Both sub-models should have the same number of channels
    assert out.bias_logits.shape[1] == out.signal_logits.shape[1]


# --- Step 12: Additional coverage tests ---


def test_pomeranian_compute_shrinkage_list_dil_kernel_size():
    """Pomeranian.compute_shrinkage handles per-layer kernel size list."""
    from cerberus.models.pomeranian import Pomeranian

    # 8-layer schedule with varying kernel sizes
    dilations = [1, 1, 2, 4, 8, 16, 32, 64]
    dil_kernels = [5, 5, 7, 7, 9, 9, 11, 11]
    result = Pomeranian.compute_shrinkage(
        dilations=dilations, dil_kernel_size=dil_kernels
    )
    # stem: 20, tower: sum(d*(k-1)), head: 44
    expected_tower = sum(
        d * (k - 1) for d, k in zip(dilations, dil_kernels, strict=False)
    )
    assert result == 20 + expected_tower + 44


def test_biasnet_compute_shrinkage_custom_args():
    """BiasNet.compute_shrinkage with non-default architecture params."""
    # Single int kernel, 3 layers, custom dilations
    result = BiasNet.compute_shrinkage(
        conv_kernel_size=21, n_layers=3, dilations=[1, 2, 4], dil_kernel_size=5
    )
    # stem: 20, tower: (1+2+4)*(5-1)=28, head: 44
    assert result == 20 + 28 + 44


def test_dalmatian_bias_args_architecture():
    """bias_args with architecture-changing params updates shrinkage correctly."""
    model = Dalmatian(bias_args={"n_layers": 3, "dilations": [1, 2, 4]})
    expected_shrinkage = BiasNet.compute_shrinkage(n_layers=3, dilations=[1, 2, 4])
    assert model.bias_input_len == 1024 + expected_shrinkage


def test_dalmatian_signal_args_expansion_override():
    """signal_args can override preset expansion."""
    model_standard = Dalmatian(signal_preset="standard")
    model_override = Dalmatian(signal_preset="standard", signal_args={"expansion": 2})
    # expansion=2 should increase params relative to standard (expansion=1)
    p_standard = sum(p.numel() for p in model_standard.signal_model.parameters())
    p_override = sum(p.numel() for p in model_override.signal_model.parameters())
    assert p_override > p_standard


def test_dalmatian_shared_bias_single_channel_equivalent():
    """shared_bias=True with 1 output channel is numerically identical to False."""
    torch.manual_seed(42)
    model_shared = Dalmatian(output_channels=["signal"], shared_bias=True)
    model_normal = Dalmatian(output_channels=["signal"], shared_bias=False)
    # Copy weights so both models are identical
    model_normal.load_state_dict(model_shared.state_dict())
    # eval() disables dropout so results are deterministic
    model_shared.eval()
    model_normal.eval()
    x = torch.randn(2, 4, 2112)
    with torch.no_grad():
        out_shared = model_shared(x)
        out_normal = model_normal(x)
    assert torch.allclose(out_shared.logits, out_normal.logits, atol=1e-6)
    assert torch.allclose(out_shared.log_counts, out_normal.log_counts, atol=1e-6)


def test_dalmatian_shared_bias_loss_all_peaks_channel_mismatch():
    """All-peaks batch with channel mismatch: L_bias is exactly zero."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_per_channel": True},
        bias_weight=1.0,
    )
    output = FactorizedProfileCountOutput(
        logits=torch.randn(4, 3, 100, requires_grad=True),
        log_counts=torch.randn(4, 3, requires_grad=True),
        bias_logits=torch.randn(4, 1, 100, requires_grad=True),
        bias_log_counts=torch.randn(4, 1, requires_grad=True),
        signal_logits=torch.randn(4, 3, 100, requires_grad=True),
        signal_log_counts=torch.randn(4, 3, requires_grad=True),
    )
    target = torch.rand(4, 3, 100).abs() + 0.1
    interval_source = ["IntervalSampler"] * 4  # all peaks

    components = loss_fn.loss_components(
        output, target, interval_source=interval_source
    )
    assert components["bias_loss"].item() == 0.0


def test_dalmatian_shared_bias_loss_sum_targets_correctness():
    """Verify summed targets are element-wise correct in shared_bias L_bias."""
    loss_fn = DalmatianLoss(
        base_loss_cls="cerberus.loss.MSEMultinomialLoss",
        base_loss_args={"count_per_channel": True},
        bias_weight=1.0,
    )
    # Known targets: 3 channels
    target = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ]
    )  # (1, 3, 3)
    # Sum across channels: [12.0, 15.0, 18.0]
    output = FactorizedProfileCountOutput(
        logits=torch.randn(1, 3, 3, requires_grad=True),
        log_counts=torch.randn(1, 3, requires_grad=True),
        bias_logits=torch.randn(1, 1, 3, requires_grad=True),
        bias_log_counts=torch.randn(1, 1, requires_grad=True),
        signal_logits=torch.randn(1, 3, 3, requires_grad=True),
        signal_log_counts=torch.randn(1, 3, requires_grad=True),
    )
    interval_source = ["ComplexityMatchedSampler"]  # background

    # Should not raise — the sum path triggers and produces valid loss
    components = loss_fn.loss_components(
        output, target, interval_source=interval_source
    )
    assert torch.isfinite(components["bias_loss"])
    assert components["bias_loss"].item() > 0.0


def test_dalmatian_shared_bias_state_dict_roundtrip(tmp_path):
    """State dict save/load with shared_bias=True preserves all weights."""
    model1 = Dalmatian(output_channels=["ct1", "ct2", "ct3"], shared_bias=True)
    path = tmp_path / "dalmatian_shared.pt"
    torch.save(model1.state_dict(), path)

    model2 = Dalmatian(output_channels=["ct1", "ct2", "ct3"], shared_bias=True)
    model2.load_state_dict(torch.load(path, weights_only=True))

    for (n1, p1), (n2, p2) in zip(
        model1.named_parameters(), model2.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} mismatch after load"


def test_load_pretrained_biasnet_into_shared_bias_dalmatian(tmp_path):
    """Standalone BiasNet (1 channel) loads into shared_bias Dalmatian's bias_model."""
    bias = BiasNet(input_len=1128, output_len=1024, filters=12)
    torch.save(bias.state_dict(), tmp_path / "biasnet.pt")

    dalmatian = Dalmatian(
        output_channels=["ct1", "ct2", "ct3"],
        shared_bias=True,
    )
    load_pretrained_weights(
        dalmatian,
        [_pc(tmp_path / "biasnet.pt", target="bias_model")],
    )

    # Bias model weights should match
    for (n1, p1), (_n2, p2) in zip(
        bias.named_parameters(), dalmatian.bias_model.named_parameters(), strict=True
    ):
        assert torch.equal(p1, p2), f"bias_model.{n1} not loaded correctly"

    # Signal model should have 3 channels, bias model 1
    assert dalmatian.bias_model.n_output_channels == 1
    assert dalmatian.signal_model.n_output_channels == 3


def test_dalmatian_shared_bias_single_biasnet():
    """shared_bias=True creates exactly one BiasNet with 1 output channel, not one per track."""
    n_channels = 14
    channel_names = [f"cell_type_{i}" for i in range(n_channels)]
    model = Dalmatian(
        output_channels=channel_names,
        shared_bias=True,
    )

    # There is exactly one bias_model, not N
    assert isinstance(model.bias_model, BiasNet)
    assert model.bias_model.n_output_channels == 1

    # SignalNet has N channels
    assert model.signal_model.n_output_channels == n_channels

    # BiasNet param count is identical regardless of number of output tracks
    model_1ch = Dalmatian(output_channels=["single"], shared_bias=True)
    bias_params_14 = sum(p.numel() for p in model.bias_model.parameters())
    bias_params_1 = sum(p.numel() for p in model_1ch.bias_model.parameters())
    assert bias_params_14 == bias_params_1, (
        f"BiasNet params should be identical: {bias_params_14} vs {bias_params_1}"
    )

    # BiasNet forward produces 1-channel output, not N
    x = torch.randn(2, 4, 2112)
    with torch.no_grad():
        bias_out = model.bias_model(x)
    assert bias_out.logits.shape == (2, 1, 1024)
    assert bias_out.log_counts.shape == (2, 1)

    # Combined output broadcasts to N channels
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (2, n_channels, 1024)
    assert out.log_counts.shape == (2, n_channels)
