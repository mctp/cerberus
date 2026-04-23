import pytest
import torch

from cerberus.config import ModelConfig
from cerberus.loss import DifferentialCountLoss
from cerberus.metrics import (
    DifferentialLogCountsMeanSquaredError,
    DifferentialLogCountsPearsonCorrCoef,
    DifferentialLogCountsRootMeanSquaredError,
)
from cerberus.models.bpnet import DifferentialBPNetMetricCollection
from cerberus.module import instantiate_metrics_and_loss
from cerberus.output import ProfileCountOutput


OUTPUT_LEN = 32


def _bnl_targets_with_known_delta(
    sum_a: torch.Tensor,
    sum_b: torch.Tensor,
    n_channels: int = 2,
    output_len: int = OUTPUT_LEN,
) -> torch.Tensor:
    batch_size = sum_a.shape[0]
    targets = torch.zeros(batch_size, n_channels, output_len)
    targets[:, 0, :] = (sum_a / output_len).view(batch_size, 1)
    targets[:, 1, :] = (sum_b / output_len).view(batch_size, 1)
    return targets


def _make_output(log_counts: torch.Tensor) -> ProfileCountOutput:
    batch_size, n_channels = log_counts.shape
    logits = torch.zeros(batch_size, n_channels, OUTPUT_LEN)
    return ProfileCountOutput(logits=logits, log_counts=log_counts)


def test_differential_log_counts_mse_zero_when_prediction_matches_target():
    pc = 1.0
    metric = DifferentialLogCountsMeanSquaredError(
        cond_a_idx=0, cond_b_idx=1, count_pseudocount=pc
    )
    sum_a = torch.tensor([3.0, 1.0, 7.0, 0.0])
    sum_b = torch.tensor([15.0, 3.0, 1.0, 0.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    expected_delta = torch.log((sum_b + pc) / (sum_a + pc))
    log_counts = torch.zeros(4, 2)
    log_counts[:, 1] = expected_delta

    metric.update(_make_output(log_counts), targets)
    assert metric.compute().item() == pytest.approx(0.0, abs=1e-6)


def test_differential_log_counts_rmse_matches_root_mse():
    pc = 1.0
    metric = DifferentialLogCountsRootMeanSquaredError(
        cond_a_idx=0, cond_b_idx=1, count_pseudocount=pc
    )
    sum_a = torch.tensor([3.0, 7.0])
    sum_b = torch.tensor([15.0, 1.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    expected_delta = torch.log((sum_b + pc) / (sum_a + pc))

    metric.update(_make_output(torch.zeros(2, 2)), targets)
    expected_rmse = torch.sqrt((expected_delta.pow(2)).mean())
    assert metric.compute().item() == pytest.approx(expected_rmse.item(), rel=1e-6)


def test_differential_log_counts_pearson_invariant_to_positive_affine_transform():
    pc = 1.0
    sum_a = torch.tensor([1.0, 2.0, 4.0, 8.0])
    sum_b = torch.tensor([8.0, 4.0, 2.0, 1.0])
    targets = _bnl_targets_with_known_delta(sum_a, sum_b)
    true_delta = torch.log((sum_b + pc) / (sum_a + pc))

    exact_metric = DifferentialLogCountsPearsonCorrCoef(count_pseudocount=pc)
    affine_metric = DifferentialLogCountsPearsonCorrCoef(count_pseudocount=pc)

    exact_log_counts = torch.zeros(4, 2)
    exact_log_counts[:, 1] = true_delta
    affine_log_counts = torch.zeros(4, 2)
    affine_log_counts[:, 1] = 2.5 * true_delta + 1.75

    exact_metric.update(_make_output(exact_log_counts), targets)
    affine_metric.update(_make_output(affine_log_counts), targets)

    assert exact_metric.compute().item() == pytest.approx(1.0, abs=1e-6)
    assert affine_metric.compute().item() == pytest.approx(1.0, abs=1e-6)


def test_differential_log_counts_metrics_reject_out_of_range_channel():
    metric = DifferentialLogCountsMeanSquaredError(cond_a_idx=0, cond_b_idx=5)
    with pytest.raises(ValueError, match="out of range"):
        metric.update(_make_output(torch.zeros(2, 2)), torch.zeros(2, 2, OUTPUT_LEN))


def test_differential_bpnet_metric_collection_exposes_delta_metrics():
    metrics = DifferentialBPNetMetricCollection(cond_a_idx=0, cond_b_idx=1)
    assert "mse_delta_log_counts" in metrics
    assert "rmse_delta_log_counts" in metrics
    assert "pearson_delta_log_counts" in metrics


def test_instantiate_metrics_and_loss_builds_differential_metric_collection():
    config = ModelConfig.model_construct(
        name="p2",
        model_cls="cerberus.models.bpnet.MultitaskBPNet",
        loss_cls="cerberus.loss.DifferentialCountLoss",
        loss_args={"cond_a_idx": 0, "cond_b_idx": 1},
        metrics_cls="cerberus.models.bpnet.DifferentialBPNetMetricCollection",
        metrics_args={
            "cond_a_idx": 0,
            "cond_b_idx": 1,
            "log_counts_include_pseudocount": True,
        },
        model_args={"output_channels": ["a", "b"]},
        pretrained=[],
        count_pseudocount=42.0,
    )

    metrics, criterion = instantiate_metrics_and_loss(config)

    assert isinstance(criterion, DifferentialCountLoss)
    assert isinstance(
        metrics["mse_delta_log_counts"], DifferentialLogCountsMeanSquaredError
    )
    assert isinstance(
        metrics["rmse_delta_log_counts"], DifferentialLogCountsRootMeanSquaredError
    )
    assert isinstance(
        metrics["pearson_delta_log_counts"], DifferentialLogCountsPearsonCorrCoef
    )
    assert criterion.count_pseudocount == 42.0
    assert metrics["mse_delta_log_counts"].count_pseudocount == 42.0
    assert metrics["pearson_delta_log_counts"].cond_a_idx == 0
    assert metrics["pearson_delta_log_counts"].cond_b_idx == 1
