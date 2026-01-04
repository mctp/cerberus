import torch
from collections.abc import Iterable
from typing import Any

from cerberus.interval import Interval
from cerberus.dataset import CerberusDataset
from cerberus.model_ensemble import ModelEnsemble
from cerberus.output import ModelOutput


def predict_intervals(
    intervals: Iterable[Interval],
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    use_folds: list[str] = ["test", "val"],
    aggregation: str = "model",
    batch_size: int = 64,
) -> ModelOutput:
    """
    Deprecated: Use model_ensemble.predict_intervals instead.
    Predicts and aggregates outputs for multiple intervals in batches.
    """
    return model_ensemble.predict_intervals(
        intervals,
        dataset,
        use_folds=use_folds,
        aggregation=aggregation,
        batch_size=batch_size,
    )


def predict_output_intervals(
    intervals: Iterable[Interval],
    dataset: CerberusDataset,
    model_ensemble: ModelEnsemble,
    stride: int | None = None,
    use_folds: list[str] = ["test", "val"],
    aggregation: str = "model",
    batch_size: int = 64,
) -> list[ModelOutput]:
    """
    Deprecated: Use model_ensemble.predict_output_intervals instead.
    Predicts outputs for a list of target intervals by tiling them with input intervals.
    """
    return model_ensemble.predict_output_intervals(
        intervals,
        dataset,
        stride=stride,
        use_folds=use_folds,
        aggregation=aggregation,
        batch_size=batch_size,
    )
