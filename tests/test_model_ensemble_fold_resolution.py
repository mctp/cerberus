import torch
import torch.nn as nn

from cerberus.model_ensemble import ModelEnsemble, resolve_fold_dir


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))


class DummyEnsemble(ModelEnsemble):
    """
    A dummy ensemble that bypasses the complex initialization of ModelEnsemble
    but retains the methods we want to test.
    """

    def __init__(self, models):
        # Initialize the parent class (nn.ModuleDict) directly, bypassing ModelEnsemble.__init__
        nn.ModuleDict.__init__(self, models)


def test_resolve_use_folds_explicit():
    """Test that explicit use_folds is returned as is."""
    ens = DummyEnsemble({})
    explicit = ["custom"]
    assert ens._resolve_use_folds(explicit) == explicit


def test_resolve_use_folds_single_model():
    """
    Test that a single-model ensemble defaults to ["train", "test", "val"].
    This ensures that for a single model (e.g. trained on one fold or all data),
    we try to use it regardless of the fold role.
    """
    models = {"0": MockModel()}
    ens = DummyEnsemble(models)
    assert len(ens) == 1

    resolved = ens._resolve_use_folds(None)
    # Order doesn't strictly matter for set equality, but list is returned
    assert set(resolved) == {"train", "test", "val"}


def test_resolve_use_folds_multi_model():
    """
    Test that a multi-model ensemble defaults to ["test", "val"].
    This is the standard behavior for cross-validation ensembles.
    """
    models = {"0": MockModel(), "1": MockModel()}
    ens = DummyEnsemble(models)
    assert len(ens) == 2

    resolved = ens._resolve_use_folds(None)
    assert set(resolved) == {"test", "val"}


def test_resolve_use_folds_multi_model_large():
    """Test with more models just in case."""
    models = {str(i): MockModel() for i in range(5)}
    ens = DummyEnsemble(models)
    assert len(ens) == 5

    resolved = ens._resolve_use_folds(None)
    assert set(resolved) == {"test", "val"}


def test_resolve_fold_dir_finds_nested_fold_directory(tmp_path):
    fold_dir = tmp_path / "nested" / "single-fold" / "fold_0"
    fold_dir.mkdir(parents=True)

    assert resolve_fold_dir(tmp_path, 0) == fold_dir
