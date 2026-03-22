
import logging
from unittest.mock import MagicMock, patch

import yaml

from cerberus.config import GenomeConfig
from cerberus.model_ensemble import (
    update_ensemble_metadata as update_ensemble_metadata_direct,
)
from cerberus.train import train_single


def _make_genome_config(k: int = 5) -> MagicMock:
    """Create a MagicMock GenomeConfig with fold_args attribute access."""
    gc = MagicMock(spec=GenomeConfig)
    gc.fold_args = {"k": k, "test_fold": None, "val_fold": None}
    gc.model_copy.return_value = gc
    return gc

def test_train_single_updates_metadata(tmp_path):
    """Test that train_single updates ensemble_metadata.yaml correctly."""

    # Mock dependencies to avoid actual training
    with patch("cerberus.train.CerberusDataModule"), \
         patch("cerberus.train.instantiate"), \
         patch("cerberus.train._train"):

        root_dir = tmp_path / "exp"
        gc = _make_genome_config(k=5)

        # Call train_single for fold 0
        train_single(
            genome_config=gc,
            data_config=MagicMock(),
            sampler_config=MagicMock(),
            model_config=MagicMock(),
            train_config=MagicMock(),
            test_fold=0,
            root_dir=root_dir
        )

        # Check metadata
        meta_path = root_dir / "ensemble_metadata.yaml"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [0]

        # Call train_single for fold 1
        train_single(
            genome_config=gc,
            data_config=MagicMock(),
            sampler_config=MagicMock(),
            model_config=MagicMock(),
            train_config=MagicMock(),
            test_fold=1,
            root_dir=root_dir
        )

        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        assert meta["folds"] == [0, 1]

def test_corrupt_metadata_warns_and_recovers(tmp_path, caplog):
    """Regression: corrupt metadata must warn (not silently pass) and still write new fold."""
    meta_path = tmp_path / "ensemble_metadata.yaml"
    meta_path.write_text(": :\n  invalid: [yaml: {{{")

    with caplog.at_level(logging.WARNING, logger="cerberus.model_ensemble"):
        update_ensemble_metadata_direct(tmp_path, fold=3)

    assert "Corrupt ensemble metadata" in caplog.text

    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert meta["folds"] == [3]

def test_corrupt_metadata_preserves_nothing_from_corrupt_file(tmp_path, caplog):
    """After corrupt metadata, only the new fold should be present (old data is lost)."""
    meta_path = tmp_path / "ensemble_metadata.yaml"
    # Write valid metadata first
    yaml.dump({"folds": [0, 1, 2]}, meta_path.open("w"))
    # Corrupt it
    meta_path.write_text("not: valid: yaml: {{{")

    with caplog.at_level(logging.WARNING, logger="cerberus.model_ensemble"):
        update_ensemble_metadata_direct(tmp_path, fold=5)

    assert "existing fold information will be lost" in caplog.text
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert meta["folds"] == [5]
