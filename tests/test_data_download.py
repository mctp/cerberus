from unittest.mock import patch

from cerberus.download import download_dataset


def test_mdapca2b_ar_dataset_download(mdapca2b_ar_dataset):
    """
    Test that the mdapca2b-ar dataset is downloaded and paths are correct.
    """
    paths = mdapca2b_ar_dataset

    assert "bigwig" in paths
    assert "narrowPeak" in paths

    assert paths["bigwig"].exists()
    assert paths["narrowPeak"].exists()

    # Check if files are not empty
    assert paths["bigwig"].stat().st_size > 0
    assert paths["narrowPeak"].stat().st_size > 0

    print(f"Verified mdapca2b-ar dataset files at: {paths}")


@patch("cerberus.download._download_file")
def test_kidney_scatac_download_mocked(mock_download, tmp_path):
    """Test kidney_scatac download logic without actually downloading."""
    results = download_dataset(tmp_path, name="kidney_scatac")

    assert mock_download.call_count == 3
    assert results["fragments"].name == "fragments.tsv.bgz"
    assert results["fragments_index"].name == "fragments.tsv.bgz.tbi"
    assert results["h5ad"].name == "gene_activity.h5ad"

    # Verify URLs passed to _download_file
    urls_called = [call.args[0] for call in mock_download.call_args_list]
    assert any("fragment.tsv.bgz" in u and not u.endswith(".tbi") for u in urls_called)
    assert any("fragment.tsv.bgz.tbi" in u for u in urls_called)
    assert any(".h5ad" in u for u in urls_called)


@patch("cerberus.download._download_file")
def test_kidney_scatac_skips_existing(mock_download, tmp_path):
    """Test that existing files are not re-downloaded."""
    out_dir = tmp_path / "kidney_scatac"
    out_dir.mkdir(parents=True)

    # Create fake existing files
    (out_dir / "fragments.tsv.bgz").touch()
    (out_dir / "fragments.tsv.bgz.tbi").touch()
    (out_dir / "gene_activity.h5ad").touch()

    results = download_dataset(tmp_path, name="kidney_scatac")

    assert mock_download.call_count == 0
    assert results["fragments"] == out_dir / "fragments.tsv.bgz"
    assert results["fragments_index"] == out_dir / "fragments.tsv.bgz.tbi"
    assert results["h5ad"] == out_dir / "gene_activity.h5ad"


@patch("cerberus.download._download_file")
def test_k562_chrombpnet_download_mocked(mock_download, tmp_path):
    """Test k562_chrombpnet download logic without actually downloading."""
    results = download_dataset(tmp_path, name="k562_chrombpnet")

    assert mock_download.call_count == 2
    assert results["narrowPeak"].name == "peaks.bed"
    assert results["bigwig"].name == "unstranded.bw"

    urls_called = [call.args[0] for call in mock_download.call_args_list]
    assert any("peaks.bed" in u for u in urls_called)
    assert any("unstranded.bw" in u for u in urls_called)


@patch("cerberus.download._download_file")
def test_k562_chrombpnet_skips_existing(mock_download, tmp_path):
    """Test that existing files are not re-downloaded."""
    out_dir = tmp_path / "k562_chrombpnet"
    out_dir.mkdir(parents=True)

    (out_dir / "peaks.bed").touch()
    (out_dir / "unstranded.bw").touch()

    results = download_dataset(tmp_path, name="k562_chrombpnet")

    assert mock_download.call_count == 0
    assert results["narrowPeak"] == out_dir / "peaks.bed"
    assert results["bigwig"] == out_dir / "unstranded.bw"


def test_download_dataset_unknown_name(tmp_path):
    """Test that unknown dataset names raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown dataset.*k562_chrombpnet"):
        download_dataset(tmp_path, name="nonexistent")
