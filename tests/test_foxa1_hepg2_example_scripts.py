from pathlib import Path


def test_foxa1_hepg2_encode_training_script():
    script_path = Path("examples/chip_foxa1_hepg2_encode_bpnet_pomeranian.sh")
    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")

    assert 'DATA_DIR="data/ENCODE_FOXA1_HepG2"' in content
    assert "ENCFF280BKL.bam" in content
    assert "ENCFF658EER.bed.gz" in content
    assert 'PREFIX="encode_foxa1_hepg2_rep1"' in content
    assert "chip_foxa1_hepg2_encode_rep1_bpnet_residual_architectures" in content
    assert "chip_foxa1_hepg2_encode_rep1_pomeranian" in content
    assert 'for RESIDUAL_ARCHITECTURE in "${RESIDUAL_ARCHITECTURES[@]}"; do' in content
    assert "python tools/train_bpnet.py \\" in content
    assert "python tools/train_pomeranian.py \\" in content


def test_foxa1_hepg2_encode_plot_predict_script():
    script_path = Path("examples/chip_foxa1_hepg2_encode_plot_predict.sh")
    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")

    assert 'DATA_DIR="data/ENCODE_FOXA1_HepG2"' in content
    assert 'PREFIX="encode_foxa1_hepg2_rep1"' in content
    assert "chip_foxa1_hepg2_encode_rep1_bpnet_residual_architectures" in content
    assert "chip_foxa1_hepg2_encode_rep1_pomeranian" in content
    assert "python tools/plot_training_results.py" in content
    assert "--include-background" in content
    assert "predictions_test_peaks.tsv.gz" in content
    assert "predictions_test_with_background.tsv.gz" in content
