from pathlib import Path


def test_foxa1_k562_encode_preprocessing_and_training_script():
    script_path = Path("examples/chip_foxa1_k562_encode_bpnet_pomeranian.sh")
    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")

    assert 'PREPROCESS_SCRIPT="tools/preprocess_bam_bed_for_cerberus.sh"' in content
    assert 'bash "${PREPROCESS_SCRIPT}" \\' in content

    expected_rep1_inputs = [
        "ENCFF543GEC_replicate1.bam",
        "ENCFF624PSE_replicate1.bed.gz",
    ]
    for input_file in expected_rep1_inputs:
        assert input_file in content
    assert "ENCFF933WEU_replicate2.bam" not in content
    assert "ENCFF122DVT_replicate2.bed.gz" not in content

    assert '--bigwig "${BIGWIG}"' in content
    assert '--peaks "${PEAKS}"' in content

    expected_architectures = [
        "residual_post-activation_conv",
        "residual_pre-activation_conv",
        "activated_residual_pre-activation_conv",
    ]
    for architecture in expected_architectures:
        assert f"\"{architecture}\"" in content

    expected_modes = [
        "default",
        "stable",
    ]
    for mode in expected_modes:
        assert f"\"{mode}\"" in content

    assert 'for RESIDUAL_ARCHITECTURE in "${RESIDUAL_ARCHITECTURES[@]}"; do' in content
    assert 'for TRAINING_MODE in "${TRAINING_MODES[@]}"; do' in content
    assert 'if [[ "${TRAINING_MODE}" == "stable" ]]; then' in content
    assert "STABLE_ARGS=(--stable)" in content
    assert '"${STABLE_ARGS[@]}"' in content
    assert '--residual-architecture "${RESIDUAL_ARCHITECTURE}"' in content

    assert "python tools/train_bpnet.py \\" in content
    assert "python tools/train_pomeranian.py \\" in content
