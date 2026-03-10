from pathlib import Path


def test_foxa1_bpnet_residual_architecture_sweep_script():
    script_path = Path("examples/chip_foxa1_22rv1_bpnet_all_residual_architectures.sh")
    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")

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
