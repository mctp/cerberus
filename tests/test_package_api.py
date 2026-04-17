import cerberus


def test_tpcav_helpers_are_not_part_of_top_level_package_api():
    helper_names = {
        "build_tpcav_target_model",
        "list_tpcav_probe_layers",
        "resolve_tpcav_layer_name",
        "resolve_fold_dir",
        "resolve_device",
    }

    assert helper_names.isdisjoint(cerberus.__all__)
    for name in helper_names:
        assert not hasattr(cerberus, name)
