import cerberus


def test_tpcav_helpers_are_not_part_of_top_level_package_api():
    # TPCAV is an optional integration; its helpers are accessed via
    # ``cerberus.tpcav`` and must not appear on the top-level package.
    # (``resolve_device`` is part of the public package API on this branch
    # and is deliberately NOT included here.)
    helper_names = {
        "build_tpcav_target_model",
        "list_tpcav_probe_layers",
        "resolve_tpcav_layer_name",
    }

    assert helper_names.isdisjoint(cerberus.__all__)
    for name in helper_names:
        assert not hasattr(cerberus, name)
