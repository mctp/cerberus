"""Tests for cerberus.freeze — declarative parameter freezing."""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from cerberus.config import FreezeSpec
from cerberus.freeze import (
    FreezeReport,
    _minimal_root_set,
    apply_freeze,
    maybe_promote_ddp_strategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class BPNetLike(nn.Module):
    """Toy BPNet-shape model for path tests."""

    def __init__(self) -> None:
        super().__init__()
        self.iconv = nn.Conv1d(4, 8, 3)
        self.iconv_act = nn.ReLU()
        self.res_layers = nn.ModuleList(
            [nn.Conv1d(8, 8, 3) for _ in range(3)]
        )
        self.profile_conv = nn.Conv1d(8, 1, 3)
        self.count_dense = nn.Linear(8, 1)


class DalmatianLike(nn.Module):
    """Toy Dalmatian-shape model: bias_model + signal_model children."""

    def __init__(self) -> None:
        super().__init__()
        self.bias_model = nn.Sequential(
            nn.Conv1d(4, 8, 3),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 1, 3),
        )
        self.signal_model = nn.Sequential(
            nn.Conv1d(4, 8, 3),
            nn.Conv1d(8, 1, 3),
        )


# ---------------------------------------------------------------------------
# _minimal_root_set
# ---------------------------------------------------------------------------


class TestMinimalRootSet:
    def test_drops_descendants(self):
        roots = _minimal_root_set(
            ["bias_model", "bias_model.layers.3", "bias_model.layers.3.conv"]
        )
        assert roots == ["bias_model"]

    def test_keeps_siblings(self):
        roots = _minimal_root_set(["bias_model", "signal_model"])
        assert sorted(roots) == ["bias_model", "signal_model"]

    def test_empty(self):
        assert _minimal_root_set([]) == []

    def test_does_not_collapse_unrelated_prefix(self):
        # "bias_modelx" is not a descendant of "bias_model" even though
        # string-prefix matches would naively flag it.
        roots = _minimal_root_set(["bias_model", "bias_modelx"])
        assert sorted(roots) == ["bias_model", "bias_modelx"]


# ---------------------------------------------------------------------------
# apply_freeze — basic behavior
# ---------------------------------------------------------------------------


class TestApplyFreezeBasic:
    def test_empty_specs_is_noop(self):
        m = BPNetLike()
        report = apply_freeze(m, [])
        assert report.frozen_param_count == 0
        for p in m.parameters():
            assert p.requires_grad is True

    def test_module_pattern_freezes_subtree(self):
        m = DalmatianLike()
        report = apply_freeze(m, [FreezeSpec(pattern="bias_model")])
        for p in m.bias_model.parameters():
            assert p.requires_grad is False
        for p in m.signal_model.parameters():
            assert p.requires_grad is True
        assert report.frozen_param_count == sum(
            1 for _ in m.bias_model.parameters()
        )

    def test_parameter_pattern_freezes_single_param(self):
        m = BPNetLike()
        report = apply_freeze(m, [FreezeSpec(pattern="iconv.weight")])
        assert m.iconv.weight.requires_grad is False
        assert m.iconv.bias is not None
        assert m.iconv.bias.requires_grad is True
        assert report.frozen_param_count == 1

    def test_nested_module_pattern(self):
        """res_layers.0 should freeze only the first residual block."""
        m = BPNetLike()
        apply_freeze(m, [FreezeSpec(pattern="res_layers.0")])
        for p in m.res_layers[0].parameters():
            assert p.requires_grad is False
        for idx in (1, 2):
            for p in m.res_layers[idx].parameters():
                assert p.requires_grad is True

    def test_multiple_patterns_accumulate(self):
        m = BPNetLike()
        specs = [
            FreezeSpec(pattern="iconv"),
            FreezeSpec(pattern="res_layers"),
        ]
        report = apply_freeze(m, specs)
        for p in m.iconv.parameters():
            assert p.requires_grad is False
        for p in m.res_layers.parameters():
            assert p.requires_grad is False
        for p in m.profile_conv.parameters():
            assert p.requires_grad is True
        assert "iconv" in report.per_pattern
        assert "res_layers" in report.per_pattern

    def test_overlapping_patterns_are_idempotent(self):
        m = BPNetLike()
        specs = [
            FreezeSpec(pattern="res_layers"),
            FreezeSpec(pattern="res_layers.0"),
        ]
        report = apply_freeze(m, specs)
        for p in m.res_layers.parameters():
            assert p.requires_grad is False
        # Deduplicated count — res_layers.0 is a subset of res_layers.
        expected = sum(1 for _ in m.res_layers.parameters())
        assert report.frozen_param_count == expected


# ---------------------------------------------------------------------------
# apply_freeze — eval mode
# ---------------------------------------------------------------------------


class TestApplyFreezeEvalMode:
    def test_eval_mode_flips_training_flag(self):
        m = DalmatianLike()
        apply_freeze(m, [FreezeSpec(pattern="bias_model", eval_mode=True)])
        assert m.bias_model.training is False
        for sub in m.bias_model.modules():
            assert sub.training is False

    def test_eval_mode_does_not_affect_siblings(self):
        m = DalmatianLike()
        apply_freeze(m, [FreezeSpec(pattern="bias_model", eval_mode=True)])
        assert m.signal_model.training is True
        # Root's training flag is strictly local — siblings and the
        # parent itself are untouched.
        assert m.training is True

    def test_eval_mode_false_leaves_training_on(self):
        m = DalmatianLike()
        apply_freeze(m, [FreezeSpec(pattern="bias_model", eval_mode=False)])
        for p in m.bias_model.parameters():
            assert p.requires_grad is False
        assert m.bias_model.training is True

    def test_eval_mode_stops_dropout(self):
        """Dropout must become identity under .eval() — the whole point."""
        m = DalmatianLike()
        apply_freeze(m, [FreezeSpec(pattern="bias_model", eval_mode=True)])

        torch.manual_seed(0)
        x = torch.randn(2, 4, 32)
        y1 = m.bias_model(x)
        torch.manual_seed(1)
        y2 = m.bias_model(x)
        assert torch.equal(y1, y2)

    def test_eval_mode_collapses_to_minimal_roots(self):
        m = DalmatianLike()
        specs = [
            FreezeSpec(pattern="bias_model"),
            FreezeSpec(pattern="bias_model.0"),
        ]
        report = apply_freeze(m, specs)
        assert report.eval_roots == ["bias_model"]

    def test_param_only_match_does_not_trigger_eval(self):
        m = BPNetLike()
        report = apply_freeze(
            m, [FreezeSpec(pattern="iconv.weight", eval_mode=True)]
        )
        assert report.eval_roots == []


# ---------------------------------------------------------------------------
# apply_freeze — error cases
# ---------------------------------------------------------------------------


class TestApplyFreezeErrors:
    def test_zero_match_raises(self):
        m = BPNetLike()
        with pytest.raises(ValueError, match="matched no module or parameter"):
            apply_freeze(m, [FreezeSpec(pattern="nonexistent")])

    def test_typo_raises(self):
        m = BPNetLike()
        # Off-by-one typo: 'iconvv' vs. 'iconv'.
        with pytest.raises(ValueError, match="iconvv"):
            apply_freeze(m, [FreezeSpec(pattern="iconvv")])

    def test_glob_wildcard_is_not_a_match(self):
        """Exact-path semantics — a glob does not match anything."""
        m = BPNetLike()
        with pytest.raises(ValueError, match="iconv.\\*"):
            apply_freeze(m, [FreezeSpec(pattern="iconv.*")])

    def test_orig_mod_prefix_rejected(self):
        m = BPNetLike()
        with pytest.raises(ValueError, match="_orig_mod"):
            apply_freeze(m, [FreezeSpec(pattern="_orig_mod.iconv")])


# ---------------------------------------------------------------------------
# apply_freeze — torch.compile wrapping
# ---------------------------------------------------------------------------


class TestApplyFreezeCompiled:
    def test_uncompile_wrapper_visible_via_orig_mod(self):
        """Simulate torch.compile by wrapping in an _orig_mod attr."""
        m = DalmatianLike()
        wrapper = nn.Module()
        wrapper._orig_mod = m  # type: ignore[assignment]

        report = apply_freeze(wrapper, [FreezeSpec(pattern="bias_model")])
        for p in m.bias_model.parameters():
            assert p.requires_grad is False
        expected = sum(1 for _ in m.bias_model.parameters())
        assert report.frozen_param_count == expected


# ---------------------------------------------------------------------------
# apply_freeze — prefix collisions & disambiguation
# ---------------------------------------------------------------------------


class _PrefixColliderModel(nn.Module):
    """
    Two submodules whose names share a prefix: ``bias_model`` and
    ``bias_model_v2``.  A naive ``startswith`` descendant check would
    leak freezing across them; the "." separator in the descendant
    check keeps them apart.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bias_model = nn.Linear(4, 4)
        self.bias_model_v2 = nn.Linear(4, 4)
        self.signal_model = nn.Linear(4, 1)


class _ParamPrefixColliderModel(nn.Module):
    """
    Two leaf parameter names sharing a prefix on the same module:
    ``weight`` and ``weight_logit``. Freezing ``weight`` must not
    pull ``weight_logit`` along.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.weight_logit = nn.Parameter(torch.zeros(4))


class _NestedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, 8, 3)
        self.conv_bias = nn.Conv1d(4, 8, 3)


class _NestedEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _NestedBlock()
        self.block_norm = nn.BatchNorm1d(8)


class _NestedColliderModel(nn.Module):
    """
    Three-level nesting where sibling names share prefixes at every
    level: ``encoder.block`` vs. ``encoder.block_norm``,
    ``encoder.block.conv`` vs. ``encoder.block.conv_bias``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _NestedEncoder()
        self.encoder_head = nn.Linear(8, 1)


class _MultiBranchModel(nn.Module):
    """A realistic multi-model composition: 3 sibling sub-models."""

    def __init__(self) -> None:
        super().__init__()
        self.bias_model = nn.Sequential(
            nn.Conv1d(4, 8, 3), nn.Dropout(0.1), nn.BatchNorm1d(8)
        )
        self.signal_model = nn.Sequential(
            nn.Conv1d(4, 8, 3), nn.Conv1d(8, 1, 3)
        )
        self.aux_head = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 1))


class TestApplyFreezePrefixCollisions:
    def test_sibling_prefix_collision_at_root(self):
        """bias_model / bias_model_v2 must not bleed into each other."""
        m = _PrefixColliderModel()
        apply_freeze(m, [FreezeSpec(pattern="bias_model")])

        assert m.bias_model.weight.requires_grad is False
        assert m.bias_model.bias.requires_grad is False
        # bias_model_v2 is a distinct module — must stay trainable.
        assert m.bias_model_v2.weight.requires_grad is True
        assert m.bias_model_v2.bias.requires_grad is True
        assert m.signal_model.weight.requires_grad is True

    def test_sibling_prefix_eval_mode_does_not_leak(self):
        m = _PrefixColliderModel()
        apply_freeze(
            m, [FreezeSpec(pattern="bias_model", eval_mode=True)]
        )
        assert m.bias_model.training is False
        assert m.bias_model_v2.training is True

    def test_freezing_longer_sibling_does_not_touch_shorter(self):
        m = _PrefixColliderModel()
        apply_freeze(m, [FreezeSpec(pattern="bias_model_v2")])
        assert m.bias_model.weight.requires_grad is True
        assert m.bias_model_v2.weight.requires_grad is False

    def test_parameter_prefix_collision(self):
        """Freezing `weight` must not pull in `weight_logit`."""
        m = _ParamPrefixColliderModel()
        report = apply_freeze(m, [FreezeSpec(pattern="weight")])
        assert m.weight.requires_grad is False
        assert m.weight_logit.requires_grad is True
        assert report.frozen_param_count == 1

    def test_nested_sibling_prefix_collision(self):
        """
        At every level of nesting, a ``.`` separator in the subtree
        prefix check prevents bleeding into similarly-named siblings.
        """
        m = _NestedColliderModel()
        apply_freeze(m, [FreezeSpec(pattern="encoder.block")])

        for p in m.encoder.block.conv.parameters():
            assert p.requires_grad is False
        for p in m.encoder.block.conv_bias.parameters():
            assert p.requires_grad is False
        # encoder.block_norm is a sibling of encoder.block, NOT a
        # descendant — must stay trainable.
        for p in m.encoder.block_norm.parameters():
            assert p.requires_grad is True
        # encoder_head is a sibling of encoder at the top level.
        for p in m.encoder_head.parameters():
            assert p.requires_grad is True

    def test_leaf_parameter_prefix_collision_in_nested_module(self):
        """
        Freezing ``encoder.block.conv`` must not leak into
        ``encoder.block.conv_bias`` — a direct test of the descendant
        ``startswith(prefix + '.')`` rule at a non-root level.
        """
        m = _NestedColliderModel()
        apply_freeze(m, [FreezeSpec(pattern="encoder.block.conv")])
        for p in m.encoder.block.conv.parameters():
            assert p.requires_grad is False
        for p in m.encoder.block.conv_bias.parameters():
            assert p.requires_grad is True


class TestApplyFreezeMultiBranch:
    def test_freeze_one_branch_of_three(self):
        m = _MultiBranchModel()
        apply_freeze(m, [FreezeSpec(pattern="bias_model")])

        for p in m.bias_model.parameters():
            assert p.requires_grad is False
        for p in m.signal_model.parameters():
            assert p.requires_grad is True
        for p in m.aux_head.parameters():
            assert p.requires_grad is True

        assert m.bias_model.training is False
        assert m.signal_model.training is True
        assert m.aux_head.training is True

    def test_freeze_two_of_three_branches(self):
        m = _MultiBranchModel()
        apply_freeze(
            m,
            [
                FreezeSpec(pattern="bias_model"),
                FreezeSpec(pattern="aux_head"),
            ],
        )
        for p in m.bias_model.parameters():
            assert p.requires_grad is False
        for p in m.aux_head.parameters():
            assert p.requires_grad is False
        for p in m.signal_model.parameters():
            assert p.requires_grad is True

        assert m.bias_model.training is False
        assert m.aux_head.training is False
        assert m.signal_model.training is True

    def test_freeze_single_param_in_one_branch_leaves_others(self):
        m = _MultiBranchModel()
        # Freeze a single parameter buried inside bias_model's first conv.
        apply_freeze(m, [FreezeSpec(pattern="bias_model.0.weight")])
        assert m.bias_model[0].weight.requires_grad is False
        assert m.bias_model[0].bias.requires_grad is True
        # Eval mode must NOT fire on a param-only match.
        assert m.bias_model.training is True

    def test_mixed_module_and_parameter_specs(self):
        """
        Combine a subtree freeze with a single-parameter freeze across
        two different branches.
        """
        m = _MultiBranchModel()
        apply_freeze(
            m,
            [
                FreezeSpec(pattern="bias_model"),
                FreezeSpec(
                    pattern="signal_model.1.weight", eval_mode=False
                ),
            ],
        )
        for p in m.bias_model.parameters():
            assert p.requires_grad is False
        assert m.bias_model.training is False
        # signal_model.1.weight is frozen but signal_model.1.bias is not.
        assert m.signal_model[1].weight.requires_grad is False
        assert m.signal_model[1].bias.requires_grad is True
        assert m.signal_model[0].weight.requires_grad is True
        # Eval must not fire anywhere in signal_model (param-only match).
        assert m.signal_model.training is True


# ---------------------------------------------------------------------------
# apply_freeze — weight_norm parametrization
# ---------------------------------------------------------------------------


class TestApplyFreezeWeightNorm:
    def test_subtree_pattern_freezes_parametrization_originals(self):
        """
        ``nn.utils.parametrizations.weight_norm`` reparameterizes a
        Conv1d's weight as ``parametrizations.weight.original0`` /
        ``original1``. A subtree pattern must freeze both.
        """

        class WNModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.iconv = weight_norm(nn.Conv1d(4, 8, 3))
                self.profile_conv = nn.Conv1d(8, 1, 3)

        m = WNModel()
        params = dict(m.iconv.named_parameters())
        # Pin the assumed parameter layout before testing freeze logic.
        assert "parametrizations.weight.original0" in params
        assert "parametrizations.weight.original1" in params

        apply_freeze(m, [FreezeSpec(pattern="iconv")])
        for p in m.iconv.parameters():
            assert p.requires_grad is False
        for p in m.profile_conv.parameters():
            assert p.requires_grad is True


# ---------------------------------------------------------------------------
# maybe_promote_ddp_strategy
# ---------------------------------------------------------------------------


class TestPromoteDDP:
    def test_promotes_false_to_true_when_frozen(self):
        kwargs = {"strategy": "ddp_find_unused_parameters_false"}
        report = FreezeReport(frozen_param_count=5)
        out = maybe_promote_ddp_strategy(kwargs, report)
        assert out["strategy"] == "ddp_find_unused_parameters_true"
        # Non-mutating.
        assert kwargs["strategy"] == "ddp_find_unused_parameters_false"

    def test_noop_when_nothing_frozen(self):
        kwargs = {"strategy": "ddp_find_unused_parameters_false"}
        report = FreezeReport(frozen_param_count=0)
        out = maybe_promote_ddp_strategy(kwargs, report)
        assert out is kwargs

    def test_noop_on_auto_strategy(self):
        kwargs = {"strategy": "auto"}
        report = FreezeReport(frozen_param_count=5)
        out = maybe_promote_ddp_strategy(kwargs, report)
        assert out is kwargs

    def test_noop_on_ddp_true(self):
        kwargs = {"strategy": "ddp_find_unused_parameters_true"}
        report = FreezeReport(frozen_param_count=5)
        out = maybe_promote_ddp_strategy(kwargs, report)
        assert out is kwargs


# ---------------------------------------------------------------------------
# PL integration — .eval() survives trainer.fit() in PL >= 2.2
# ---------------------------------------------------------------------------


class _ToyDataModule(pl.LightningDataModule):
    """Single-batch loader for both train and val."""

    def __init__(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        super().__init__()
        self._batch = batch

    def _loader(self) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(*self._batch)
        return torch.utils.data.DataLoader(dataset, batch_size=self._batch[0].size(0))

    def train_dataloader(self):
        return self._loader()

    def val_dataloader(self):
        return self._loader()


class _ToyLightningModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.bias_model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.1))
        self.signal_model = nn.Linear(4, 1)
        self.train_bias_training: list[bool] = []
        self.val_bias_training: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.signal_model(self.bias_model(x))

    def training_step(self, batch, _batch_idx):
        x, y = batch
        self.train_bias_training.append(self.bias_model.training)
        return ((self(x) - y) ** 2).mean()

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        self.val_bias_training.append(self.bias_model.training)
        return ((self(x) - y) ** 2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(
            [p for p in self.parameters() if p.requires_grad], lr=1e-3
        )


class TestPLEvalModePersistence:
    def test_bias_model_stays_in_eval_across_epochs(self, tmp_path):
        """
        PL 2.2+ preserves per-submodule eval state across validation
        transitions — pins the behavior the design relies on.
        """
        pl.seed_everything(0, workers=True)
        lm = _ToyLightningModule()
        apply_freeze(lm, [FreezeSpec(pattern="bias_model", eval_mode=True)])
        assert lm.bias_model.training is False

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        dm = _ToyDataModule((x, y))

        trainer = pl.Trainer(
            max_epochs=2,
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            devices=1,
        )
        trainer.fit(lm, datamodule=dm)

        assert lm.train_bias_training, "training_step should have run"
        assert all(flag is False for flag in lm.train_bias_training), (
            f"bias_model training flag leaked True during training: "
            f"{lm.train_bias_training}"
        )
        assert all(flag is False for flag in lm.val_bias_training), (
            f"bias_model training flag leaked True during validation: "
            f"{lm.val_bias_training}"
        )

    def test_frozen_params_not_updated_by_optimizer(self, tmp_path):
        pl.seed_everything(0, workers=True)
        lm = _ToyLightningModule()
        apply_freeze(lm, [FreezeSpec(pattern="bias_model", eval_mode=True)])

        before = {
            name: p.detach().clone()
            for name, p in lm.bias_model.named_parameters()
        }

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        dm = _ToyDataModule((x, y))

        trainer = pl.Trainer(
            max_epochs=2,
            default_root_dir=str(tmp_path),
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu",
            devices=1,
        )
        trainer.fit(lm, datamodule=dm)

        for name, p in lm.bias_model.named_parameters():
            assert torch.equal(before[name], p.detach()), (
                f"frozen param {name} was updated"
            )


# ---------------------------------------------------------------------------
# Hparams round-trip
# ---------------------------------------------------------------------------


class TestHparamsRoundTrip:
    def test_model_config_freeze_serializes_and_reloads(self):
        from cerberus.config import ModelConfig

        mc = ModelConfig(
            name="m",
            model_cls="torch.nn.Linear",
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
            metrics_cls="torchmetrics.MetricCollection",
            metrics_args={},
            model_args={},
            freeze=[
                FreezeSpec(pattern="bias_model", eval_mode=True),
                FreezeSpec(pattern="iconv.weight", eval_mode=False),
            ],
        )
        dumped = mc.model_dump(mode="json")
        assert dumped["freeze"] == [
            {"pattern": "bias_model", "eval_mode": True},
            {"pattern": "iconv.weight", "eval_mode": False},
        ]
        rebuilt = ModelConfig(**dumped)
        assert rebuilt.freeze == mc.freeze

    def test_model_config_freeze_defaults_to_empty_list(self):
        """Existing configs that predate ModelConfig.freeze must still parse."""
        from cerberus.config import ModelConfig

        mc = ModelConfig(
            name="m",
            model_cls="torch.nn.Linear",
            loss_cls="cerberus.loss.MSEMultinomialLoss",
            loss_args={},
            metrics_cls="torchmetrics.MetricCollection",
            metrics_args={},
            model_args={},
        )
        assert mc.freeze == []


# ---------------------------------------------------------------------------
# Composition: PretrainedConfig (load) + ModelConfig.freeze (freeze)
# ---------------------------------------------------------------------------


class TestFreezeComposesWithPretrained:
    """Exercise the intended Dalmatian-style workflow:
    load standalone bias weights via PretrainedConfig, then freeze the
    bias subtree via FreezeSpec. The two surfaces are orthogonal."""

    def test_load_then_freeze_via_apply_freeze(self, tmp_path):
        from cerberus.config import PretrainedConfig
        from cerberus.pretrained import load_pretrained_weights

        bias_weights = DalmatianLike()
        weights_path = tmp_path / "bias.pt"
        torch.save(bias_weights.bias_model.state_dict(), weights_path)

        model = DalmatianLike()
        load_pretrained_weights(
            model,
            [
                PretrainedConfig(
                    weights_path=str(weights_path),
                    source=None,
                    target="bias_model",
                )
            ],
        )
        # After load, all params are trainable — freezing is a separate step.
        for p in model.bias_model.parameters():
            assert p.requires_grad is True

        apply_freeze(model, [FreezeSpec(pattern="bias_model", eval_mode=True)])

        for p in model.bias_model.parameters():
            assert p.requires_grad is False
        for p in model.signal_model.parameters():
            assert p.requires_grad is True
        assert model.bias_model.training is False
