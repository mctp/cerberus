import pytest
import torch
import torch.nn as nn
from cerberus.module import CerberusModule
from cerberus.config import TrainConfig
from cerberus.loss import ProfilePoissonNLLLoss
from cerberus.metrics import DefaultMetricCollection
from cerberus.output import ProfileLogits
from timm.scheduler.cosine_lr import CosineLRScheduler

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
    def forward(self, x):
        return ProfileLogits(logits=self.layer(x))

@pytest.fixture
def base_config():
    return TrainConfig.model_construct(
        batch_size=10,
        max_epochs=5,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        optimizer="adamw",
        filter_bias_and_bn=True,
        scheduler_type="default",
        scheduler_args={},
        adam_eps=1e-8,
        gradient_clip_val=None,
        reload_dataloaders_every_n_epochs=0,
    )

def test_configure_optimizers_default(base_config):
    model = DummyModel()
    module = CerberusModule(model, criterion=ProfilePoissonNLLLoss(log_input=True, full=False), metrics=DefaultMetricCollection(), train_config=base_config)

    optim_conf = module.configure_optimizers()

    assert isinstance(optim_conf["optimizer"], torch.optim.AdamW)
    assert optim_conf["monitor"] == "val_loss"
    assert "lr_scheduler" not in optim_conf # Default has no scheduler

def test_configure_optimizers_sgd(base_config):
    config = TrainConfig.model_construct(
        **{**base_config.__dict__, "optimizer": "sgd"},
    )
    model = DummyModel()
    module = CerberusModule(model, criterion=ProfilePoissonNLLLoss(log_input=True, full=False), metrics=DefaultMetricCollection(), train_config=config)

    optim_conf = module.configure_optimizers()
    assert isinstance(optim_conf["optimizer"], torch.optim.SGD)

def test_configure_optimizers_with_cosine_scheduler(base_config):
    config = TrainConfig.model_construct(
        batch_size=10,
        max_epochs=5,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        optimizer="adamw",
        filter_bias_and_bn=True,
        scheduler_type="cosine",
        scheduler_args={"warmup_epochs": 5, "min_lr": 1e-5, "num_epochs": 100},
        adam_eps=1e-8,
        gradient_clip_val=None,
        reload_dataloaders_every_n_epochs=0,
    )

    model = DummyModel()
    module = CerberusModule(model, criterion=ProfilePoissonNLLLoss(log_input=True, full=False), metrics=DefaultMetricCollection(), train_config=config)

    optim_conf = module.configure_optimizers()

    assert isinstance(optim_conf["optimizer"], torch.optim.AdamW)

    assert "lr_scheduler" in optim_conf
    scheduler_conf = optim_conf["lr_scheduler"]

    assert scheduler_conf["interval"] == "step"
    scheduler = scheduler_conf["scheduler"]
    assert isinstance(scheduler, CosineLRScheduler)
    assert scheduler.t_initial == 100
    assert scheduler.warmup_t == 5

def test_configure_optimizers_invalid_opt(base_config):
    config = TrainConfig.model_construct(
        **{**base_config.__dict__, "optimizer": "invalid_opt_name_xyz"},
    )

    model = DummyModel()
    module = CerberusModule(model, criterion=ProfilePoissonNLLLoss(log_input=True, full=False), metrics=DefaultMetricCollection(), train_config=config)

    # timm raises Exception/KeyError
    with pytest.raises(Exception):
        module.configure_optimizers()
