import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric, MetricCollection

from cerberus.config import TrainConfig
from cerberus.module import CerberusModule


class MockTupleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        out = self.layer(x)
        return (out, out)  # Returns tuple


class MockTupleLoss(nn.Module):
    def loss_components(self, outputs, targets, **kwargs):
        return {"mse_loss": nn.functional.mse_loss(outputs[0], targets)}

    def forward(self, outputs, targets, **kwargs):
        components = self.loss_components(outputs, targets, **kwargs)
        return components["mse_loss"]


class MockTupleMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        # preds should be tuple (out1, out2)
        assert isinstance(preds, (tuple, list))
        # Ensure elements are detached tensors (no grad)
        assert not preds[0].requires_grad
        self.sum += torch.sum(preds[0] - target)
        self.count += 1

    def compute(self):
        return self.sum / self.count  # type: ignore


class DictDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"inputs": self.inputs[idx], "targets": self.targets[idx]}


def test_cerberus_module_tuple_output():
    model = MockTupleModel()
    loss = MockTupleLoss()
    metrics = MetricCollection({"test_metric": MockTupleMetric()})

    train_config = TrainConfig.model_construct(
        batch_size=2,
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer="adam",
        scheduler_type="default",
        filter_bias_and_bn=False,
        patience=5,
        scheduler_args={},
        reload_dataloaders_every_n_epochs=0,
        adam_eps=1e-8,
        gradient_clip_val=None,
    )

    module = CerberusModule(model, loss, metrics, train_config=train_config)

    # Mock data
    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 10)
    dataset = DictDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    # Use Trainer to run fit loop, which handles logging context correctly
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=dataloader)
