import os

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = size
        self.data = torch.randn(size, length)
        self.resample_count = 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def resample(self, seed):
        self.resample_count += 1
        print(f"Resampled with seed {seed}")


class MockDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dataset = RandomDataset(32, 64)
        self._is_initialized = True
        self.batch_size = 4
        self.num_workers = 0

    def train_dataloader(self):
        if self.trainer:
            try:
                rank = self.trainer.global_rank
                epoch = self.trainer.current_epoch
                seed = int(epoch + rank)
                self.train_dataset.resample(seed=seed)
            except:
                pass
        return DataLoader(self.train_dataset, batch_size=self.batch_size)


class MockModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(64, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") is None, reason="Skipping slow tests"
)
def test_reloading_dataloaders_integration():
    """
    Verify that reloading dataloaders triggers resampling in train_dataloader.
    """
    import warnings

    warnings.filterwarnings("ignore", ".*GPU available but not used.*")
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    dm = MockDataModule()
    model = MockModel()

    # Run for 2 epochs to ensure reloading happens
    trainer = pl.Trainer(
        max_epochs=2,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
        devices=1,
        reload_dataloaders_every_n_epochs=1,  # CRITICAL
    )
    trainer.fit(model, datamodule=dm)

    # We expect resample to be called at least once per epoch
    assert dm.train_dataset.resample_count >= 2, (
        f"Resample should be called at least twice, got {dm.train_dataset.resample_count}"
    )
