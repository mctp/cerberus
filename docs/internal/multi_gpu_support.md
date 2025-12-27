# Multi-GPU Training Support

Cerberus leverages [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for model training and therefore fully supports multi-GPU and multi-node training strategies without requiring custom implementation in your model code.

## Support Overview

Cerberus does not implement its own distributed training logic. Instead, it relies on the underlying PyTorch Lightning infrastructure:

*   **`CerberusModule`**: Inherits from `pl.LightningModule`. It is designed to be strategy-agnostic, using standard hooks (like `training_step`) and logging methods (`self.log`) that automatically handle synchronization across devices.
*   **`train()` Entrypoint**: The `cerberus.entrypoints.train` function acts as a wrapper around `pl.Trainer`. It accepts arbitrary keyword arguments (`**trainer_kwargs`) that are passed directly to the `pl.Trainer` constructor, allowing full access to Lightning's distributed training configuration.

## Training Strategy

Since Cerberus delegates training to PyTorch Lightning, the strategy used for multi-GPU training depends on the configuration passed at runtime.

### Default Strategy
When multiple GPUs are detected or requested (e.g., `devices=2`), PyTorch Lightning typically defaults to **DDP (Distributed Data Parallel)**.

### Supported Strategies
You can use any strategy supported by your version of PyTorch Lightning, including:
*   `ddp`: Distributed Data Parallel (generally recommended for speed).
*   `ddp_spawn`: Spawns processes (useful for debugging but slower).
*   `deepspeed`: Integration with DeepSpeed for large models.
*   `fsdp`: Fully Sharded Data Parallel.

## Configuration

Multi-GPU settings are **not** defined in the `TrainConfig` dictionary (which handles hyperparameters like learning rate and batch size). Instead, they are passed as arguments to the `train()` function.

### Example: Training on 4 GPUs with DDP

```python
from cerberus.entrypoints import train

# ... (initialize module and datamodule) ...

trainer = train(
    module=module,
    datamodule=datamodule,
    train_config=train_config,
    # PyTorch Lightning distributed arguments
    accelerator="gpu",
    devices=4,
    strategy="ddp"
)
```

### Data Handling
The `CerberusDataModule` is compatible with distributed training. When using strategies like DDP, PyTorch Lightning automatically wraps the DataLoaders with a `DistributedSampler`, ensuring that each GPU processes a unique subset of the data during each epoch.
