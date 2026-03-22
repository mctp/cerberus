# Multi-GPU Support

Cerberus fully supports multi-GPU training by leveraging PyTorch Lightning.

*   **Support**: Implicit via `pl.Trainer`. Any strategy supported by PyTorch Lightning (DDP, DeepSpeed, FSDP) works with Cerberus.
*   **Default Strategy**: **DDP (Distributed Data Parallel)** is typically used when multiple devices are provided.
*   **Usage**: Pass `accelerator="gpu"`, `devices=N`, and optionally `strategy="ddp"` as `**trainer_kwargs` to `cerberus.train.train_single()` or `cerberus.train.train_multi()`.

For detailed implementation notes and internal design, see `docs/internal/multi_gpu_support.md` in the repository.

## Minimal Working Example

### CLI (training tools)

All training tool scripts (`train_bpnet.py`, `train_pomeranian.py`, etc.) accept `--accelerator`, `--devices`, and `--multi` flags:

```bash
# Single-fold training on 4 GPUs
python tools/train_bpnet.py \
    --config hparams.yaml \
    --accelerator gpu \
    --devices 4

# Multi-fold cross-validation on 4 GPUs
python tools/train_bpnet.py \
    --config hparams.yaml \
    --accelerator gpu \
    --devices 4 \
    --multi
```

The `--multi` flag runs full k-fold cross-validation (one fold at a time, each using all available GPUs).

### Python API

```python
from cerberus import train_single

trainer = train_single(
    genome_config=genome_config,
    data_config=data_config,
    sampler_config=sampler_config,
    model_config=model_config,
    train_config=train_config,
    num_workers=8,
    # Multi-GPU arguments (passed through to pl.Trainer)
    accelerator="gpu",
    devices=4,
    strategy="ddp",
)
```

## Interaction with DataLoader Settings

### `num_workers`

Each DDP process spawns its own set of DataLoader workers. With `num_workers=8` and `devices=4`, the system runs **32 worker processes** total. Keep total workers within your CPU core count to avoid contention.

### `reload_dataloaders_every_n_epochs`

Setting `reload_dataloaders_every_n_epochs=1` in `TrainConfig` triggers `resample()` at the start of each epoch. In DDP, each rank receives a unique seed derived from `seed + (epoch * world_size) + rank`, ensuring that all GPUs see different data while remaining deterministic.

!!! note "Recommended settings for DDP"
    - Set `num_workers` to roughly `(total_cores / num_gpus)` to avoid over-subscribing CPU.
    - Use `reload_dataloaders_every_n_epochs=1` with randomized samplers (Peak, ComplexityMatched, Random) so that each epoch sees a fresh sample across all ranks.
    - The `prepare_data()` hook runs on rank 0 only. Complexity metrics are cached to disk and loaded by all ranks in `setup()`, avoiding redundant FASTA reads.
