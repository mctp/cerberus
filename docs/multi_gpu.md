# Multi-GPU Support

Cerberus fully supports multi-GPU training by leveraging PyTorch Lightning.

*   **Support**: Implicit via `pl.Trainer`. Any strategy supported by PyTorch Lightning (DDP, DeepSpeed, FSDP) works with Cerberus.
*   **Default Strategy**: **DDP (Distributed Data Parallel)** is typically used when multiple devices are provided.
*   **Usage**: Pass `accelerator="gpu"`, `devices=N`, and optionally `strategy="ddp"` as `**trainer_kwargs` to `cerberus.train.train_single()` or `cerberus.train.train_multi()`.

For detailed implementation notes and internal design, see `docs/internal/multi_gpu_support.md` in the repository.
