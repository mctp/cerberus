# Multi-GPU Support

Cerberus fully supports multi-GPU training by leveraging PyTorch Lightning.

*   **Support**: Implicit via `pl.Trainer`. Any strategy supported by PyTorch Lightning (DDP, DeepSpeed, FSDP) works with Cerberus.
*   **Default Strategy**: **DDP (Distributed Data Parallel)** is typically used when multiple devices are provided.
*   **Usage**: Pass `accelerator="gpu"`, `devices=N`, and optionally `strategy="ddp"` directly to the `cerberus.train.train()` function.

For detailed implementation notes and internal design, see `docs/internal/multi_gpu_support.md` in the repository.
