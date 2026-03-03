# Cerberus Documentation

<img src="cerberus_small.jpg" alt="Cerberus Logo" width="25%">

Cerberus is a PyTorch-based framework for genomic sequence-to-function (S2F) model training. It implements efficient data loading infrastructure for handling genomic intervals, DNA sequences (FASTA), and functional signal tracks (BigWig/BigBed).


## User Guides

*   [**Overview**](overview.md): Introduction to the Cerberus architecture and key concepts.
*   [**Configuration**](configuration.md): Detailed reference for all configuration schemas.
*   [**Core Components**](components.md): Deep dive into Samplers, Extractors, Transforms, Losses, and Metrics.
*   [**Samplers**](samplers.md): Detailed guide to sampler types, seeding, and resampling behavior.
*   [**Sequence Complexity**](complexity.md): Reference for GC, DUST, and CpG complexity metrics.
*   [**Models**](models.md): Overview of built-in model architectures.
*   [**Usage**](usage.md): Step-by-step guide to setting up a training pipeline.
*   [**Prediction**](prediction.md): Detailed guide to inference, tiling, and BigWig generation.
*   [**Workflow Lifecycle**](workflow.md): High-level overview of Training, Prediction, and Model Exploration.
*   [**Codebase Structure**](structure.md): Guide to the file organization and module responsibilities.
*   [**Multi-GPU Training**](multi_gpu.md): Notes on distributed training with PyTorch Lightning.

!!! note "Development Notes"
    Internal implementation notes are maintained in `docs/internal/` in the repository and are not part of the public documentation.
