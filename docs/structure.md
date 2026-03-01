# Codebase Structure

Cerberus is organized into modular components to separate concerns between configuration, data sampling, data extraction, and processing.

## File Organization

```
src/cerberus/
├── __init__.py         # Top-level API exports
├── complexity.py       # Sequence complexity metrics (GC, DUST, CpG)
├── config.py           # TypedDict schemas and validation (GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig)
├── datamodule.py       # PyTorch Lightning DataModule implementation
├── dataset.py          # PyTorch Dataset implementation
├── download.py         # Utilities for downloading reference genomes and datasets
├── exclude.py          # Logic for handling exclusion intervals (blacklists, gaps)
├── genome.py           # Genome configuration helpers and folding strategies
├── interval.py         # Core Interval dataclass
├── layers.py           # Neural network building blocks (PGC, ConvNeXt)
├── logging.py          # Logging configuration helper (setup_logging)
├── loss.py             # Loss functions (MSEMultinomialLoss, PoissonMultinomialLoss, etc.)
├── mask.py             # Extractors for binary mask data (BigBed, BED)
├── metrics.py          # Evaluation metrics (Pearson, MSE, log counts)
├── model_ensemble.py   # Model loading and ensemble prediction logic
├── module.py           # PyTorch Lightning Module wrapper (CerberusModule)
├── output.py           # Model output dataclasses (ProfileLogits, ProfileLogRates, ProfileCountOutput)
├── predict_bigwig.py   # Genome-wide BigWig generation
├── samplers.py         # Sampling logic (IntervalSampler, SlidingWindowSampler, PeakSampler, etc.)
├── sequence.py         # DNA sequence extraction from FASTA files
├── signal.py           # Signal extraction from BigWig files
├── train.py            # High-level training workflows (train_single, train_multi)
└── transform.py        # Data augmentation transforms (Jitter, ReverseComplement, Scale, etc.)
```

## Module Responsibilities

### Configuration (`config.py`)
Defines the "schema" for the application. It ensures that user inputs are validated before they reach the core logic. All TypedDicts and their `validate_*` functions live here.

### Core Logic (`dataset.py`, `datamodule.py`, `module.py`, `train.py`)
These are the main entry points.
*   `dataset.py`: Ties together the Sampler (where), Extractors (what), and Transforms (how).
*   `datamodule.py`: Wraps the dataset for PyTorch Lightning, managing distributed training concerns like worker seeding and splitting.
*   `module.py`: `CerberusModule` ties together the Model, Optimizer, Scheduler, and Loss. It defines the training step.
*   `train.py`: High-level training workflows (`train_single`, `train_multi`) and orchestration.

### Model & Training (`loss.py`, `metrics.py`, `module.py`, `output.py`, `layers.py`)
*   `loss.py`: Domain-specific loss functions (e.g., `MSEMultinomialLoss`, `PoissonMultinomialLoss` for profile prediction).
*   `metrics.py`: Metrics for evaluation (Pearson Correlation, MSE, log-count metrics).
*   `output.py`: Standardized data structures for model outputs and logic for aggregating/unbatching them.
*   `module.py`: `CerberusModule` (PL wrapper), optimization configuration, and model factory logic (`instantiate`).
*   `layers.py`: Reusable neural network blocks (PGC, ConvNeXtV2, GRN).

### Prediction (`model_ensemble.py`, `predict_bigwig.py`)
*   `model_ensemble.py`: Manages loading models (single or multi-fold) and selecting models for intervals. Delegates aggregation to `output.py`.
*   `predict_bigwig.py`: Streamlines genome-wide prediction generation into BigWig files.

### Sampling (`samplers.py`)
Contains the logic for generating lists of `Interval` objects. This is decoupled from data reading, allowing lightweight iteration and manipulation of genomic regions before any heavy I/O occurs.

### Extraction (`sequence.py`, `signal.py`, `mask.py`)
Handles the low-level I/O with genomic file formats.
*   Abstracts away the differences between on-disk reading (via `pyfaidx`/`pybigtools`) and in-memory access.

### Utilities
*   `complexity.py`: Functions for calculating sequence metrics like GC content, DUST score, and CpG ratio.
*   `exclude.py`: Utilities for loading and querying exclusion intervals (blacklists, gaps).
*   `genome.py`: Helper functions to load chromosome sizes and generate cross-validation folds.
*   `download.py`: Scripts to fetch public datasets and references, useful for testing and tutorials.
*   `logging.py`: `setup_logging()` helper for configuring the `cerberus` logger hierarchy.
