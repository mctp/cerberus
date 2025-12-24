# Codebase Structure

Cerberus is organized into modular components to separate concerns between configuration, data sampling, data extraction, and processing.

## File Organization

```
src/cerberus/
├── __init__.py         # Top-level API exports
├── config.py           # TypedDict definitions (GenomeConfig, DataConfig, SamplerConfig, TrainConfig, ModelConfig)
├── core.py             # Core data structures (e.g., Interval class)
├── datamodule.py       # PyTorch Lightning DataModule implementation
├── dataset.py          # PyTorch Dataset implementation
├── download.py         # Utilities for downloading reference genomes and datasets
├── entrypoints.py      # High-level training and model instantiation utilities
├── exclude.py          # Logic for handling exclusion intervals (blacklists)
├── genome.py           # Genome configuration helpers and folding strategies
├── loss.py             # Loss functions (BPNetLoss) and metrics
├── mask.py             # Extractors for mask data (BigBed)
├── module.py           # PyTorch Lightning Module wrapper (CerberusModule)
├── samplers.py         # Sampling logic (Interval, SlidingWindow, MultiSampler)
├── sequence.py         # Sequence extraction (FASTA handling)
├── signal.py           # Signal extraction (BigWig handling)
└── transform.py        # Data augmentation pipelines (Jitter, Crop, etc.)
```

## Module Responsibilities

### Configuration (`config.py`)
Defines the "schema" for the application. It ensures that user inputs are validated before they reach the core logic.

### Core Logic (`dataset.py`, `datamodule.py`, `module.py`, `entrypoints.py`)
These are the main entry points.
*   `dataset.py`: Ties together the Sampler (where), Extractors (what), and Transforms (how).
*   `datamodule.py`: Wraps the dataset for PyTorch Lightning, managing distributed training concerns like worker seeding and splitting.
*   `module.py`: `CerberusModule` ties together the Model, Optimizer, Scheduler, and Loss. It defines the training step.
*   `entrypoints.py`: High-level functions (`instantiate`, `train`) to wire everything up from configuration.

### Model & Training (`loss.py`, `module.py`)
*   `loss.py`: Domain-specific loss functions (e.g., `BPNetLoss` for profile prediction) and metrics.
*   `module.py`: Handles optimization configuration (using `timm`) and logging.

### Sampling (`samplers.py`)
Contains the logic for generating lists of `Interval` objects. This is decoupled from data reading, allowing lightweight iteration and manipulation of genomic regions before any heavy I/O occurs.

### Extraction (`sequence.py`, `signal.py`, `mask.py`)
Handles the low-level I/O with genomic file formats.
*   Abstracts away the differences between on-disk reading (via `pyfaidx`/`pybigtools`) and in-memory access (via dictionaries/arrays).

### Utilities
*   `genome.py`: Helper functions to load chromosome sizes and generate cross-validation folds.
*   `download.py`: Scripts to fetch public datasets and references, useful for testing and tutorials.
