# cerberus

![Cerberus Logo](docs/cerberus_small.jpg)

Attentive multi-headed and highly trained companion to sequence-to-function deep-learning models.

Cerberus is a PyTorch-based framework for genomic sequence-to-function (S2F) model training. It implements efficient data loading infrastructure for handling genomic intervals, DNA sequences (FASTA), and functional signal tracks (BigWig/BigBed). The library provides composable sampling strategies—including sliding windows and weighted multi-source mixing—and on-the-fly data transformations such as jittering and reverse-complement augmentation. By abstracting these components into a unified pipeline, Cerberus facilitates the training of deep learning models with complex input/output architectures on large-scale genomic datasets.

## Installation

Cerberus requires Python 3.12 or later. It is recommended to install Cerberus in a virtual environment.

**1. Create and activate a virtual environment**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**2. Install Cerberus**

To install Cerberus and its dependencies, use `pip` from the root directory of the repository.

**Standard Installation**
```bash
pip install .
```

**Editable Installation** (recommended for development)
```bash
pip install -e .
```

## Development

**Install Development Dependencies**

To run tests and contribute, install the development dependencies:

```bash
pip install -e .[dev]
```

**Running Tests**

Run the standard test suite with `pytest`:

```bash
pytest tests/
```

**Running Slow Tests**

Some tests require downloading large genomic files and are skipped by default. To run these tests, set the `RUN_SLOW_TESTS` environment variable:

```bash
RUN_SLOW_TESTS=1 pytest tests/
```
