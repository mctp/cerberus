# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `export_predictions.py`: `--eval-split` argument (`test`/`val`/`train`/`all`, default `test`) restricts evaluation to the correct held-out chromosome set, preventing data leakage from training chromosomes into reported metrics.
- `export_predictions.py`: `--include-background` flag adds complexity-matched background intervals alongside peaks, replicating the training evaluation setup. The output TSV gains a `peak_status` column (1=peak, 0=background); background is fold-restricted to match the requested `--eval-split`. Also adds `--background-ratio` and `--seed`.

### Changed
- `export_predictions.py`: Default evaluation now covers only test-fold chromosomes (`--eval-split test`). Previous behaviour (all chromosomes) is available via `--eval-split all`.

## [0.9.1] - 2026-03-01

### Added
- Stable training mode for BPNet using weight normalization and GELU activation.

### Fixed
- Removed unused imports from `bpnet.py`.

## [0.9.0] - 2026-03-01

### Added
- Initial tracked release.