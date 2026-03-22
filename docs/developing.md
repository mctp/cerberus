# Developing

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

This installs test dependencies, ruff, and pre-commit, then registers the
pre-commit hooks so that linting and formatting run automatically on every
`git commit`.

## Code quality tools

| Tool | Purpose | Command |
|------|---------|---------|
| **ruff check** | Linting (unused imports, bugbear, pyupgrade, isort) | `ruff check src/ tests/ tools/` |
| **ruff format** | Deterministic formatting | `ruff format src/ tests/ tools/` |
| **pyright** | Static type checking | `npx pyright src/ tests/` |
| **pytest** | Test suite | `pytest -v tests/` |

### Ruff rules

The project enforces these rule sets (configured in `pyproject.toml`):

- **F** — Pyflakes: unused imports/variables, undefined names, redefined symbols
- **I** — isort: import sorting and grouping
- **UP** — pyupgrade: modernize syntax for Python 3.12+ (e.g., `X | Y` in `isinstance`, `collections.abc` imports, `yield from`)
- **B** — flake8-bugbear: likely bugs (mutable defaults, missing exception chaining, `zip()` without `strict=`)

### Pre-commit hooks

Two hooks run on every commit via [pre-commit](https://pre-commit.com/):

1. **ruff-format** — reformats staged files in place
2. **ruff --fix** — auto-fixes lint violations where safe

If a hook fails, it either fixes the files (re-stage and commit again) or
prints an error you need to fix manually.

To run hooks against all files (not just staged):

```bash
pre-commit run --all-files
```

To skip hooks temporarily (not recommended):

```bash
git commit --no-verify
```

## Running tests

```bash
pytest -v tests/
```

Benchmark scripts in `tests/benchmark/` are excluded by default
(`norecursedirs` in `pyproject.toml`). Run them directly:

```bash
RUN_SLOW_TESTS=1 python tests/benchmark/bench_sampler_context_length.py
```

## Type checking

```bash
npx pyright src/ tests/
```

The project targets zero errors. Warnings for missing parameter type
annotations in tests are expected and tracked via `pyrightconfig.json`.

## Style conventions

- **Line length**: 88 (ruff/Black default)
- **Quotes**: double
- **Imports**: sorted by isort (stdlib → third-party → local), one `from` per line for multi-name imports
- **Type annotations**: PEP 604 style (`X | None`, not `Optional[X]`); `from __future__ import annotations` in all `src/` modules
- **`zip()`**: always pass `strict=True` when iterables must be the same length
- **Exception chaining**: use `raise ... from err` or `raise ... from None` inside `except` blocks
