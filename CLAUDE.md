# CLAUDE.md

## Post-Task Requirements

After completing each task, always:

1. **Update documentation** — Update or add relevant docstrings, comments, and any affected files in `docs/` to reflect the changes made.
2. **Update tests** — Add or update unit tests in `tests/` to cover the new or modified functionality. Ensure all existing tests still pass.
3. **Update examples and tools** — Review and update any affected scripts in `examples/`, `tools/`, and `notebooks/` to stay consistent with the changes.

## Config Conventions

- **No implicit defaults when reading config dicts.** Always use `config["key"]` (bracket access), never `config.get("key", default)`. This applies everywhere — validators, transforms, dataset, module, etc. All config fields must be explicitly present in the config dict. Defaults should be set at the call site (e.g., in `parse_hparams_config` or training scripts), never where the config is consumed.

## Logging Conventions

- Every module with I/O, initialization, or file loading must use `import logging` and `logger = logging.getLogger(__name__)` at module level.
- Use `logger.info()` for significant lifecycle events: loading files into memory, completing initialization, setup summaries.
- Use `logger.debug()` for detailed operational info: lazy-loading, file routing decisions, interval counts, fallback-to-zeros.
- Pure computation modules (loss, metrics, layers, transforms, complexity) do **not** need logging.
- Never use `print()` for diagnostics in library code — always use `logger`.
