# CLAUDE.md

## Post-Task Requirements

After completing each task, always:

1. **Update documentation** — Update or add relevant docstrings, comments, and any affected files in `docs/` to reflect the changes made.
2. **Update tests** — Add or update unit tests in `tests/` to cover the new or modified functionality. Ensure all existing tests still pass.
3. **Update examples and tools** — Review and update any affected scripts in `examples/`, `tools/`, and `notebooks/` to stay consistent with the changes.
4. **Update changelog** — Add an entry to `CHANGELOG.md` under the current unreleased version describing what was added, changed, or fixed.

## Config Conventions

- **Config types are Pydantic V2 `BaseModel` classes** with `frozen=True` and `extra="forbid"`.
- Always use **attribute access**: `config.key` (not `config["key"]`).
- Defaults are declared in the model field definition: `Field(default=...)` or `Field(default_factory=...)`.
- Mutations use `config.model_copy(update={...})` — never assign to attributes.
- `CerberusConfig.model_config_` is used to access `ModelConfig` (Pydantic reserves `model_config` for its own `ConfigDict`).
- Validation happens at model construction time — no separate `validate_*` calls.
- Serialization: `config.model_dump(mode="json")` replaces `_sanitize_config()`.
- `count_pseudocount` lives on `ModelConfig` in scaled units (not `DataConfig`).
- Use `model_construct()` in tests to skip validation when paths don't exist.

## Logging Conventions

- Every module with I/O, initialization, or file loading must use `import logging` and `logger = logging.getLogger(__name__)` at module level.
- Use `logger.info()` for significant lifecycle events: loading files into memory, completing initialization, setup summaries.
- Use `logger.debug()` for detailed operational info: lazy-loading, file routing decisions, interval counts, fallback-to-zeros.
- Pure computation modules (loss, metrics, layers, transforms, complexity) do **not** need logging.
- Never use `print()` for diagnostics in library code — always use `logger`.

## Run and fix all tests (after each task)

- pytest -v tests/

## Run and fix pyright (after each task)

- npx pyright tests/ src/

## Documentation (mkdocs)

- Public docs live in `docs/` as plain markdown. Nav structure is in `mkdocs.yml`.
- Files in `docs/internal/` are excluded from the published site.
- After changing docs, regenerate LLM context files: `python tools/generate_llms_txt.py`
- Preview locally: `mkdocs serve` (live-reload at `http://127.0.0.1:8000/cerberus/`)
- To add a new page: create a `.md` file in `docs/`, then add it to `nav:` in `mkdocs.yml`.

### Deployment

- **CI/CD is configured.** Docs auto-deploy when `docs/`, `mkdocs.yml`, or `pyproject.toml` change on `main` (via `.github/workflows/docs.yml`).
- PRs touching docs get a build check (`mkdocs build --strict`) without deploying.
- Manual deploy is also possible: `mkdocs gh-deploy` (from any branch).
- Site URL: `https://mctp.github.io/cerberus/`
- GitHub Pages source branch is `gh-pages` in repo Settings → Pages.

## DONT DO EMBARASSING THINGS

