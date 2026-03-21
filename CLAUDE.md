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

- **No CI/CD is configured.** Docs deployment is manual.
- Deploy: `mkdocs gh-deploy --force` (builds site and pushes to `gh-pages` branch).
- Site URL: `https://mctp.github.io/cerberus/`
- GitHub Pages source branch must be set to `gh-pages` in repo Settings → Pages.
- To automate, add `.github/workflows/docs.yml` with the official mkdocs-material workflow:

```yaml
name: docs
on:
  push:
    branches: [main]
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure Git Credentials
        run: |
          git config user.name github-actions[bot]
          git config user.email 41898282+github-actions[bot]@users.noreply.github.com
      - uses: actions/setup-python@v5
        with:
          python-version: 3.x
      - run: echo "cache_id=$(date --utc '+%V')" >> $GITHUB_ENV
      - uses: actions/cache@v4
        with:
          key: mkdocs-material-${{ env.cache_id }}
          path: ~/.cache
          restore-keys: |
            mkdocs-material-
      - run: pip install mkdocs-material mkdocs-git-revision-date-localized-plugin
      - run: mkdocs gh-deploy --force
```

## DONT DO EMBARASSING THINGS

