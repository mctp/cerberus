# Maintaining the Documentation

## Setup (one time)

```bash
pip install -e ".[dev]"
```

## Editing docs

All public documentation lives in `docs/` as plain markdown files. Edit them directly — no special syntax required beyond standard markdown. The nav structure is defined in `mkdocs.yml`.

To add a new page:

1. Create a `.md` file in `docs/`
2. Add it to the `nav:` section of `mkdocs.yml`

Files in `docs/internal/` are excluded from the site and will not be published.

## Preview locally

```bash
mkdocs serve
```

Opens a live-reloading preview at `http://127.0.0.1:8000/cerberus/`. The browser refreshes automatically on every save.

## Deploy

Documentation is **automatically deployed** by GitHub Actions whenever changes to `docs/`, `mkdocs.yml`, or `pyproject.toml` are pushed to `main`. The workflow (`.github/workflows/docs.yml`) builds the site and pushes it to the `gh-pages` branch. The live site at `https://mctp.github.io/cerberus/` updates within ~1 minute.

For pull requests that touch docs, the CI runs a build check (`mkdocs build --strict`) without deploying.

You can also trigger a manual deploy from the Actions tab using the "workflow_dispatch" trigger.

### Before pushing

```bash
python tools/generate_llms_txt.py  # regenerate LLM context files
```

### Manual deploy (optional)

If you need to deploy without pushing to `main` (e.g. from a feature branch):

```bash
mkdocs gh-deploy
```

!!! note
    You need push access to the `mctp/cerberus` repository to deploy (manually or via CI).
