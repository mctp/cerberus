# Maintaining the Documentation

## Setup (one time)

```bash
pip install -e ".[docs]"
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

```bash
python tools/generate_llms_txt.py  # regenerate LLM context files
mkdocs gh-deploy
```

This builds the site and pushes it directly to the `gh-pages` branch on GitHub. The live site at `https://mctp.github.io/cerberus/` updates within ~1 minute.

!!! note
    You need push access to the `mctp/cerberus` repository to deploy.
