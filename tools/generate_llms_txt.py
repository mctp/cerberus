#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt for LLM consumption.

llms.txt  — compact index: project summary + doc links + API signatures
llms-full.txt — full dump: all docs concatenated + all source files

Run from any directory:
    python tools/generate_llms_txt.py
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
SRC_DIR = ROOT / "src" / "cerberus"
SITE_URL = "https://mctp.github.io/cerberus"

# Public docs in nav order (title, filename)
NAV_PAGES = [
    ("Home", "README.md"),
    ("Overview", "overview.md"),
    ("Quick Start", "usage.md"),
    ("Workflow", "workflow.md"),
    ("Configuration", "configuration.md"),
    ("Samplers", "samplers.md"),
    ("Core Components", "components.md"),
    ("Prediction", "prediction.md"),
    ("Multi-GPU", "multi_gpu.md"),
    ("Maintaining Docs", "contributing.md"),
    ("Models", "models.md"),
    ("Complexity Metrics", "complexity.md"),
    ("Codebase Structure", "structure.md"),
    ("Model Comparison (K5 vs K9)", "model_comparison_k5_vs_k9.md"),
]

# Source files in logical reading order
SRC_FILES = [
    "__init__.py",
    "config.py",
    "interval.py",
    "genome.py",
    "sequence.py",
    "signal.py",
    "mask.py",
    "complexity.py",
    "samplers.py",
    "transform.py",
    "dataset.py",
    "datamodule.py",
    "module.py",
    "train.py",
    "output.py",
    "loss.py",
    "metrics.py",
    "layers.py",
    "model_ensemble.py",
    "predict_bigwig.py",
    "download.py",
    "logging.py",
    "models/__init__.py",
    "models/gopher.py",
    "models/bpnet.py",
    "models/pomeranian.py",
    "models/geminet.py",
    "models/lyra.py",
    "models/asap.py",
]


def _first_docstring_line(node: ast.AST) -> str:
    """Return the first line of a docstring, or empty string."""
    if (
        isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value.strip().splitlines()[0]
    return ""


def extract_api(filepath: Path) -> list[dict]:
    """Extract top-level public classes and functions from a Python source file."""
    try:
        tree = ast.parse(filepath.read_text())
    except SyntaxError:
        return []

    items = []
    for node in tree.body:  # top-level only
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute) and isinstance(b.value, ast.Name):
                    bases.append(f"{b.value.id}.{b.attr}")
            items.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "bases": f"({', '.join(bases)})" if bases else "",
                    "doc": _first_docstring_line(node),
                }
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            items.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "doc": _first_docstring_line(node),
                }
            )

    return [i for i in items if not i["name"].startswith("_")]


def _page_url(filename: str) -> str:
    if filename == "README.md":
        return f"{SITE_URL}/"
    return f"{SITE_URL}/{filename.removesuffix('.md')}/"


def generate_llms_txt() -> str:
    out = []

    out.append("# Cerberus")
    out.append("")
    out.append(
        "> A PyTorch-based framework for genomic sequence-to-function (S2F) model training."
    )
    out.append("")
    out.append(
        "Cerberus provides modular data loading for genomic intervals, DNA sequences (FASTA), "
        "and functional signal tracks (BigWig/BigBed). It supports multiple neural network "
        "architectures (BPNet, GemiNet, LyraNet, GlobalProfileCNN) with PyTorch Lightning "
        "integration for training, cross-validation, and genome-wide inference."
    )
    out.append("")

    # Documentation index
    out.append("## Documentation")
    out.append("")
    for title, filename in NAV_PAGES:
        out.append(f"- [{title}]({_page_url(filename)})")
    out.append("")

    # API signatures
    out.append("## API Reference")
    out.append("")

    for rel_path in SRC_FILES:
        filepath = SRC_DIR / rel_path
        if not filepath.exists():
            continue
        items = extract_api(filepath)
        if not items:
            continue

        module = rel_path.replace("/", ".").removesuffix(".py")
        module = "cerberus" if module == "__init__" else f"cerberus.{module}"
        out.append(f"### `{module}`")
        out.append("")
        for item in items:
            if item["kind"] == "class":
                label = f"class {item['name']}{item['bases']}"
            else:
                label = f"{item['name']}()"
            doc = f" — {item['doc']}" if item["doc"] else ""
            out.append(f"- **`{label}`**{doc}")
        out.append("")

    return "\n".join(out)


def generate_llms_full_txt() -> str:
    SEP = "=" * 72
    out = []

    out.append("# Cerberus — Full Documentation and Source Code")
    out.append("")
    out.append(
        "This file contains the complete Cerberus documentation and source code "
        "concatenated for use as LLM context. Auto-generated by tools/generate_llms_txt.py."
    )
    out.append(f"Site: {SITE_URL}")
    out.append("")
    out.append(SEP)
    out.append("# DOCUMENTATION")
    out.append(SEP)

    for title, filename in NAV_PAGES:
        filepath = DOCS_DIR / filename
        if not filepath.exists():
            continue
        out.append("")
        out.append(f"## {title}  [{filename}]")
        out.append("")
        out.append(filepath.read_text().strip())

    out.append("")
    out.append(SEP)
    out.append("# SOURCE CODE")
    out.append(SEP)

    for rel_path in SRC_FILES:
        filepath = SRC_DIR / rel_path
        if not filepath.exists():
            continue
        out.append("")
        out.append(f"## src/cerberus/{rel_path}")
        out.append("")
        out.append(filepath.read_text().strip())

    return "\n".join(out)


def main() -> None:
    for name, fn in [
        ("llms.txt", generate_llms_txt),
        ("llms-full.txt", generate_llms_full_txt),
    ]:
        print(f"Generating {name}...", end=" ", flush=True)
        content = fn()
        dest = DOCS_DIR / name
        dest.write_text(content)
        print(f"{len(content):,} chars → {dest}")


if __name__ == "__main__":
    main()
