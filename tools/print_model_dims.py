import argparse
import importlib
import inspect
import pkgutil
import sys
from dataclasses import is_dataclass

import torch
import torch.nn as nn

# Import cerberus.models to inspect available models
import cerberus.models


def find_model_class(model_name):
    """
    Finds a model class in cerberus.models by name (case-insensitive).
    """
    # 1. Check directly exposed models in cerberus.models
    if hasattr(cerberus.models, model_name):
        return getattr(cerberus.models, model_name)

    # 2. Iterate over all submodules
    path = cerberus.models.__path__
    prefix = cerberus.models.__name__ + "."

    for _, name, _ in pkgutil.iter_modules(path, prefix):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue

        # Case insensitive match
        for attr_name in dir(module):
            if attr_name.lower() == model_name.lower():
                return getattr(module, attr_name)

    return None


def get_shape_hook(name):
    def hook(model, input, output):
        # Format the output shape(s)
        if isinstance(output, (tuple, list)):
            shapes = []
            for o in output:
                if hasattr(o, "shape"):
                    shapes.append(tuple(o.shape))
                else:
                    shapes.append(type(o).__name__)
            print(f"{name}: {shapes}")
        elif is_dataclass(output):
            shapes = {}
            for field in output.__dataclass_fields__:
                val = getattr(output, field)
                if hasattr(val, "shape"):
                    shapes[field] = tuple(val.shape)
                else:
                    shapes[field] = type(val).__name__
            print(f"{name}: {shapes}")
        elif hasattr(output, "shape"):
            print(f"{name}: {tuple(output.shape)}")
        else:
            print(f"{name}: {type(output).__name__}")

    return hook


def print_model_dims(model_name, input_len=None, output_len=None, verbose=False):
    model_cls = find_model_class(model_name)
    if model_cls is None:
        print(f"Error: Model '{model_name}' not found in cerberus.models.")
        sys.exit(1)

    # Inspect signature to resolve defaults if not provided
    sig = inspect.signature(model_cls.__init__)

    final_input_len = input_len
    final_output_len = output_len

    # Resolve input_len
    if final_input_len is None:
        param = sig.parameters.get("input_len")
        if param and param.default != inspect.Parameter.empty:
            final_input_len = param.default
            print(f"Using default input_len from class: {final_input_len}")
        else:
            final_input_len = 2112
            print(
                f"Warning: input_len not provided and no default found. Using fallback: {final_input_len}"
            )

    # Resolve output_len
    if final_output_len is None:
        param = sig.parameters.get("output_len")
        if param and param.default != inspect.Parameter.empty:
            final_output_len = param.default
            print(f"Using default output_len from class: {final_output_len}")
        else:
            final_output_len = 1024
            print(
                f"Warning: output_len not provided and no default found. Using fallback: {final_output_len}"
            )

    print(f"Initializing {model_cls.__name__}...")
    try:
        # Most Cerberus models accept input_len and output_len
        model = model_cls(input_len=final_input_len, output_len=final_output_len)
    except Exception as e:
        print(f"Error initializing model with explicit lengths: {e}")
        print("Attempting default initialization...")
        try:
            model = model_cls()
            # Try to update lengths from model attributes if they differ
            if hasattr(model, "input_len"):
                final_input_len = model.input_len
            if hasattr(model, "output_len"):
                final_output_len = model.output_len
        except Exception as e2:
            print(f"Failed to initialize model: {e2}")
            sys.exit(1)

    print(f"Input Length: {final_input_len}")
    print(f"Output Length: {final_output_len}")
    print("-" * 50)

    # Create dummy input: (Batch=1, Channels=4, Length)
    x = torch.randn(1, 4, final_input_len)

    # Register hooks
    # We register on named children to get a high-level view (Stem, Body, Heads)
    # If a child is a container (Sequential/ModuleList), we optionally inspect inside.

    # Heuristic:
    # Register on all immediate named children.
    # If a child is a container, also register on its children?
    # For now, let's just do a recursive walk if verbose, otherwise just top-level children + 1 level?

    # Let's mirror the behavior of the original script which was quite detailed but structured.
    # It printed: Stem, Layer 1..N, Profile Pointwise, Act, Spatial, Count MLP layers.

    def register_hooks_recursive(module, prefix=""):
        # Helper to decide if we should recurse or hook
        # We hook everything that is a leaf OR a container that computes something meaningful?
        # Actually hooks on containers fire after children.

        # Strategy: iterate named_children. Hook them.
        # If child is Sequential or ModuleList, recurse.

        for name, child in module.named_children():
            full_name = f"{prefix}{name}"

            # Hook this module
            child.register_forward_hook(get_shape_hook(full_name))

            # If it's a container, recurse to see internal structure
            if isinstance(child, (nn.Sequential, nn.ModuleList)):
                register_hooks_recursive(child, prefix=f"{full_name}.")
            # Special case for Pomeranian/BPNet:
            # PGCBlock, DilatedResidualBlock are modules but we might want to see inside if verbose?
            # For now, stick to top-level containers and their immediate children (if containers).

    # Simple approach: Hook all immediate children.
    # If an immediate child is a List/Sequential, hook its items.
    for name, child in model.named_children():
        child.register_forward_hook(get_shape_hook(name))

        if isinstance(child, (nn.ModuleList, nn.Sequential)):
            for _i, (sub_name, sub_child) in enumerate(child.named_children()):
                # named_children for list/sequential returns "0", "1", etc.
                child_name = f"{name}.{sub_name}"
                sub_child.register_forward_hook(get_shape_hook(child_name))

    print("Running Forward Pass...")
    model.eval()
    with torch.no_grad():
        try:
            output = model(x)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            sys.exit(1)

    print("-" * 50)
    print("Final Output:")
    if is_dataclass(output):
        for field in output.__dataclass_fields__:
            val = getattr(output, field)
            if hasattr(val, "shape"):
                print(f"  {field}: {tuple(val.shape)}")
            else:
                print(f"  {field}: {val}")
    else:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print dimensions of a Cerberus model."
    )
    parser.add_argument(
        "model", type=str, help="Name of the model class (e.g., Pomeranian, BPNet)"
    )
    parser.add_argument(
        "--input_len",
        type=int,
        default=None,
        help="Input sequence length (default: inferred from model or 2112)",
    )
    parser.add_argument(
        "--output_len",
        type=int,
        default=None,
        help="Output sequence length (default: inferred from model or 1024)",
    )
    # parser.add_argument("--verbose", action="store_true", help="Print more detailed layer information")

    args = parser.parse_args()
    print_model_dims(args.model, args.input_len, args.output_len)
