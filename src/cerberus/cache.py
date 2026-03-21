"""
Utilities for prepare_data() caching.

Provides deterministic cache directory resolution, serialization, and loading
of precomputed data (e.g. complexity metrics) to avoid redundant computation
across DDP ranks and training runs.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import numpy as np

from .config import SamplerConfig

logger = logging.getLogger(__name__)


def get_default_cache_dir() -> Path:
    """
    Returns the default cerberus cache directory.

    Uses $XDG_CACHE_HOME/cerberus if XDG_CACHE_HOME is set,
    otherwise falls back to ~/.cache/cerberus.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "cerberus"


def resolve_cache_dir(
    cache_dir: Path,
    fasta_path: str | Path,
    sampler_config: SamplerConfig,
    seed: int,
    chrom_sizes: dict[str, int],
) -> Path:
    """
    Computes a deterministic cache subdirectory from config inputs.

    The subdirectory name is a truncated SHA-256 hash of the serialized
    inputs, ensuring that different configs produce different cache paths.

    Args:
        cache_dir: Base cache directory.
        fasta_path: Path to genome FASTA file.
        sampler_config: Sampler configuration dictionary.
        seed: Random seed (always an int; CerberusDataModule auto-generates one if not provided).
        chrom_sizes: Chromosome sizes dictionary.

    Returns:
        Path to the config-specific cache subdirectory.
    """
    fasta_path = str(fasta_path)
    key_data = json.dumps({
        "fasta_path": fasta_path,
        "fasta_mtime": os.path.getmtime(fasta_path),
        "sampler_config": sampler_config.model_dump(mode="json"),
        "seed": seed,
        "chrom_sizes": chrom_sizes,
    }, sort_keys=True)
    h = hashlib.sha256(key_data.encode()).hexdigest()[:16]
    return cache_dir / h


def save_prepare_cache(cache_dir: Path, cache: dict[str, np.ndarray]) -> None:
    """
    Serializes a prepare_data cache dict to disk.

    Args:
        cache_dir: Directory to write cache files into.
        cache: Dictionary mapping interval string keys to metric arrays.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(cache.keys()))
    values = np.array(list(cache.values()))
    cache_path = cache_dir / "metrics_cache.npz"
    np.savez_compressed(cache_path, keys=keys, values=values)
    (cache_dir / "ready").touch()
    logger.info(f"Saved {len(cache)} cache entries to {cache_dir}")


def load_prepare_cache(cache_dir: Path) -> dict[str, np.ndarray] | None:
    """
    Loads a prepare_data cache from disk if available.

    Returns the cache dict if a valid cache exists (metrics_cache.npz + ready
    sentinel), or None if no cache is available.

    Args:
        cache_dir: Directory containing cache files.

    Returns:
        Cache dict mapping interval keys to metric arrays, or None.
    """
    if not (cache_dir / "ready").exists():
        return None
    cache_path = cache_dir / "metrics_cache.npz"
    if not cache_path.exists():
        return None

    logger.info(f"Loading prepare_data cache from {cache_dir}")
    data = np.load(cache_path, allow_pickle=False)
    keys = data["keys"]
    values = data["values"]
    cache = {str(k): v for k, v in zip(keys, values)}
    logger.info(f"Loaded {len(cache)} cached entries")
    return cache
