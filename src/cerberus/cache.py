"""
Utilities for prepare_data() caching.

Provides deterministic cache directory resolution, serialization, and loading
of precomputed data (e.g. complexity metrics) to avoid redundant computation
across DDP ranks and training runs.
"""

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .config import SamplerConfig

try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX platforms
    _HAS_FCNTL = False

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
        seed: Random seed. Always an int; ``CerberusDataModule`` uses a fixed
            default (``42``) when none is given, so two runs that do not both
            override it share the same cache key.
        chrom_sizes: Chromosome sizes dictionary.

    Returns:
        Path to the config-specific cache subdirectory.
    """
    fasta_path = str(fasta_path)
    key_data = json.dumps(
        {
            "fasta_path": fasta_path,
            "fasta_mtime": os.path.getmtime(fasta_path),
            "sampler_config": sampler_config.model_dump(mode="json"),
            "seed": seed,
            "chrom_sizes": chrom_sizes,
        },
        sort_keys=True,
    )
    h = hashlib.sha256(key_data.encode()).hexdigest()[:16]
    return cache_dir / h


@contextlib.contextmanager
def cache_build_lock(cache_dir: Path) -> Iterator[None]:
    """Best-effort exclusive lock that serializes cache *construction*.

    Among processes that share a cache directory — which happens by default for
    parallel cross-validation folds or a hyperparameter sweep that hash to the
    same config key (the key excludes the model, learning rate and fold, and the
    seed defaults to a fixed ``42``) — this ensures only one process computes and
    writes the cache while the others block, then fall through to read it. Use it
    together with a *re-check* of the ``ready`` sentinel after acquiring the
    lock (double-checked locking), e.g.::

        with cache_build_lock(cache_dir):
            if (cache_dir / "ready").exists():
                return            # built by another process while we waited
            ...                   # compute
            save_prepare_cache(cache_dir, metrics)

    The lock is advisory (``fcntl.flock``) and is released automatically when the
    file object is closed — including on process crash — so a dead holder never
    leaves a stale lock. On platforms / filesystems without ``flock`` support it
    degrades to a no-op: correctness still holds because
    :func:`save_prepare_cache` publishes atomically; the contended processes
    merely each recompute (Layer 1's guarantee), they cannot corrupt the file.

    Args:
        cache_dir: The config-specific cache subdirectory to guard.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not _HAS_FCNTL:  # pragma: no cover - non-POSIX platforms
        yield
        return
    lock_path = cache_dir / ".lock"
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError as exc:  # pragma: no cover - exotic network filesystems
            logger.debug(
                "cache_build_lock: flock unavailable (%s); proceeding without "
                "the lock — writes remain atomic but may be recomputed.",
                exc,
            )
        yield
        # The advisory lock is released when ``lock_file`` is closed here.


def save_prepare_cache(cache_dir: Path, cache: dict[str, np.ndarray]) -> None:
    """
    Serializes a prepare_data cache dict to disk **atomically**.

    The ``.npz`` payload is written to a unique temporary file inside
    ``cache_dir`` and then atomically moved into place with :func:`os.replace`.
    Because ``os.replace`` is atomic on POSIX within a single filesystem, a
    concurrent reader — or a second writer racing on the same config-hash
    directory (e.g. parallel cross-validation folds or a hyperparameter sweep
    sharing one cache key) — can never observe a partially written file: it
    sees either the previous complete file or the new complete one. The
    ``ready`` sentinel is touched only *after* the payload is fully published,
    so a reader gated on ``ready`` is always paired with a complete ``.npz``.

    Note: ``os.replace`` is atomic only within one filesystem, which is why the
    temporary file is created inside ``cache_dir`` rather than ``$TMPDIR``. The
    unique temp name lets two concurrent writers proceed without clobbering each
    other's in-progress file (last rename wins; both files are complete).

    Args:
        cache_dir: Directory to write cache files into.
        cache: Dictionary mapping interval string keys to metric arrays.

    Raises:
        ValueError: If the metric arrays are ragged (different lengths), which
            would serialize as an ``object`` array and fail to round-trip
            through ``np.load(..., allow_pickle=False)``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(cache.keys()))
    # Ragged metric arrays cannot round-trip through np.load(allow_pickle=False).
    # Newer numpy raises ValueError when stacking inhomogeneous rows; older
    # numpy silently builds an object array. Normalize both into a clear error.
    try:
        values = np.array(list(cache.values()))
    except ValueError as exc:
        raise ValueError(
            "metrics_cache has ragged rows; refusing to write a "
            "non-roundtrippable npz (all metric arrays must share a length)."
        ) from exc
    if values.dtype == object:
        raise ValueError(
            "metrics_cache has ragged rows; refusing to write a "
            "non-roundtrippable npz (all metric arrays must share a length)."
        )
    cache_path = cache_dir / "metrics_cache.npz"

    # Write to a unique temp file in the SAME directory, fsync, then rename.
    fd, tmp_name = tempfile.mkstemp(
        dir=cache_dir, prefix=".metrics_cache.", suffix=".npz.tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, keys=keys, values=values)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, cache_path)
    except BaseException:
        # Best-effort cleanup so a failed write does not leave a stray temp file.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

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
    cache = {str(k): v for k, v in zip(keys, values, strict=True)}
    logger.info(f"Loaded {len(cache)} cached entries")
    return cache
