import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_count_scatter(
    pred_log_counts: np.ndarray,
    target_log_counts: np.ndarray,
    save_dir: str | Path,
    epoch: int,
) -> None:
    """
    Generate and save a hexbin scatter of predicted vs. true log counts.

    Produces ``val_count_scatter_epoch_NNN.png`` inside ``save_dir/plots/``.
    Silently skips if matplotlib is not installed.

    Args:
        pred_log_counts: 1-D array of predicted log counts for all validation examples.
        target_log_counts: 1-D array of true log counts for all validation examples.
        save_dir: Root directory under which the ``plots/`` subdirectory is created.
        epoch: Current epoch index, used in the filename and plot title.
    """
    try:
        import matplotlib.pyplot as plt

        plt.switch_backend("agg")
    except ImportError:
        logger.debug("matplotlib not available; skipping count scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    hb = ax.hexbin(
        target_log_counts, pred_log_counts, gridsize=50, mincnt=1, cmap="viridis"
    )
    ax.set_xlabel("True log counts")
    ax.set_ylabel("Predicted log counts")
    ax.set_title(f"Val counts — epoch {epoch}")
    fig.colorbar(hb, ax=ax, label="n")

    plot_dir = Path(save_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"val_count_scatter_epoch_{epoch:03d}.png"
    fig.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved count scatter plot to {fig_path}")
