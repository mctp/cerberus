import logging
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


SeqlogoMode = Literal["attribution", "probability", "ic"]


def _to_numpy(attrs) -> np.ndarray:
    """Accept numpy array or torch tensor; return a 2-D numpy array."""
    if hasattr(attrs, "detach") and hasattr(attrs, "cpu"):
        attrs = attrs.detach().cpu().numpy()
    return np.asarray(attrs)


def _require_extras(*pkgs: str) -> None:
    """Raise ImportError with an install hint if any of ``pkgs`` is missing."""
    missing = []
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"cerberus.plots requires {missing!r}. "
            f"Install with: pip install 'cerberus[extras]'"
        )


def _apply_seqlogo_mode(attrs: np.ndarray, mode: SeqlogoMode) -> np.ndarray:
    """Transform raw ``(A, L)`` attributions per ``mode``.

    * ``'attribution'``: unchanged — letter heights are signed attribution values.
    * ``'probability'``: column-wise softmax over the alphabet axis.
    * ``'ic'``: probability × per-position information content (bits).
    """
    if mode == "attribution":
        return attrs
    # Stabilized softmax over the alphabet axis.
    shifted = attrs - attrs.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)
    if mode == "probability":
        return probs
    if mode == "ic":
        n_channels = attrs.shape[0]
        max_entropy = np.log2(n_channels)
        log2_p = np.log2(probs + 1e-12)
        ic = np.clip(max_entropy + (probs * log2_p).sum(axis=0), 0.0, max_entropy)
        return probs * ic[np.newaxis, :]
    raise ValueError(
        f"Unsupported mode: {mode!r}. Must be 'attribution', 'probability', or 'ic'."
    )


def plot_seqlogo(
    ax,
    attrs,
    *,
    alphabet: str = "ACGT",
    mode: SeqlogoMode = "attribution",
    color_scheme: str | dict | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draw a stacked sequence logo of ``attrs`` onto ``ax``.

    Letter heights encode ``mode``-transformed values; positive contributions
    stack up from zero, negative contributions stack down (via ``logomaker``).

    Parameters
    ----------
    ax:
        A ``matplotlib.axes.Axes`` to draw into.
    attrs:
        ``(A, L)`` attribution matrix — numpy array or torch tensor.
    alphabet:
        Letter set; ``"ACGT"`` for DNA, ``"ACGU"`` for RNA, or the 20-letter
        amino acid alphabet. Length must match ``attrs.shape[0]``.
    mode:
        ``'attribution'`` (default) — raw signed attribution heights.
        ``'probability'`` — column-wise softmax over alphabet (heights sum to 1).
        ``'ic'`` — probabilities scaled by per-position information content
        (heights sum to [0, log2(|alphabet|)]).
    color_scheme:
        Passed through to ``logomaker.Logo(color_scheme=...)``. ``None`` lets
        logomaker pick a default per alphabet.
    ylim:
        Optional ``(ymin, ymax)`` override. ``None`` (default) lets matplotlib
        auto-scale from the data.
    """
    _require_extras("logomaker", "pandas")
    import logomaker  # type: ignore[import-untyped]
    import pandas as pd

    attrs = _to_numpy(attrs)
    if attrs.ndim != 2:
        raise ValueError(f"attrs must be 2-D (alphabet, length); got shape {attrs.shape}")
    n_channels, _ = attrs.shape
    if n_channels != len(alphabet):
        raise ValueError(
            f"attrs has {n_channels} channels but alphabet {alphabet!r} has "
            f"{len(alphabet)} letters."
        )

    values = _apply_seqlogo_mode(attrs, mode)

    # logomaker expects (position, letter) orientation.
    df = pd.DataFrame(data=values.T, columns=list(alphabet))  # type: ignore[arg-type]

    kwargs: dict = {"ax": ax}
    if color_scheme is not None:
        kwargs["color_scheme"] = color_scheme
    logomaker.Logo(df, **kwargs)

    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_attribution_heatmap(
    ax,
    attrs,
    *,
    alphabet: str = "ACGT",
    vlim: float | None = None,
    cmap: str = "coolwarm",
):
    """Draw an ``(A, L)`` attribution heatmap onto ``ax``.

    Uses a symmetric color scale centred at zero so positive / negative
    contributions are visually comparable. Returns the ``AxesImage`` so the
    caller can attach a colorbar.

    Parameters
    ----------
    ax:
        A ``matplotlib.axes.Axes`` to draw into.
    attrs:
        ``(A, L)`` attribution matrix — numpy array or torch tensor.
    alphabet:
        Letter set; rows are labelled with its characters.
    vlim:
        Half-width of the symmetric color range. ``None`` (default) uses
        ``max(|attrs|)`` so the strongest signal sets full saturation.
    cmap:
        Matplotlib colormap name; default ``"coolwarm"`` is diverging.

    Returns
    -------
    matplotlib.image.AxesImage
    """
    _require_extras("matplotlib")

    attrs = _to_numpy(attrs)
    if attrs.ndim != 2:
        raise ValueError(f"attrs must be 2-D (alphabet, length); got shape {attrs.shape}")
    n_channels, _ = attrs.shape
    if n_channels != len(alphabet):
        raise ValueError(
            f"attrs has {n_channels} channels but alphabet {alphabet!r} has "
            f"{len(alphabet)} letters."
        )

    if vlim is None:
        vlim = float(np.abs(attrs).max())
        if vlim == 0:
            vlim = 1.0

    img = ax.imshow(
        attrs,
        aspect="auto",
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(alphabet)))
    ax.set_yticklabels(list(alphabet))
    return img


def plot_attribution_panel(
    fig,
    attrs,
    *,
    alphabet: str = "ACGT",
    heatmap: bool = True,
    logo_mode: SeqlogoMode = "attribution",
    height_ratios: tuple[int, int] = (10, 5),
):
    """Composite: sequence logo on top, optional attribution heatmap below.

    Reproduces the stacked "logo + heatmap" layout from Sasse et al. 2024
    Fig. 1B using :func:`plot_seqlogo` and :func:`plot_attribution_heatmap`
    as primitives. Caller owns ``fig``; this helper just adds subplots.

    Parameters
    ----------
    fig:
        A ``matplotlib.figure.Figure`` to populate.
    attrs:
        ``(A, L)`` attribution matrix — numpy array or torch tensor.
    alphabet:
        Letter set; default ``"ACGT"``.
    heatmap:
        If ``True`` (default), add a heatmap below the logo. If ``False``,
        only draw the logo.
    logo_mode:
        Passed through to :func:`plot_seqlogo`.
    height_ratios:
        Relative heights of ``(logo, heatmap)`` when ``heatmap=True``.
        Ignored when ``heatmap=False``.

    Returns
    -------
    tuple[Axes, Axes | None]
        ``(logo_ax, heatmap_ax)`` — ``heatmap_ax`` is ``None`` when
        ``heatmap=False``. Both are returned so the caller can set titles,
        xticks, or attach a colorbar.
    """
    _require_extras("matplotlib")

    if heatmap:
        gs = fig.add_gridspec(2, 1, height_ratios=list(height_ratios), hspace=0.1)
        logo_ax = fig.add_subplot(gs[0])
        heatmap_ax = fig.add_subplot(gs[1], sharex=logo_ax)
        plot_seqlogo(logo_ax, attrs, alphabet=alphabet, mode=logo_mode)
        plot_attribution_heatmap(heatmap_ax, attrs, alphabet=alphabet)
        logo_ax.tick_params(labelbottom=False)
        return logo_ax, heatmap_ax

    logo_ax = fig.add_subplot(1, 1, 1)
    plot_seqlogo(logo_ax, attrs, alphabet=alphabet, mode=logo_mode)
    return logo_ax, None


def save_count_scatter(
    pred_log_counts: np.ndarray,
    target_log_counts: np.ndarray,
    save_dir: str | Path,
    epoch: int,
    x_label: str = "True log counts",
    y_label: str = "Predicted log counts",
    title: str = "Val counts",
    filename_prefix: str = "val_count_scatter",
) -> None:
    """
    Generate and save a hexbin scatter of predicted vs. true log counts.

    Produces ``<filename_prefix>_epoch_NNN.png`` inside ``save_dir/plots/``.
    Silently skips if matplotlib is not installed.

    Args:
        pred_log_counts: 1-D array of predicted log counts for all validation examples.
        target_log_counts: 1-D array of true log counts for all validation examples.
        save_dir: Root directory under which the ``plots/`` subdirectory is created.
        epoch: Current epoch index, used in the filename and plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Plot title prefix (``epoch`` is appended automatically).
        filename_prefix: Output filename prefix under ``save_dir/plots/``.
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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title} — epoch {epoch}")
    fig.colorbar(hb, ax=ax, label="n")

    plot_dir = Path(save_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig_path = plot_dir / f"{filename_prefix}_epoch_{epoch:03d}.png"
    fig.savefig(fig_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved count scatter plot to {fig_path}")
