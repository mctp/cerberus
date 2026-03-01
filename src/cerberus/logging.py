import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures the ``cerberus`` logger hierarchy.

    Sets up a StreamHandler on the ``cerberus`` library logger so that all
    ``cerberus.*`` module loggers emit formatted output.  Disables propagation
    to the root logger so that messages are not duplicated when the host
    application (e.g. PyTorch Lightning, pytest) installs its own root handler.

    Checks if handlers already exist to avoid duplication.

    Args:
        level: Logging level (default: logging.INFO)
    """
    logger = logging.getLogger("cerberus")

    # Avoid adding handlers if they already exist (e.g. from a previous call).
    # Check logger.handlers directly (not hasHandlers()) to avoid counting
    # parent/root handlers which are not ours.
    if logger.handlers:
        logger.setLevel(level)
        return

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
