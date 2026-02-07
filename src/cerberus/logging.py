import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures the root logger for Cerberus.
    
    Sets up a StreamHandler with a standard format.
    Checks if handlers already exist to avoid duplication.
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    root_logger = logging.getLogger()
    
    # Avoid adding handlers if they already exist (e.g. from another library or previous call)
    if root_logger.hasHandlers():
        # Update level just in case
        root_logger.setLevel(level)
        return

    root_logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    root_logger.addHandler(handler)
