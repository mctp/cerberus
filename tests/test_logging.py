import logging

from cerberus.logging import setup_logging


def test_setup_logging_basic():
    """Test that setup_logging configures the cerberus logger."""
    logger = logging.getLogger("cerberus")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    setup_logging()

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_setup_logging_idempotent():
    """Test that setup_logging doesn't add duplicate handlers."""
    logger = logging.getLogger("cerberus")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    setup_logging()
    setup_logging()

    assert len(logger.handlers) == 1
    assert logger.propagate is False


def test_setup_logging_level():
    """Test setting a custom logging level."""
    logger = logging.getLogger("cerberus")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    setup_logging(level=logging.DEBUG)

    assert logger.level == logging.DEBUG
    assert logger.handlers[0].level == logging.DEBUG


def test_setup_logging_does_not_modify_root():
    """Regression: setup_logging must not add handlers to the root logger."""
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]

    cerberus_logger = logging.getLogger("cerberus")
    for handler in cerberus_logger.handlers[:]:
        cerberus_logger.removeHandler(handler)

    setup_logging()

    assert root_logger.handlers == original_handlers
