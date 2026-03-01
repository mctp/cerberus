"""Coverage tests for cerberus.logging — untested code paths."""
import logging
from cerberus.logging import setup_logging


class TestSetupLogging:

    def test_handlers_already_exist(self):
        """When handlers already exist, setup_logging only updates the level."""
        cerberus_logger = logging.getLogger("cerberus")
        original_handlers = list(cerberus_logger.handlers)

        # Ensure at least one handler exists on the cerberus logger
        if not cerberus_logger.handlers:
            cerberus_logger.addHandler(logging.NullHandler())

        handler_count_before = len(cerberus_logger.handlers)
        setup_logging(level=logging.WARNING)
        # No new handler should be added
        assert len(cerberus_logger.handlers) == handler_count_before
        assert cerberus_logger.level == logging.WARNING

        # Restore
        cerberus_logger.setLevel(logging.WARNING)
        cerberus_logger.handlers = original_handlers

    def test_custom_log_level(self):
        """setup_logging with custom level sets it on the cerberus logger."""
        cerberus_logger = logging.getLogger("cerberus")
        original_handlers = list(cerberus_logger.handlers)
        original_level = cerberus_logger.level

        # Remove handlers to trigger full setup
        cerberus_logger.handlers = []
        try:
            setup_logging(level=logging.DEBUG)
            assert cerberus_logger.level == logging.DEBUG
            assert len(cerberus_logger.handlers) == 1
        finally:
            # Restore
            cerberus_logger.handlers = original_handlers
            cerberus_logger.setLevel(original_level)
