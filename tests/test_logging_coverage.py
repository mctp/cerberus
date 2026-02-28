"""Coverage tests for cerberus.logging — untested code paths."""
import logging
from cerberus.logging import setup_logging


class TestSetupLogging:

    def test_handlers_already_exist(self):
        """When handlers already exist, setup_logging only updates the level."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        original_count = len(original_handlers)

        # Ensure at least one handler exists
        if not root.hasHandlers():
            root.addHandler(logging.NullHandler())

        handler_count_before = len(root.handlers)
        setup_logging(level=logging.WARNING)
        # No new handler should be added
        assert len(root.handlers) == handler_count_before
        assert root.level == logging.WARNING

        # Restore
        root.setLevel(logging.WARNING)
        root.handlers = original_handlers

    def test_custom_log_level(self):
        """setup_logging with custom level sets it on the root logger."""
        root = logging.getLogger()
        original_handlers = list(root.handlers)
        original_level = root.level

        # Remove handlers to trigger full setup
        root.handlers = []
        try:
            setup_logging(level=logging.DEBUG)
            assert root.level == logging.DEBUG
            assert len(root.handlers) == 1
        finally:
            # Restore
            root.handlers = original_handlers
            root.setLevel(original_level)
