import logging
import pytest
from unittest.mock import patch, MagicMock
from cerberus.logging import setup_logging

def test_setup_logging_basic():
    """Test that setup_logging configures the root logger."""
    # Reset logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    setup_logging()
    
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], logging.StreamHandler)
    assert root_logger.level == logging.INFO

def test_setup_logging_idempotent():
    """Test that setup_logging doesn't add duplicate handlers."""
    # Reset logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    setup_logging()
    setup_logging()
    
    assert len(root_logger.handlers) == 1

def test_setup_logging_level():
    """Test setting a custom logging level."""
    # Reset logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    setup_logging(level=logging.DEBUG)
    
    assert root_logger.level == logging.DEBUG
    assert root_logger.handlers[0].level == logging.DEBUG
