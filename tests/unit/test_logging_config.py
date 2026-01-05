"""Unit tests for logging configuration."""

import logging

import pytest

from llm_assignment.logging_config import setup_logging


def test_setup_logging_basic():
    """Test setup_logging with basic level."""
    setup_logging(level="DEBUG")
    logger = logging.getLogger("llm_assignment")
    assert logger.getEffectiveLevel() == logging.DEBUG


def test_setup_logging_invalid_level():
    """Test setup_logging with invalid level raises ValueError."""
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging(level="INVALID")


def test_setup_logging_with_file(temp_output_dir):
    """Test setup_logging with a log file."""
    log_file = temp_output_dir / "test.log"
    setup_logging(level="INFO", log_file=log_file)

    logger = logging.getLogger()
    # Check that a FileHandler was added
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert has_file_handler
    assert log_file.exists()
