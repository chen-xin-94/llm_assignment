"""Pytest configuration."""

from pathlib import Path
import sys

import pytest

# Add src to path so we can import llm_assignment
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))


@pytest.fixture
def mock_pdf_content():
    """Return a mock PDF text content."""
    return """
    CORPORATE COMMUNICATIONS

    Media Information
    01 January 2024

    The New BMW Model

    Munich. This is a paragraph about the new model.
    It breaks across lines.

    Media Contact.
    Name
    Phone: +49 123 456

    The BMW Group
    With its four brands BMW, MINI, Rolls-Royce and BMW Motorrad...
    """


@pytest.fixture
def temp_output_dir(tmp_path):
    """Return a temporary directory for output."""
    d = tmp_path / "output"
    d.mkdir()
    return d
