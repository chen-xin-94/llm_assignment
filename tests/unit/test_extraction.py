"""Unit tests for extraction module."""

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from llm_assignment.data_engine.extraction import clean_pdf_text
from llm_assignment.data_engine.extraction import extract_text_from_pdf


def test_clean_pdf_text(mock_pdf_content):
    """Test PDF text cleaning."""
    cleaned = clean_pdf_text(mock_pdf_content)

    # Check that headers and footers are removed
    assert "CORPORATE COMMUNICATIONS" not in cleaned
    assert "Media Information" not in cleaned
    assert "Media Contact" not in cleaned
    assert "The BMW Group" not in cleaned

    # Check content retention
    assert "The New BMW Model" in cleaned
    assert "Munich. This is a paragraph about the new model." in cleaned

    # Check formatting
    assert "\n\n" in cleaned  # Paragraph breaks preserves
    assert "lines." in cleaned


def test_hyphen_reassembly():
    """Test fixing hyphenated words across lines."""
    text = "This is a beauti-\nful car."
    cleaned = clean_pdf_text(text)
    assert "beautiful car" in cleaned


def test_extract_text_from_pdf():
    """Test extraction with mocked PDF reader."""
    with patch("llm_assignment.data_engine.extraction.PdfReader") as mock_reader:
        mock_instance = mock_reader.return_value
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 Content"
        mock_instance.pages = [mock_page]

        text = extract_text_from_pdf(Path("dummy.pdf"))

        assert text == "Page 1 Content"
        mock_reader.assert_called_once()


def test_extract_text_failure():
    """Test extraction failure handling."""
    with patch("llm_assignment.data_engine.extraction.PdfReader", side_effect=Exception("Read error")):
        text = extract_text_from_pdf(Path("bad.pdf"))
        assert text == ""
