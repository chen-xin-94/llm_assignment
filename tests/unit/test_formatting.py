"""Unit tests for formatting module."""

from llm_assignment.data_engine.formatting import format_for_qwen3
from llm_assignment.data_engine.formatting import read_processed_text
from llm_assignment.data_engine.formatting import save_processed_text


def test_format_for_qwen3_instruct():
    """Test instruct formatting."""
    title = "Test Article"
    content = "Some content."
    formatted = format_for_qwen3(title, content, style="instruct")

    assert "<|im_start|>system" in formatted
    assert "BMW automotive expert" in formatted
    assert f"Tell me about: {title}" in formatted
    assert content in formatted
    assert "<|im_end|>" in formatted


def test_format_for_qwen3_article():
    """Test article formatting."""
    title = "Test Article"
    content = "Some content."
    formatted = format_for_qwen3(title, content, style="article")

    assert f"### BMW Press Release: {title}" in formatted
    assert content in formatted


def test_save_and_read_processed_text(temp_output_dir):
    """Test saving and reading processed text files."""
    file_path = temp_output_dir / "test.txt"
    title = "My Title"
    date = "2024-01-01"
    source = "source.pdf"
    content = "Body content."
    reasoning = "Because it is good."

    save_processed_text(file_path, title=title, date=date, source=source, content=content, reasoning=reasoning)

    assert file_path.exists()

    # Read back
    r_title, r_date, r_source, r_content = read_processed_text(file_path)

    assert r_title == title
    assert r_date == date
    assert r_source == source
    assert r_content == content


def test_read_malformed_file(temp_output_dir):
    """Test reading a file without proper metadata headers."""
    file_path = temp_output_dir / "malformed.txt"
    content = "Just some text."
    file_path.write_text(content, encoding="utf-8")

    r_title, r_date, r_source, r_content = read_processed_text(file_path)

    assert r_content == content
    assert r_title == "malformed"  # Falls back to filename STEM


def test_format_for_qwen3_qa():
    """Test Q&A formatting."""
    title = "BMW i4"
    content = "The BMW i4 is an electric sedan.\n\nIt has great range."
    formatted = format_for_qwen3(title, content, style="qa")

    assert "<|im_start|>user" in formatted
    assert "What is the latest news about BMW i4?" in formatted
    assert "The BMW i4 is an electric sedan." in formatted
    assert "<|im_end|>" in formatted


def test_format_for_qwen3_fallback():
    """Test fallback for unknown formatting style."""
    content = "Raw content."
    formatted = format_for_qwen3("Title", content, style="unknown")
    assert formatted == content
