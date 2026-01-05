"""Formatting utility functions."""

from __future__ import annotations

from pathlib import Path

# Delimiters for preprocessed text files
METADATA_START = "--- METADATA ---"
REASONING_START = "--- REASONING ---"
CONTENT_START = "--- CONTENT ---"


def save_processed_text(
    path: Path,
    title: str,
    date: str | None,
    source: str,
    content: str,
    reasoning: str | None = None,
) -> None:
    """Save processed text with explicit metadata/content blocks."""
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{METADATA_START}\n")
        f.write(f"TITLE: {title}\n")
        f.write(f"DATE: {date if date else ''}\n")
        f.write(f"SOURCE: {source}\n")

        if reasoning:
            f.write(f"{REASONING_START}\n")
            f.write(f"{reasoning}\n")

        f.write(f"{CONTENT_START}\n")
        f.write(content)


def read_processed_text(path: Path) -> tuple[str, str | None, str, str]:
    """Read processed text file with metadata/content blocks.

    Returns:
        Tuple of (title, date, source, content)
    """
    with path.open(encoding="utf-8") as f:
        file_content = f.read()

    # Split by CONTENT_START to separate header/reasoning and content
    parts = file_content.split(CONTENT_START + "\n", 1)
    if len(parts) != 2:
        # Fallback for old format or malformed files
        return path.stem, None, path.name, file_content

    header_section, text_content = parts

    # Check for reasoning section in header part
    if REASONING_START in header_section:
        header_part, _ = header_section.split(REASONING_START, 1)
    else:
        header_part = header_section

    # Parse header lines
    title = ""
    date = None
    source = ""

    for line in header_part.split("\n"):
        if line.startswith("TITLE: "):
            title = line[7:].strip()
        elif line.startswith("DATE: "):
            date_str = line[6:].strip()
            date = date_str if date_str else None
        elif line.startswith("SOURCE: "):
            source = line[8:].strip()

    return title, date, source, text_content.strip()


def format_for_qwen3(title: str, content: str, style: str = "instruct") -> str:
    """Format article for Qwen3 training.

    Uses Qwen3's ChatML-style format:
    <|im_start|>system\n...<|im_end|>
    <|im_start|>user\n...<|im_end|>
    <|im_start|>assistant\n...<|im_end|>

    Args:
        title: Article title
        content: Article content
        style: Format style - 'instruct', 'article', or 'qa'

    Returns:
        Formatted text for training
    """
    # Clean title
    title = title.strip().rstrip(".")

    if style == "instruct":
        # Instruction-following format
        return (
            f"<|im_start|>system\n"
            f"You are a BMW automotive expert assistant. "
            f"Provide accurate, detailed information about BMW vehicles, technology, and company news.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Tell me about: {title}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{content}<|im_end|>"
        )

    if style == "article":
        # Simple article continuation format
        return f"### BMW Press Release: {title}\n\n{content}"

    if style == "qa":
        # Q&A format - extract first paragraph as summary
        paragraphs = content.split("\n\n")
        summary = paragraphs[0] if paragraphs else content[:500]
        return (
            f"<|im_start|>user\n"
            f"What is the latest news about {title}?<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{summary}<|im_end|>"
        )

    return content
