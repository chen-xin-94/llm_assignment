"""PDF extraction and cleaning logic."""

from __future__ import annotations

from pathlib import Path
import re

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        reader = PdfReader(pdf_path)
        text_parts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts)

    except Exception as e:
        print(f"  Error reading {pdf_path.name}: {e}")
        return ""


def clean_pdf_text(text: str) -> str:
    """Clean extracted PDF text.

    Removes common artifacts from PDF extraction like:
    - Page headers/footers
    - Excessive whitespace
    - Hyphenated line breaks

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Fix hyphenated line breaks (word-\nbreak -> wordbreak)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize horizontal whitespace (spaces/tabs -> single space)
    text = re.sub(r"[ \t]+", " ", text)

    # Strip whitespace from the beginning and end of each line
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)

    # Replace multiple newlines (3 or more) with double newline (paragraph break)
    # This handles cases where we had \n\n\n or \n \n \n (after stripping)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove common BMW press release artifacts
    footer_patterns = [
        # Image reference codes (P90XXXXXX format)
        r"\bP90\d{6}\b",
        # Page markers
        r"\nPage\s+\d+",
        r"\bPage\s+\d+\b",
        r"\bSeite\s+\d+\b",
        # Corporate headers
        r"(?:^|\n)\s*CORPORATE\s+COMMUNICATIONS\s*(?:\n|\s)",
        r"(?:^|\n)\s*Corporate\s+Communications\s*(?:\n|$)",
        r"Corporate\s+Communication\s+Media\s+Information\s+.*?\d{4}\s*",
        r"BMW\s+Corporate\s+Communications\s*\n?",
        r"PRESSE-?\s*UND\s*ÖFFENTLICHKEITSARBEIT\s*\n?",
        # Press/Media information headers
        r"(?:^|\n)\s*Media\s+[Ii]nformation\s+(?:BMW\s+)?.*?\d{4}\s*(?:Subject\s*)?",
        r"(?:^|\n)\s*Media\s+[Ii]nformation\s*\n\s*\d+\s+\w+\s+\d{4}\s*\n?",
        r"(?:^|\n)\s*MEDIA\s+INFORMATION\s+.*?\d{4}\s*",
        r"Media\s+Information\s+Date\s+.*?(?=\n\n|\n[A-Z])",
        r"Presse\s+Information\s*\n",
        r"Datum\s+\d+.*?\d{4}\s*\n",
        r"Subject\s+.*?(?=\n\n)",
        r"Thema\s+.*?(?=\n\n)",
        # Contact blocks
        r"Media\s+Contact\..*?(?=\n\n|\Z)",
        r"Bitte\s+wenden\s+Sie\s+sich\s+bei\s+Rückfragen.*?(?=\n\n|\Z)",
        r"(?:Telefon|Phone|T\s*elephone):\s*\+?\d[\d\s\-()]+\s*\n?",
        r"E-?[Mm]ail:\s*[\w\.\-]+@[\w\.\-]+\s*\n?",
        r"Fax\s+\+?\d[\d\s\-]+\s*\n?",
        r"Press\s+Officer.*?(?=\n\n|\Z)",
        r"Spokesperson.*?(?=\n\n|\Z)",
        r"Head\s+of\s+.*?Communications.*?(?=\n\n|\Z)",
        # Company address blocks
        r"Firma\s+Bayerische.*?(?=\n\n)",
        r"Postanschrift\s+BMW\s+AG.*?(?=\n\n)",
        r"Company\s+Bayerische\s+Motoren.*?(?=\n\n)",
        r"Bayerische\s+Motoren\s*\nAktiengesellschaft.*?(?=\n\n|\Z)",
        # Social media and website blocks
        r"(?:LinkedIn|YouTube|Instagram|Facebook|X|Twitter):\s*https?://[^\s]+\s*\n?",
        r"www\.(?:facebook|instagram|twitter|x)\.com/[^\s]+\s*\n?",
        r"↗\s*www\.[^\s]+\s*\n?",
        r"Internet:\s*www\.[\w\.]+\s*\n?",
        r"(?:Media\s+)?Website\.?\s*\n?.*?www\.[\w\.\-/]+\s*\n?",
        r"BMW\s+M\s+Motorsport\s+on\s+the\s+Web\..*?(?=\n\n|\Z)",
        # The BMW Group boilerplate
        r"The\s+BMW\s+Group\s+With\s+its\s+four\s+brands.*?useful\s+life\.?",
        r"Die\s+BMW\s+Group\s+ist\s+mit\s+ihren.*?Nutzungsphase.*?Produkte\.?",
        r"The\s+BMW\s+Group\s*\n\s*With\s+its\s+four\s+brands.*?(?=\n\n|\Z)",
        # CONFIDENTIAL markers
        r"CONFIDENTIAL\s*",
        # Regulatory/legal disclaimers
        r"(?:Die|The)\s+(?:Angaben|information).*?(?:WLTP|NEFZ).*?(?=\n\n|\Z)",
        r"www\.bmw\.de/wltp.*?(?=\n)",
        r"(?:Weitere\s+)?(?:Informationen|Information).*?(?:Kraftstoffverbrauch|fuel\s+consumption).*?unentgeltlich\s+erhältlich.*?(?=\n\n|\Z)",
        # Published by / contacts sections
        r"PUBLISHED\s+BY\s*\n.*?(?=\n\n|\Z)",
        r"CONTACTS?\s*\n.*?(?=\n\n|\Z)",
        r"BUSINESS\s+AND\s+FINANCE\s+PRESS.*?(?=\n\n|\Z)",
        r"INVESTOR\s+RELATIONS.*?(?=\n\n|\Z)",
        r"THE\s+BMW\s+GROUP\s+ON\s+THE\s+INTERNET.*?(?=\n\n|\Z)",
    ]
    for pattern in footer_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)

    # Clean up whitespace
    text = text.strip()
    return re.sub(r"\n{3,}", "\n\n", text)
