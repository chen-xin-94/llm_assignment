"""BMW PDF processor.

Extracts text from PDFs and creates training dataset in Qwen3 format.
Creates train/eval splits compatible with HuggingFace datasets.

Supports phase-based processing:
- regex: Extract and clean with regex, save to preprocessed/
- llm: Apply LLM filtering to preprocessed texts
- format: Transform to Qwen3 format, save to processed/
- all: Run all phases (regex -> llm -> format)
- no-llm: Run regex and format phases only (regex + format, default)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import re
from typing import TYPE_CHECKING

from datasets import Dataset
from datasets import DatasetDict
from pypdf import PdfReader
from tqdm import tqdm

if TYPE_CHECKING:
    from llm_assignment.data_engine.llm_filter import LLMFilter

logger = logging.getLogger(__name__)

# Delimiters for preprocessed text files
METADATA_START = "--- METADATA ---"
REASONING_START = "--- REASONING ---"
CONTENT_START = "--- CONTENT ---"
TEXT_LENGTH_THRESHOLD = 1000000
BATCH_SIZE = 14


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
        # We ignore reasoning for now as the return signature doesn't include it
        # and downstream format phase doesn't need it.
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

    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove common BMW press release artifacts
    # These patterns are based on analysis of actual PDF content
    footer_patterns = [
        # Image reference codes (P90XXXXXX format)
        r"\bP90\d{6}\b",
        # Page markers (English and German) - more flexible matching
        r"\nPage\s+\d+",
        r"\bPage\s+\d+\b",
        r"\bSeite\s+\d+\b",
        # Corporate headers - leading \n is optional, handle any whitespace around it
        r"(?:^|\n)\s*CORPORATE\s+COMMUNICATIONS\s*(?:\n|\s)",
        r"(?:^|\n)\s*Corporate\s+Communications\s*(?:\n|$)",
        r"Corporate\s+Communication\s+Media\s+Information\s+.*?\d{4}\s*",
        r"BMW\s+Corporate\s+Communications\s*\n?",
        r"PRESSE-?\s*UND\s*ÖFFENTLICHKEITSARBEIT\s*\n?",
        # Press/Media information headers - leading \n is optional
        # Handles: department names, various date formats, multi-line formats
        r"(?:^|\n)\s*Media\s+[Ii]nformation\s+(?:BMW\s+)?.*?\d{4}\s*(?:Subject\s*)?",
        r"(?:^|\n)\s*Media\s+[Ii]nformation\s*\n\s*\d+\s+\w+\s+\d{4}\s*\n?",
        r"(?:^|\n)\s*MEDIA\s+INFORMATION\s+.*?\d{4}\s*",
        r"Media\s+Information\s+Date\s+.*?(?=\n\n|\n[A-Z])",
        r"Presse\s+Information\s*\n",
        r"Datum\s+\d+.*?\d{4}\s*\n",
        r"Subject\s+.*?(?=\n\n)",
        r"Thema\s+.*?(?=\n\n)",
        # Contact blocks and spokesperson info
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
        # The BMW Group boilerplate - multiple patterns for different formats
        r"The\s+BMW\s+Group\s+With\s+its\s+four\s+brands.*?useful\s+life\.?",
        r"Die\s+BMW\s+Group\s+ist\s+mit\s+ihren.*?Nutzungsphase.*?Produkte\.?",
        r"The\s+BMW\s+Group\s*\n\s*With\s+its\s+four\s+brands.*?(?=\n\n|\Z)",
        # CONFIDENTIAL markers
        r"CONFIDENTIAL\s*",
        # Regulatory/legal disclaimers about fuel consumption
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

    # Note: Smart content trimming (header removal, title finding) is now handled
    # by the LLM filter when using phase='all' or 'llm'. This function only
    # handles the final formatting step.

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


def preprocess_pdfs(
    data_dir: str | Path = "data",
    train_ratio: float = 0.9,
    style: str = "instruct",
    seed: int = 42,
    phase: str = "no-llm",
    llm_model: str = "openai/gpt-oss-120b",
) -> DatasetDict:
    """Extract text from PDFs and create HuggingFace dataset.

    Supports phase-based processing:
    - regex: Extract and clean with regex, save to preprocessed_regex/
    - llm: Apply LLM filtering to preprocessed texts
    - format: Transform to Qwen3 format, save to processed/
    - all: Run all phases (regex -> llm -> format)
    - no-llm: Run regex and format only (skips LLM)

    Args:
        data_dir: Base data directory containing pdfs/ subdirectory
        train_ratio: Proportion for training set
        style: Format style for Qwen3
        seed: Random seed for reproducibility
        phase: Processing phase to run
        llm_model: HuggingFace model name for LLM filtering

    Returns:
        DatasetDict with train and eval splits
    """
    data_dir = Path(data_dir)
    pdfs_dir = data_dir / "pdfs"
    preprocessed_dir = data_dir / "preprocessed_regex"
    llm_filtered_dir = data_dir / "preprocessed_llm"
    output_dir = data_dir / "processed"

    # Determine if LLM filter is needed based on phase
    use_llm_filter = phase in ("llm", "all")

    # Create directories
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if use_llm_filter:
        llm_filtered_dir.mkdir(parents=True, exist_ok=True)

    # Load article metadata for title/date info
    metadata_path = data_dir / "all_articles.json"
    metadata: dict[int, dict] = {}
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            articles = json.load(f)
            # Index by position (matches PDF naming)
            metadata = dict(enumerate(articles))

    # Initialize LLM filter if needed
    llm_filter: LLMFilter | None = None
    if use_llm_filter:
        from llm_assignment.data_engine.llm_filter import LLMFilter

        print(f"Loading LLM filter: {llm_model}")
        llm_filter = LLMFilter(model_name=llm_model)

    rejected_articles: list[dict] = []
    processed_articles: list[dict] = []

    # ==================== PHASE: REGEX ====================
    if phase in ("regex", "all", "no-llm"):
        if not pdfs_dir.exists():
            print(f"Error: {pdfs_dir} not found. Run pdf_downloader first.")
            return DatasetDict()

        pdf_files = sorted(pdfs_dir.glob("*.pdf"))
        print(f"\\n{'=' * 50}")
        print(f"REGEX PHASE: Processing {len(pdf_files)} PDF files")
        print(f"{'=' * 50}")

        skipped_count = 0
        for pdf_path in pdf_files:
            # Check if preprocessed file already exists - skip if so
            preprocessed_path = preprocessed_dir / f"{pdf_path.stem}.txt"
            if preprocessed_path.exists():
                skipped_count += 1
                continue

            print(f"Processing: {pdf_path.name}")

            # Extract index from filename (e.g., "000_title.pdf" -> 0)
            idx_match = re.match(r"(\d+)_", pdf_path.name)
            idx = int(idx_match.group(1)) if idx_match else None

            # Extract text
            raw_text = extract_text_from_pdf(pdf_path)
            if not raw_text:
                print("  No text extracted")
                continue

            # Clean text with regex
            cleaned_text = clean_pdf_text(raw_text)

            # get rid of extremely long texts
            if len(cleaned_text) > TEXT_LENGTH_THRESHOLD:
                print(f"Text too long, skipping: {pdf_path.name}")
                continue

            # Get metadata if available
            article_meta = metadata.get(idx, {}) if idx is not None else {}
            title = article_meta.get("title", pdf_path.stem)

            # Save preprocessed text
            preprocessed_path = preprocessed_dir / f"{pdf_path.stem}.txt"
            save_processed_text(
                preprocessed_path,
                title=title,
                date=article_meta.get("date", ""),
                source=pdf_path.name,
                content=cleaned_text,
            )

            print(f"  Saved preprocessed: {preprocessed_path.name} ({len(cleaned_text.split())} words)")

        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} already-preprocessed files")
        print(f"Regex phase complete. Output: {preprocessed_dir}")

    # ==================== PHASE: LLM ====================
    if phase in ("llm", "all"):
        txt_files = sorted(preprocessed_dir.glob("*.txt"))
        print(f"\n{'=' * 50}")
        print(f"LLM PHASE: Filtering {len(txt_files)} preprocessed files")
        print(f"{'=' * 50}")

        # Explicitly load model first
        if llm_filter:
            print(f"Loading model: {llm_model} ...")
            llm_filter.load_model()
            print("Model loaded.")

        # Prepare batch processing
        batch_size = BATCH_SIZE  # Adjust based on VRAM (120B model is large, keep small)

        # Gather all inputs first, skipping already-processed files
        inputs = []
        skipped_count = 0
        for txt_path in txt_files:
            # Check if LLM-filtered file already exists - skip if so
            filtered_path = llm_filtered_dir / txt_path.name
            if filtered_path.exists():
                skipped_count += 1
                continue

            title, date, source, text_content = read_processed_text(txt_path)
            inputs.append({"text": text_content, "title": title, "path": txt_path, "date": date, "source": source})

        if skipped_count > 0:
            print(f"Skipped {skipped_count} already-processed files")

        # Process in batches
        print(f"Processing in batches of {batch_size}...")
        for i in tqdm(range(0, len(inputs), batch_size)):
            batch = inputs[i : i + batch_size]

            # Run batch inference
            results = llm_filter.filter_batch(batch, batch_size=len(batch))

            # Save results
            for j, result in enumerate(results):
                input_item = batch[j]
                txt_path = input_item["path"]

                if result["rejected"]:
                    print(f"\nREJECTED: {txt_path.name} ({result['reason']})")
                    rejected_articles.append(
                        {
                            "source": input_item["source"],
                            "title": input_item["title"],
                            "reason": result["reason"],
                            "reasoning": result.get("reasoning"),
                        }
                    )
                else:
                    # Save LLM-filtered text with reasoning
                    filtered_path = llm_filtered_dir / txt_path.name
                    save_processed_text(
                        filtered_path,
                        title=input_item["title"],
                        date=input_item["date"],
                        source=input_item["source"],
                        content=result["content"],
                        reasoning=result.get("reasoning"),
                    )

        # Save rejection log
        if rejected_articles:
            rejected_path = llm_filtered_dir / "rejected.json"
            with rejected_path.open("w", encoding="utf-8") as f:
                json.dump(rejected_articles, f, indent=2, ensure_ascii=False)
            print(f"\nRejected {len(rejected_articles)} articles. See: {rejected_path}")

        print(f"\nLLM phase complete. Output: {llm_filtered_dir}")

    # ==================== PHASE: FORMAT ====================
    if phase in ("format", "all", "no-llm"):
        # Determine source directory for formatting
        # Prefer LLM filtered if available, UNLESS we are in 'no-llm' phase
        if phase != "no-llm" and llm_filtered_dir.exists() and any(llm_filtered_dir.glob("*.txt")):
            source_dir = llm_filtered_dir
            print(f"Using LLM-filtered content from {llm_filtered_dir}")
        else:
            source_dir = preprocessed_dir
            print(f"Using regex-cleaned content from {preprocessed_dir}")

        txt_files = sorted(source_dir.glob("*.txt"))
        print(f"\n{'=' * 50}")
        print(f"FORMAT PHASE: Formatting {len(txt_files)} files from {source_dir.name}")
        print(f"{'=' * 50}")

        for txt_path in txt_files:
            if txt_path.name == "rejected.json":
                continue

            # Read file
            title, date, source, text_content = read_processed_text(txt_path)

            # Format for Qwen3
            formatted_text = format_for_qwen3(title, text_content, style)

            processed_articles.append(
                {
                    "text": formatted_text,
                    "title": title,
                    "source_pdf": source,
                    "date": date if date else None,
                    "word_count": len(text_content.split()),
                }
            )

            print(f"  Formatted: {txt_path.name}")

    # Create train/eval splits and save dataset
    if not processed_articles:
        if phase in ("regex", "llm"):
            print("\nPhase complete. No formatting performed.")
            return DatasetDict()
        print("\nNo articles were processed for formatting.")
        return DatasetDict()

    random.seed(seed)
    shuffled = processed_articles.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]

    # Create HuggingFace dataset
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_data),
            "eval": Dataset.from_list(eval_data),
        }
    )

    # Save dataset
    dataset.save_to_disk(str(output_dir))

    # Also save as JSONL for inspection
    for split_name, split_data in dataset.items():
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(dict(item), ensure_ascii=False) + "\\n")

    # Save summary (only if in format phase or all/no-llm)
    if phase in ("format", "all", "no-llm") and processed_articles:
        summary = {
            "total_processed": len(processed_articles),
            "train_samples": len(train_data),
            "eval_samples": len(eval_data),
            "total_words": sum(a["word_count"] for a in processed_articles),
            "format_style": style,
            "llm_filtered": use_llm_filter,
            "rejected_count": len(rejected_articles) if use_llm_filter else 0,
        }
        with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\\n{'=' * 50}")
        print("Preprocessing Complete")
        print(f"{'=' * 50}")
        print(f"Successfully processed: {summary['total_processed']}")
        print(f"Train samples: {summary['train_samples']}")
        print(f"Eval samples: {summary['eval_samples']}")
        print(f"Total words: {summary['total_words']}")
        if use_llm_filter:
            print(f"LLM rejected: {summary['rejected_count']}")
        print(f"Output: {output_dir}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BMW PDFs for Qwen3 training")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory containing pdfs/ subdirectory (default: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Proportion for training set (default: 0.9)",
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["instruct", "article", "qa"],
        default="instruct",
        help="Format style for training (default: instruct)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["regex", "llm", "format", "all", "no-llm"],
        default="no-llm",
        help="Processing phase to run (default: no-llm)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai/gpt-oss-120b",
        help="LLM model for filtering (default: openai/gpt-oss-120b)",
    )
    args = parser.parse_args()
    preprocess_pdfs(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        style=args.style,
        seed=args.seed,
        phase=args.phase,
        llm_model=args.llm_model,
    )
