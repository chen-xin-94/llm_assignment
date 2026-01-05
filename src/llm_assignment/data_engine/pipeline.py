"""Pipeline orchestration for PDF processing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import random
import re
from typing import TYPE_CHECKING

from datasets import Dataset
from datasets import DatasetDict
from tqdm import tqdm

from llm_assignment.data_engine.extraction import clean_pdf_text
from llm_assignment.data_engine.extraction import extract_text_from_pdf
from llm_assignment.data_engine.formatting import format_for_qwen3
from llm_assignment.data_engine.formatting import read_processed_text
from llm_assignment.data_engine.formatting import save_processed_text

if TYPE_CHECKING:
    from llm_assignment.data_engine.llm_filter import LLMFilter

logger = logging.getLogger(__name__)

TEXT_LENGTH_THRESHOLD = 1000000
BATCH_SIZE = 14


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
