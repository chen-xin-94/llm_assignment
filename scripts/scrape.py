#!/usr/bin/env python
"""BMW Press Release Data Pipeline - Scraper.

Handles data collection:
1. Collecting article URLs
2. Scraping article content
3. Downloading PDF attachments

Preprocessing is now handled separately by `scripts/preprocess.py`.

Usage:
    python scripts/scrape.py --target 100 --all
    python scripts/scrape.py --target 100 --scrape
"""

import argparse
import asyncio
from pathlib import Path
import sys

# Add project root to path if needed
sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_assignment.data_engine.pdf_downloader import download_pdfs_async
from llm_assignment.data_engine.scraper import collect_urls_async
from llm_assignment.data_engine.scraper import scrape_articles_async


async def run_pipeline(
    target_count: int,
    data_dir: str,
    do_scrape: bool,
    do_download_pdfs: bool,
):
    """Run the data collection pipeline phases."""
    data_path = Path(data_dir)

    print("=" * 40)
    print("BMW Press Release Scraper")
    print(f"Target: {target_count} articles")
    print(f"Data directory: {data_path}")
    print("=" * 40)
    print()

    # Phase 0: Collect URLs (always runs)
    print("Phase 0: Collecting article URLs...")
    await collect_urls_async(data_path, target_count=target_count)

    # Phase 1: Scrape content
    if do_scrape:
        print("\nPhase 1: Scraping article content...")
        await scrape_articles_async(data_path, max_articles=target_count)

    # Phase 2: Download PDFs
    if do_download_pdfs:
        print("\nPhase 2: Downloading PDF attachments...")
        await download_pdfs_async(data_path, max_downloads=None)

    print("\n" + "=" * 40)
    print("Collection Complete!")
    if not do_scrape:
        print("Tip: Add --scrape to also fetch article content")
    if not do_download_pdfs:
        print("Tip: Add --download-pdfs to download PDF attachments")
    print("\nNext step: Run 'python scripts/preprocess.py' to prepare data for training.")


def main():
    parser = argparse.ArgumentParser(description="BMW Press Release Scraper")
    parser.add_argument("--target", type=int, default=100, help="Target number of articles")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--scrape", action="store_true", help="Also scrape article content")
    parser.add_argument("--download-pdfs", action="store_true", help="Download PDF attachments")
    parser.add_argument("--all", action="store_true", help="Run full collection (scrape + download)")

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        args.scrape = True
        args.download_pdfs = True

    asyncio.run(
        run_pipeline(
            target_count=args.target,
            data_dir=args.data_dir,
            do_scrape=args.scrape,
            do_download_pdfs=args.download_pdfs,
        )
    )


if __name__ == "__main__":
    main()
