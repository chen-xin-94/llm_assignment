"""BMW PDF Downloader.

Downloads PDF attachments from BMW press releases.
Uses all_articles.json metadata to find PDF URLs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re

import aiohttp


async def download_pdf(
    session: aiohttp.ClientSession,
    url: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> bool:
    """Download a single PDF file.

    Args:
        session: aiohttp session for making requests
        url: URL of the PDF to download
        output_path: Path to save the PDF
        overwrite: Whether to overwrite existing files

    Returns:
        True if download succeeded, False otherwise
    """
    if output_path.exists() and not overwrite:
        print(f"  Skipping (exists): {output_path.name}")
        return True

    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"  Failed ({response.status}): {url}")
                return False

            content = await response.read()

            # Verify it's actually a PDF
            if not content.startswith(b"%PDF"):
                print(f"  Not a PDF: {url}")
                return False

            output_path.write_bytes(content)
            print(f"  Downloaded: {output_path.name} ({len(content) / 1024:.1f} KB)")
            return True

    except aiohttp.ClientError as e:
        print(f"  Error downloading {url}: {e}")
        return False


async def download_pdfs_async(
    data_dir: Path | str = "data",
    max_downloads: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download PDF attachments for all articles.

    Args:
        data_dir: Base data directory containing all_articles.json
        max_downloads: Maximum number of PDFs to download (None = all)
        overwrite: Whether to overwrite existing PDFs

    Returns:
        List of paths to downloaded PDFs
    """
    data_dir = Path(data_dir)
    pdfs_dir = data_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Load article metadata
    metadata_path = data_dir / "all_articles.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Run scraper first.")
        return []

    with metadata_path.open(encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Found {len(articles)} articles in metadata")

    # Filter articles with PDF URLs
    articles_with_pdfs = [(i, a) for i, a in enumerate(articles) if a.get("pdf_url")]
    print(f"Articles with PDF URLs: {len(articles_with_pdfs)}")

    if max_downloads:
        articles_with_pdfs = articles_with_pdfs[:max_downloads]

    downloaded = []

    # Use aiohttp for async downloads
    connector = aiohttp.TCPConnector(limit=5)  # Limit concurrent connections
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i, article in articles_with_pdfs:
            pdf_url = article["pdf_url"]
            title = article.get("title", "Untitled")

            # Create filename matching raw JSON naming convention
            filename = re.sub(r"[^\w\-]", "_", title)[:50]
            output_path = pdfs_dir / f"{i:03d}_{filename}.pdf"

            print(f"[{i + 1}/{len(articles_with_pdfs)}] {title[:50]}...")
            success = await download_pdf(session, pdf_url, output_path, overwrite=overwrite)

            if success and output_path.exists():
                downloaded.append(output_path)

            # Rate limiting
            await asyncio.sleep(0.5)

    print(f"\nDownloaded {len(downloaded)} PDFs to {pdfs_dir}")
    return downloaded


def download_pdfs(
    data_dir: str = "data",
    max_downloads: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Synchronous wrapper for PDF downloads.

    Args:
        data_dir: Base data directory containing all_articles.json
        max_downloads: Maximum number of PDFs to download (None = all)
        overwrite: Whether to overwrite existing PDFs

    Returns:
        List of paths to downloaded PDFs
    """
    return asyncio.run(download_pdfs_async(Path(data_dir), max_downloads, overwrite))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BMW PDF attachments")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory containing all_articles.json (default: data)",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Maximum number of PDFs to download (default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDF files",
    )
    args = parser.parse_args()
    download_pdfs(data_dir=args.data_dir, max_downloads=args.max_downloads, overwrite=args.overwrite)
