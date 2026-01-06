# Data Collection Pipeline

The data collection engine is responsible for gathering press releases from the BMW Group PressClub. It is built using `Crawl4AI` for robust web scraping and `aiohttp` for efficient PDF downloading.

## Workflow Overview

The pipeline operates in three distinct phases:

1. **URL Collection (Phase 0)**: Discovers article URLs using dynamic scrolling and JavaScript execution.
2. **Scraping (Phase 1)**: Visits each discovered URL to extract content and metadata.
3. **PDF Downloading (Phase 2)**: Downloads the associated PDF attachments for each article.

## Usage

The main entry point is `scripts/scrape.py`.

```bash
# Full pipeline (Collect + Scrape + Download)
python scripts/scrape.py --target 1000 --all

# Collect URLs only
python scripts/scrape.py --target 1000

# Scrape content for collected URLs
python scripts/scrape.py --scrape

# Download PDFs for collected URLs
python scripts/scrape.py --download-pdfs
```

## Technical Implementation

### 1. URL Collection (`scraper.py`)

The BMW PressClub website uses infinite scrolling and lazy loading. The `collect_urls_async` function handles this by:

* **Dynamic Scrolling**: Automatically scrolls to the bottom of the page and clicks the "Load More" (`#lazy-load-button`) button.
* **Duplicate Prevention**: Tracks unique article IDs (e.g., `T044108EN`) to avoid duplicates from "Trending" or "Related" sections.
* **JavaScript Execution**: Injects custom JavaScript to robustly count and extract article URLs directly from the DOM, ensuring high recall.

The discovered URLs are saved to `data/all_articles.json`.

### 2. Content Scraping (`scraper.py`)

Once URLs are collected, `scrape_articles_async` processes each article:

* **Crawl4AI**: Uses `AsyncWebCrawler` to render the page and extract markdown.
* **Boilerplate Removal**: Regex patterns remove standard footer text (e.g., "Press Contact", "Downloads") to ensure clean training data.
* **PDF Detection**: identifying the PDF attachment URL associated with the article.
* **Metadata**: Extracts title and publication date using multiple regex date patterns (e.g., "DD Month YYYY", "DD.MM.YYYY").

Scraped articles are saved as individual JSON files in `data/scraped/` (e.g., `001_Title_of_Article.json`).

### 3. PDF Downloading (`pdf_downloader.py`)

The `download_pdfs_async` function uses `aiohttp` for concurrent downloads:

* **Concurrency**: Limits concurrent connections (default: 5) to respect server limits.
* **Validation**: Verifies that downloaded files start with the `%PDF` header.
* **Naming**: saves files to `data/pdfs/` with filenames matching their corresponding JSON metadata.

## Output Directory Structure

```text
data/
├── all_articles.json       # Master list of all discovered URLs and metadata
├── scraped/                # Individual JSON files containing article text
│   ├── 000_Article_Title.json
│   └── ...
└── pdfs/                   # Downloaded PDF attachments
    ├── 000_Article_Title.pdf
    └── ...
```
