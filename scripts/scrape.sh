#!/usr/bin/env bash
# BMW Press Release Data Pipeline
#
# Usage:
#   ./scripts/scrape.sh [OPTIONS]
#
# Options:
#   --target NUM      Target number of articles (default: 100)
#   --scrape          Also scrape article content (not just URLs)
#   --download-pdfs   Download PDF attachments
#   --preprocess      Preprocess PDFs for training
#   --all             Run full pipeline (scrape + download + preprocess)
#   --data-dir DIR    Data directory (default: data)
#
# Examples:
# Collect URLs only
#   ./scripts/scrape.sh --target 100
# Collect URLs and scrape content
#   ./scripts/scrape.sh --target 100 --scrape
# Full pipeline (scrape + download PDFs + preprocess)
#   ./scripts/scrape.sh --target 100 --all
# Just download and preprocess (if URLs already collected)
#   ./scripts/scrape.sh --download-pdfs --preprocess

set -euo pipefail

# Default values
TARGET=1000
DATA_DIR="data"
DO_SCRAPE=false
DO_DOWNLOAD_PDFS=false
DO_PREPROCESS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --scrape)
            DO_SCRAPE=true
            shift
            ;;
        --download-pdfs)
            DO_DOWNLOAD_PDFS=true
            shift
            ;;
        --preprocess)
            DO_PREPROCESS=true
            shift
            ;;
        --all)
            DO_SCRAPE=true
            DO_DOWNLOAD_PDFS=true
            DO_PREPROCESS=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -n 20 "$0" | tail -n 18
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== BMW Press Release Data Pipeline ==="
echo "Target: $TARGET articles"
echo "Data directory: $DATA_DIR"
echo ""

# Phase 0: Collect article URLs (always runs)
echo "Phase 0: Collecting article URLs..."
uv run python -m llm_assignment.data_engine.scraper --target "$TARGET" --data-dir "$DATA_DIR"

# Phase 1: Scrape article content
if [[ "$DO_SCRAPE" == true ]]; then
    echo ""
    echo "Phase 1: Scraping article content..."
    uv run python -m llm_assignment.data_engine.scraper --target "$TARGET" --data-dir "$DATA_DIR" --scrape
fi

# Phase 2: Download PDFs
if [[ "$DO_DOWNLOAD_PDFS" == true ]]; then
    echo ""
    echo "Phase 2: Downloading PDF attachments..."
    uv run python -m llm_assignment.data_engine.pdf_downloader --data-dir "$DATA_DIR"
fi

# Phase 3: Preprocess PDFs
if [[ "$DO_PREPROCESS" == true ]]; then
    echo ""
    echo "Phase 3: Preprocessing PDFs for training..."
    uv run python -m llm_assignment.data_engine.pdf_preprocessor --data-dir "$DATA_DIR"
fi

echo ""
echo "=== Done! ==="
if [[ "$DO_SCRAPE" == false ]]; then
    echo "Next: Add --scrape to also fetch article content"
fi
if [[ "$DO_DOWNLOAD_PDFS" == false ]]; then
    echo "Next: Add --download-pdfs to download PDF attachments"
fi
if [[ "$DO_PREPROCESS" == false ]]; then
    echo "Next: Add --preprocess to create training dataset"
fi
