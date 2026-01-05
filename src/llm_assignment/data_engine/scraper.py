"""BMW Press Release Scraper using Crawl4AI.

Scrapes press releases from BMW Group PressClub and saves as clean Markdown.
Articles are ordered by date (latest first) and PDF attachment URLs are extracted.

Two-phase workflow:
1. collect: Discover article URLs and save to all_articles.json (Default)
2. scrape: Scrape content from URLs in all_articles.json
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import re

from crawl4ai import AsyncWebCrawler
from crawl4ai import BrowserConfig
from crawl4ai import CrawlerRunConfig


async def get_article_urls(
    crawler: AsyncWebCrawler, target_count: int = 100, existing_ids: set[str] | None = None
) -> list[dict]:
    """Extract article URLs from BMW PressClub using dynamic discovery.

    The BMW site loads ~30 articles initially. More are loaded by:
    1. Clicking #lazy-load-button
    2. Scrolling to bottom for infinite scroll

    URLs are extracted via JavaScript from DOM elements to avoid counting
    duplicates from sidebars/footers/trending sections.
    """
    base_url = "https://www.press.bmwgroup.com/global/article"
    articles = []
    seen_ids = existing_ids.copy() if existing_ids else set()

    def normalize_url(article_id: str, slug: str) -> str:
        clean_slug = slug.split("?")[0].split("#")[0]
        return f"https://www.press.bmwgroup.com/global/article/detail/{article_id}/{clean_slug}"

    # JS to click button, scroll, and collect article URLs from DOM elements only
    # We use DOM element count for scrolling decisions (faster), but extract unique
    # article IDs from .article-list-item-wrapper at the end (to avoid duplicates)
    js_discover = f"""
    const sleep = ms => new Promise(r => setTimeout(r, ms));

    // Count DOM elements for scroll decisions (broader selector to ensure we load enough)
    const getDomElementCount = () => document.querySelectorAll('.article-list-item-wrapper, h3 > a[href*="/detail/"]').length;

    // Extract unique article URLs from main article list (broader selector)
    const getArticleData = () => {{
        const seen = new Set();
        const articles = [];
        // Use the same broad selector as counting, but extract unique IDs
        document.querySelectorAll('.article-list-item-wrapper a[href*="/detail/"], h3 > a[href*="/detail/"]').forEach(a => {{
            const href = a.getAttribute('href');
            const match = href && href.match(/\\/detail\\/(T\\d+EN)\\/([^\\/\\?#]+)/);
            if (match && !seen.has(match[1])) {{
                seen.add(match[1]);
                articles.push({{id: match[1], slug: match[2]}});
            }}
        }});
        return articles;
    }};

    // Count unique articles for scroll loop condition
    const getUniqueArticleCount = () => getArticleData().length;

    window.discoveryFinished = false;
    window.collectedArticles = [];

    // Accept Cookie Banner
    try {{
        const acceptBtn = Array.from(document.querySelectorAll('button')).find(b =>
            b.textContent.includes('ACCEPT ALL') || b.textContent.includes('Agree')
        );
        if (acceptBtn) {{
            acceptBtn.click();
            await sleep(1000);
        }}
    }} catch (e) {{}}

    let lastUniqueCount = getUniqueArticleCount();
    let targetCount = {target_count};
    let stallCount = 0;

    // Initial click
    const btn = document.querySelector('#lazy-load-button');
    if (btn && btn.offsetParent !== null) {{
        btn.click();
        await sleep(3000);
    }}

    // Scroll loop - continue until we have target unique articles OR stall out
    while (getUniqueArticleCount() < targetCount && stallCount < 10) {{
        window.scrollTo(0, document.body.scrollHeight);
        await sleep(3000);

        const currentUniqueCount = getUniqueArticleCount();
        if (currentUniqueCount > lastUniqueCount) {{
            lastUniqueCount = currentUniqueCount;
            stallCount = 0;
        }} else {{
            stallCount++;
            window.scrollBy(0, -500);
            await sleep(500);
            window.scrollTo(0, document.body.scrollHeight);
            await sleep(1500);
        }}

        const retryBtn = document.querySelector('#lazy-load-button');
        if (retryBtn && retryBtn.offsetParent !== null) {{
           retryBtn.click();
           await sleep(2000);
        }}
    }}

    // Collect final article data (unique IDs only) and store in global variable
    window.collectedArticles = getArticleData();
    window.discoveryFinished = true;
    """

    print(f"Discovering up to {target_count} articles...")

    config = CrawlerRunConfig(
        js_code=js_discover,
        wait_for="js:() => window.discoveryFinished === true",
        page_timeout=300000,
    )

    result = await crawler.arun(url=base_url, config=config)

    if result.success:
        # Extract article data from the page using js_result or by re-running extraction
        # The collected articles are stored in window.collectedArticles
        # We need to extract them - Crawl4AI stores JS results in result.js_result or we parse from HTML

        # Try to get the data from a script injection that outputs the result
        # Since Crawl4AI may not directly return JS variable, we'll parse a marker from HTML
        # Alternative: use the executed_js_result if available

        collected_data = []

        # Check if Crawl4AI provides the JS execution result
        if hasattr(result, "js_result") and result.js_result:
            with suppress(json.JSONDecodeError, TypeError):
                collected_data = json.loads(result.js_result)

        # Fallback: Look for the data embedded in HTML via a data attribute or script
        if not collected_data and result.html:
            # Parse from window.collectedArticles if it's in the HTML
            import re as regex_module

            pattern = regex_module.compile(r"window\.collectedArticles\s*=\s*(\[.*?\]);", regex_module.DOTALL)
            match = pattern.search(result.html)
            if match:
                with suppress(json.JSONDecodeError):
                    collected_data = json.loads(match.group(1))

        # If we still don't have data, fall back to regex extraction from article-list wrappers
        if not collected_data and result.html:
            # More targeted regex: only match URLs that appear within article-list-item-wrapper context
            # This is a fallback - extract all and deduplicate
            article_pattern = re.compile(r"/global/article/detail/(T\d+EN)/([^\"'<>\s?]+)")
            matches = article_pattern.findall(result.html)
            seen_in_page = set()
            for article_id, slug in matches:
                if article_id not in seen_in_page:
                    seen_in_page.add(article_id)
                    collected_data.append({"id": article_id, "slug": slug})

        new_count = 0
        for item in collected_data:
            article_id = item.get("id") or item.get("article_id")
            slug = item.get("slug", "")
            if article_id and article_id not in seen_ids:
                seen_ids.add(article_id)
                url = normalize_url(article_id, slug)
                articles.append(
                    {
                        "article_id": article_id,
                        "url": url,
                        "title": None,
                        "date": None,
                        "pdf_url": None,
                        "scraped": False,
                    }
                )
                new_count += 1
                if len(articles) + len(existing_ids or set()) >= target_count:
                    break

        print(f"Found {new_count} new articles (Total collected: {len(seen_ids)})")
    else:
        print(f"Discovery failed: {result.error_message}")

    return articles


async def collect_urls_async(
    data_dir: Path,
    target_count: int = 100,
) -> list[dict]:
    """Collect article URLs until we reach target_count."""
    metadata_path = data_dir.absolute() / "all_articles.json"
    print(f"Checking for existing articles at: {metadata_path}")

    # Load existing articles
    existing_articles = []
    existing_ids = set()
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as f:
            existing_articles = json.load(f)
            existing_ids = {a.get("article_id") or a.get("url", "").split("/")[-2] for a in existing_articles}
        print(f"Loaded {len(existing_articles)} existing articles")
    else:
        print("No existing articles found.")

    if len(existing_articles) >= target_count:
        print(f"Already have {len(existing_articles)} articles (target: {target_count})")
        return existing_articles

    # Collect more URLs
    browser_config = BrowserConfig(headless=True, verbose=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        new_articles = await get_article_urls(crawler, target_count=target_count, existing_ids=existing_ids)

    # Merge and save
    all_articles = existing_articles + new_articles

    # Save updated metadata
    data_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    print(f"Total entries in {metadata_path.name}: {len(all_articles)}")
    return all_articles


def parse_date_from_content(content: str) -> datetime | None:
    """Extract publication date from article content."""
    # Pattern: DD Month YYYY
    pattern1 = re.compile(
        r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        re.IGNORECASE,
    )
    match = pattern1.search(content)
    if match:
        day, month, year = match.groups()
        try:
            dt = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
            return dt.replace(tzinfo=UTC)
        except ValueError:
            pass

    # Pattern: DD.MM.YYYY
    pattern2 = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
    match = pattern2.search(content)
    if match:
        day, month, year = match.groups()
        try:
            dt = datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
            return dt.replace(tzinfo=UTC)
        except ValueError:
            pass

    # Pattern: Month DD, YYYY
    pattern3 = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
        re.IGNORECASE,
    )
    match = pattern3.search(content)
    if match:
        month, day, year = match.groups()
        try:
            dt = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
            return dt.replace(tzinfo=UTC)
        except ValueError:
            pass

    return None


def extract_pdf_url(content: str, html: str | None = None) -> str | None:
    """Extract PDF attachment URL."""
    pdf_pattern = re.compile(r"https://www\.press\.bmwgroup\.com/global/article/attachment/T\d+EN/\d+")
    match = pdf_pattern.search(content)
    if match:
        return match.group(0)
    if html:
        match = pdf_pattern.search(html)
        if match:
            return match.group(0)
    return None


async def scrape_single_article(crawler: AsyncWebCrawler, url: str) -> dict | None:
    """Scrape a single article."""
    config = CrawlerRunConfig(
        word_count_threshold=50,
        remove_overlay_elements=True,
        excluded_tags=["nav", "footer", "header", "aside"],
    )

    result = await crawler.arun(url=url, config=config)

    if not result.success or not result.markdown:
        return None

    raw_content = result.markdown
    pdf_url = extract_pdf_url(raw_content, result.html)

    # Clean boilerplate
    content = raw_content
    boilerplate_patterns = [
        r"#### Press Contact\.\n.*?(?=\n####|\n\n[A-Z]|\Z)",
        r"#### Author\.\n.*?(?=\n####|\n\n[A-Z]|\Z)",
        r"#### Downloads\.\n.*?(?=\n####|\n\n[A-Z]|\Z)",
        r"#### This article in other PressClubs\n.*?(?=\n####|\n\n[A-Z]|\Z)",
        r"#### Article Offline Attachments\.\n.*?(?=\n####|\n\n[A-Z]|\Z)",
        r"#### Article Media Material\.\n.*?(?=\n####|\Z)",
    ]
    for pattern in boilerplate_patterns:
        content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else "Untitled"

    content = re.sub(r"\n{3,}", "\n\n", content)
    content = content.strip()

    pub_date = parse_date_from_content(raw_content)

    return {
        "url": url,
        "title": title,
        "date": pub_date.isoformat() if pub_date else None,
        "pdf_url": pdf_url,
        "content": content,
        "scraped_at": datetime.now(tz=UTC).isoformat(),
    }


def _get_article_index(all_articles: list[dict], article_id: str) -> int:
    """Get the index of an article in the list based on its article_id."""
    for idx, article in enumerate(all_articles):
        if article.get("article_id") == article_id:
            return idx
    return -1


def _generate_scraped_filename(index: int, title: str | None, article_id: str) -> str:
    """Generate a filename with numeric prefix like 000_title.json."""
    prefix = f"{index:03d}_"
    if title:
        safe_title = re.sub(r"[^\w\-]", "_", title)[:50]
        return f"{prefix}{article_id}_{safe_title}.json"
    return f"{prefix}{article_id}.json"


async def scrape_articles_async(
    data_dir: Path,
    max_articles: int | None = None,
) -> list[dict]:
    """Scrape content from collected URLs."""
    metadata_path = data_dir / "all_articles.json"
    scraped_dir = data_dir / "scraped"
    scraped_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found. Run discovery first.")
        return []

    with metadata_path.open(encoding="utf-8") as f:
        all_articles = json.load(f)

    # Filter articles: skip if scraped=true OR if file already exists
    to_scrape = []
    for article in all_articles:
        article_id = article.get("article_id") or article.get("url", "").split("/")[-2]

        # Check 1: Skip if already marked as scraped in all_articles.json
        if article.get("scraped", False):
            continue

        # Check 2: Skip if file already exists (check for any file with this article_id)
        existing_files = list(scraped_dir.glob(f"*{article_id}*.json"))
        if existing_files:
            print(f"Skipping {article_id}: file already exists at {existing_files[0].name}")
            continue

        to_scrape.append(article)

    if max_articles:
        to_scrape = to_scrape[:max_articles]

    print(f"Found {len(to_scrape)} articles to scrape")

    if not to_scrape:
        print("All articles already scraped!")
        return []

    scraped_data = []
    browser_config = BrowserConfig(headless=True, verbose=False)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, article_meta in enumerate(to_scrape):
            url = article_meta["url"]
            article_id = article_meta.get("article_id") or url.split("/")[-2]

            # Double-check file doesn't exist before scraping
            existing_files = list(scraped_dir.glob(f"*{article_id}*.json"))
            if existing_files:
                print(f"Skipping {article_id}: file already exists")
                continue

            print(f"Scraping article {i + 1}/{len(to_scrape)}: {url[:70]}...")

            article = await scrape_single_article(crawler, url)
            if article:
                scraped_data.append(article)
                article_meta["title"] = article["title"]
                article_meta["date"] = article["date"]
                article_meta["pdf_url"] = article["pdf_url"]
                article_meta["scraped"] = True
                article_meta["scraped_at"] = article["scraped_at"]

            await asyncio.sleep(1.5)

    # Save updated all_articles.json
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    # Save individual JSON files with numeric prefix based on index in all_articles
    for article in scraped_data:
        article_id = article["url"].split("/")[-2]
        # Get the index of this article in all_articles for consistent numbering
        idx = _get_article_index(all_articles, article_id)
        if idx == -1:
            idx = 0  # Fallback

        filename = _generate_scraped_filename(idx, article["title"], article_id)
        filepath = scraped_dir / filename
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(article, f, indent=2, ensure_ascii=False)

    print(f"\nScraped {len(scraped_data)} articles")
    return scraped_data


async def scrape_bmw_articles(max_articles: int = 100, output_dir: Path | None = None) -> list[dict]:
    """Compatibility function."""
    data_dir = output_dir or Path("data")
    await collect_urls_async(data_dir, target_count=max_articles)
    return await scrape_articles_async(data_dir, max_articles=max_articles)


def run_scraper(max_articles: int = 100, output_dir: str = "data") -> list[dict]:
    """Synchronous wrapper."""
    return asyncio.run(scrape_bmw_articles(max_articles, Path(output_dir)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMW Press Release Scraper")
    parser.add_argument("--scrape", action="store_true", help="Phase 2: Scrape content for collected URLs")
    parser.add_argument(
        "--target",
        "--max-articles",
        type=int,
        default=100,
        help="Target number of articles to collect/scrape",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()
    d_dir = Path(args.data_dir)

    # Default: Always collect
    asyncio.run(collect_urls_async(d_dir, target_count=args.target))

    # Optional: Scrape
    if args.scrape:
        asyncio.run(scrape_articles_async(d_dir, max_articles=args.target))
