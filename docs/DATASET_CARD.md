---
license: apache-2.0
task_categories:
- text-generation
- question-answering
language:
- en
size_categories:
- n<1K
tags:
- bmw
- corporate
- press-release
- automotive
pretty_name: BMW Press Releases Dataset (1K)
---

# BMW Press Releases Dataset (1K)

## Dataset Description

*   **Repository:** [Moonxc/bmw-press-1k](https://huggingface.co/Moonxc/bmw-press-1k)
*   **Source:** [BMW Group PressClub](https://www.press.bmwgroup.com/global/)
*   **Language:** English
*   **Size:** ~1,000 processed press releases

### Dataset Summary

This dataset consists of approximately 1,000 press releases scraped from the official BMW Group PressClub. It focuses on recent corporate news, vehicle launches (especially EVs and Neue Klasse), financial results, and sustainability initiatives. The data has been processed to filter out non-informative content (like simple photo descriptions) and formatted into Qwen/ChatML style for instruction tuning.

This dataset was created for the "LLM Fine-tuning Pipeline" assignment, specifically to fine-tune a Qwen3-8B model to adopt the corporate tone and factual knowledge of BMW.

## Dataset Structure

The dataset contains the complete data pipeline artifacts, organized into the following directories:

*   **`all_articles.json`**: Metadata for all collected articles (URLs, titles, dates).
*   **`scraped/`**: Raw HTML content scraped from the press site.
*   **`pdfs/`**: Raw PDF attachments downloaded from the articles.
*   **`preprocessed_regex/`**: text extracted from PDFs and cleaned via Regex.
*   **`preprocessed_llm/`**: Text content after LLM-based filtering (removing irrelevant files).
*   **`processed/`**: Final, formatted datasets ready for training:
    *   `train_instruct.jsonl`
    *   `val_instruct.jsonl`

### Data Instances

**Processed Data (`processed/`)** follows the ChatML format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Tell me about the BMW iX5 Hydrogen pilot fleet."
    },
    {
      "role": "assistant",
      "content": "Munich. The BMW Group is actively driving the transformation... [Full Press Release Content]"
    }
  ]
}
```

## Data Collection & Processing

### 1. Scraping
*   **Tools:** Custom Python scraper using `Crawl4AI`.
*   **Source:** `press.bmwgroup.com/global`.
*   **Method:** Async scraping of article URLs followed by PDF attachment downloading.

### 2. Preprocessing Pipeline
The raw PDFs went through a multi-stage cleaning pipeline:
1.  **Text Extraction**: `pypdf` extracted raw text from PDF attachments.
2.  **Regex Cleaning**: Removed headers, footers, page numbers, and standard boilerplate text.
3.  **LLM Filtering**: Used `gpt-oss-120b` (simulated) to filter out "empty" content files that contained only images or media contacts without substantial text.
4.  **Formatting**: Converted valid text into User/Assistant pairs. The "User" prompt is derived from the article title, and the "Assistant" response is the full article text.

## Intended Use

*   **Fine-tuning**: Ideal for Domain Adaptation of SLMs (Small Language Models) like Qwen, Llama, or Mistral.
*   **RAG**: Can be used as a knowledge base for Retrieval-Augmented Generation systems focused on BMW.

## Limitations

*   **Knowledge Cutoff**: Contains press releases up to the date of scraping (early 2026 or late 2025).
*   **Domain Specific**: Highly specific to BMW's corporate voice; not suitable for general-purpose chat.
