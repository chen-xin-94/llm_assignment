# Data Preprocessing Pipeline

The preprocessing pipeline converts raw PDF press releases into a clean, formatted dataset for instruction tuning.

## Pipeline Architecture

The pipeline is split into three modular phases orchestrated by `src/llm_assignment/data_engine/pipeline.py`.

### 1. Extraction Phase (`extraction.py`)
*   **Input**: Raw PDFs in `data/pdfs/`.
*   **Logic**:
    *   Uses `pypdf` to extract text.
    *   Applies regex-based cleaning to remove:
        *   Headers/Footers ("Media Information", "Contact")
        *   Page numbers
        *   Legal disclaimers
        *   Boilerplate text ("The BMW Group...")
    *   Normalizes whitespace and hyphenation.
*   **Output**: Cleaned text files in `data/preprocessed_regex/`.

### 2. LLM Filtering Phase (`llm_filter.py`)
*   **Input**: Text files from Phase 1.
*   **Logic**:
    *   Uses a local or API-based LLM (e.g., GPT-4 or local Qwen) to validate content quality.
    *   Filters out files that are:
        *   Too short.
        *   Not relevant (e.g., just tables or lists).
        *   Malformed.
*   **Output**: High-quality text files in `data/preprocessed_llm/`.

### 3. Formatting Phase (`formatting.py`)
*   **Input**: Text files from Phase 2 (or Phase 1 if LLM skipped).
*   **Logic**:
    *   Formats text into Qwen3 ChatML structure:
        ```
        <|im_start|>system
        You are a BMW automotive expert...
        <|im_start|>user
        Tell me about: {Title}
        <|im_start|>assistant
        {Content}
        ```
    *   Splits data into Train (90%) and Eval (10%).
*   **Output**: HuggingFace Dataset saved to `data/processed/`.

## Usage

Run the pipeline via the CLI script:

```bash
# Standard run (fast, no LLM filtering)
python scripts/preprocess.py --phase no-llm

# Full run with LLM filtering
python scripts/preprocess.py --phase all --llm-model openai/gpt-4o

# Change formatting style
python scripts/preprocess.py --phase no-llm --style qa
```
