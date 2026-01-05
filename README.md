# BMW LLM Fine-tuning Pipeline

End-to-end pipeline for fine-tuning Qwen3-8B on BMW press releases with model architecture comparison.

## Quick Start
```bash
# 1. Setup
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create & activate venv
uv venv --python 3.11.12 .venv
uv sync --all-groups
uv run playwright install-deps
uv run playwright install
source .venv/bin/activate

# 2. Collect data
# Default: Runs full pipeline (URLs -> Scrape Content -> Download PDFs)
python scripts/scrape.py --target 1000 --all

# Options:
# Only collect URLs (metadata only)
python scripts/scrape.py --target 1000 
# Only scrape content for existing URLs
python scripts/scrape.py --scrape
# Only download PDFs for existing articles
python scripts/scrape.py --download-pdfs

# 3. Preprocess data
# Full pipeline with Regex cleaning + LLM filtering (requires GPU) + dataset formatting
python scripts/preprocess.py --phase all --llm-model openai/gpt-oss-120b

# Options:
# Regex cleaning + dataset formatting (No LLM filtering)
python scripts/preprocess.py --phase no-llm
# Only regex filtering
python scripts/preprocess.py --phase regex
# Use specific formatting style (e.g., Q&A format)
python scripts/preprocess.py --phase no-llm --style qa

# 4. Train models
# Set GPU devices
export CUDA_VISIBLE_DEVICES=0,1
# Original model with LoRA
python scripts/train.py --config configs/original_lora.yaml

# Variants:
# Dropped layer with LoRA
python scripts/train.py --config configs/dropped_lora.yaml
# Pruned model with LoRA
python scripts/train.py --config configs/pruned_lora.yaml

# No LoRA (Full Fine-tuning):
python scripts/train.py --config configs/original.yaml
python scripts/train.py --config configs/dropped.yaml
python scripts/train.py --config configs/pruned.yaml

# 5. Evaluate
# Set GPU devices
export CUDA_VISIBLE_DEVICES=0
# Local checkpoint
python scripts/evaluate.py --model checkpoints/dropped_lora/final --train-config configs/dropped_lora.yaml

# HuggingFace model
python scripts/evaluate.py --model Qwen/Qwen3-8B --max-seq-length 4096
```

## Project Structure

```
src/
└── llm_assignment/
    ├── data_engine/            # Data collection & preprocessing
    │   ├── scraper.py          # Crawl4AI scraper
    │   ├── pdf_downloader.py   # Attachment downloader
    │   ├── extraction.py       # PDF text extraction & cleaning
    │   ├── formatting.py       # Qwen3 formatting utils
    │   └── pipeline.py         # Orchestration logic
    ├── models/                 # Model factory & wrappers
    │   ├── factory.py          # Centralized model creation
    │   ├── base.py             # Shared unsloth/lora logic
    │   ├── model.py            # Original model wrapper
    │   ├── dropped_model.py    # Dropped layer variant wrapper
    │   └── pruned_model.py     # Pruned variant wrapper
    ├── training/trainer.py     # SFTTrainer logic & config
    └── evaluation/             # Metrics & generation
        ├── perplexity.py
        ├── semantic_entropy.py
        └── generate.py

scripts/                        # Entry points
├── scrape.py                   # Data collection pipeline
├── preprocess.py               # Data processing pipeline
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
└── utils/                   # Utility scripts
    ├── analyze_params.py
    ├── analyze_data.py
    └── token_counter.py

configs/                        # YAML configurations
├── base.yaml                   # Shared defaults
├── original.yaml               # Original model config
├── dropped.yaml                # Dropped model config
└── pruned.yaml                 # Pruned model config
```

## Configuration

The project uses a hierarchical configuration system. `configs/base.yaml` contains shared defaults for training, logging, and data. Specific configurations inherit from base via the `_extends` key.

Example (`configs/dropped_lora.yaml`):
```yaml
_extends: "base.yaml"

model_type: "dropped"
layer_to_drop: 16
use_lora: true

lora:
  r: 64
  alpha: 16
```

## Preprocessing Pipeline

The data engine supports phase-based processing via `scripts/preprocess.py`:

| Phase | Description |
|-------|-------------|
| `regex` | Extract text from PDFs and apply regex cleaning |
| `llm` | Use an LLM to filter nonsensical content (optional) |
| `format` | Format cleaned text into Qwen3 ChatML style |
| `no-llm` | Run `regex` and `format` phases (Default) |
| `all` | Run all phases (`regex` -> `llm` -> `format`) |

## Model Variants

| Model Type | Description | Config |
|------------|-------------|--------|
| `original` | Standard Qwen3-8B (36 layers) | `configs/original.yaml` |
| `dropped` | Single transformer layer removed (e.g., layer 16) | `configs/dropped.yaml` |
| `pruned` | Truncated to first N layers (e.g., first 24) | `configs/pruned.yaml` |

## Dependencies

- **Core**: `torch`, `unsloth`, `transformers`, `trl`, `peft`
- **Data**: `crawl4ai`, `playwright`, `pypdf`
- **Utils**: `wandb`, `tensorboard`, `pyyaml`, `pytest`

## License

For interview assignment purposes only.
