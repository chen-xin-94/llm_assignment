# BMW LLM Fine-tuning Pipeline

End-to-end pipeline for fine-tuning Qwen3-8B on BMW press releases with model architecture comparison.

## Quick Start

```bash
# 1. Setup
uv venv --python 3.11.12 .venv
uv sync --all-groups.
uv run playwright install-deps
uv run playwright install
source .venv/bin/activate

# 2. Collect data (full pipeline: scrape + download PDFs + preprocess)
./scripts/scrape.sh --target 1000 --all

# 3. Train models
# Train original model
python scripts/train.py --config configs/original.yaml
# Train dropped model (one layer removed)
python scripts/train.py --config configs/dropped.yaml
# Train pruned model (layers truncated)
python scripts/train.py --config configs/pruned.yaml

# 4. Evaluate
# Fine-tuned local checkpoint with training config
python scripts/evaluate.py --model checkpoints/dropped/final --train-config configs/dropped.yaml

# HuggingFace model (no config needed)
python scripts/evaluate.py --model Qwen/Qwen3-8B

# Original model checkpoint
python scripts/evaluate.py --model checkpoints/original_lora/final --train-config configs/original.yaml
```

## Project Structure

```
src/
└── llm_assignment/
    ├── data_engine/                    # Data collection & preprocessing
    │   ├── scraper.py          # Crawl4AI scraper for BMW PressClub
    │   ├── pdf_downloader.py   # Download PDF attachments
    │   └── pdf_preprocessor.py # Convert to Qwen3 format
    ├── models/                  # Model loading
    │   ├── model.py             # Unsloth + LoRA setup, unified inference loader
    │   ├── dropped_model.py     # 35-layer variant (one layer dropped)
    │   └── pruned_model.py      # N-layer variant (layers truncated)
    ├── training/trainer.py      # Training with SFTTrainer
    └── evaluation/              # Metrics & generation
        ├── perplexity.py
        ├── semantic_entropy.py
        └── generate.py

scripts/                     # Entry points
data/                        # Raw PDFs & processed dataset
results/                     # Evaluation outputs
```

## PDF Preprocessing

The preprocessing pipeline supports **phase-based processing** with optional LLM filtering.

### Pipeline Phases

```
PDF → regex phase → preprocessed/*.txt → llm phase → llm_filtered/*.txt → format phase → processed/
```

| Phase | Description | Output |
|-------|-------------|--------|
| `regex` | Extract text, apply regex cleaning | `data_engine/preprocessed/*.txt` |
| `llm` | LLM-based header removal & quality filter | `data_engine/llm_filtered/*.txt` |
| `format` | Convert to Qwen3 ChatML format | `data_engine/processed/train.jsonl` |
| `all` | Full pipeline: regex → llm → format | `data_engine/processed/train.jsonl` |
| `no-llm` | Basic pipeline: regex → format (skips LLM) | `data_engine/processed/train.jsonl` |

### Usage

```bash
# Run basic pipeline (regex + format, no LLM) - Default
python -m llm_assignment.data_engine.pdf_preprocessor

# Run full pipeline with LLM filtering (requires GPU)
python -m llm_assignment.data_engine.pdf_preprocessor --phase all

# Run only regex phase
python -m llm_assignment.data_engine.pdf_preprocessor --phase regex

# Run only LLM phase on existing regex output
python -m llm_assignment.data_engine.pdf_preprocessor --phase llm

# Use smaller model for LLM phase
python -m llm_assignment.data_engine.pdf_preprocessor --phase llm --llm-model openai/gpt-oss-20b
```

### CLI Arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--phase` | `regex`, `llm`, `format`, `all`, `no-llm` | `no-llm` | Processing phase to run |
| `--llm-model` | model name | `openai/gpt-oss-120b` | HuggingFace model for filtering |
| `--train-ratio` | float | `0.8` | Train/eval split ratio |
| `--style` | `instruct`, `article`, `qa` | `instruct` | Output format style |

## Model Parameter Analysis

Analyze parameter distribution across model components:

```bash
# Basic analysis
python scripts/analyze_params.py --model Qwen/Qwen3-8B

# With per-layer breakdown
python scripts/analyze_params.py --model Qwen/Qwen3-8B --detailed
```

Example output for Qwen3-8B:
```
  Total parameters:     8.191B
  Parameters per layer: 192.95M
  Number of layers:     36
  Layer params total:   6.946B (84.8%)
  Non-layer params:     1.245B (15.2%)

  If you remove 1 layer:
    New total:          7.998B
    Reduction:          192.95M (2.36%)
```

## Design Choices

### Model Selection
- **Qwen3-8B**: SOTA 8B model with 36 transformer layers
- **Reduced variant**: 35 layers (~3% fewer parameters)
- **Training**: LoRA with 4-bit quantization via Unsloth (2-5x faster)

### Data Collection
- **Crawl4AI + Playwright**: Handles JavaScript-heavy BMW PressClub
- **PDF-based**: Download official press release PDFs for cleaner text

### Evaluation Metrics
- **Perplexity**: Standard LM quality metric (lower = better)
- **Semantic Entropy**: Uncertainty in meaning (higher = more hallucination risk)
- **Sample Generations**: Qualitative comparison of outputs

## Model Comparison

| Metric | Original (36L) | Dropped (35L) | Pruned (24L) |
|--------|----------------|---------------|---------------|
| Parameters | ~8.19B | ~8.00B | ~5.88B |
| Layers | 36 | 35 | 24 |
| Reduction | — | ~2.4% | ~28% |
| Perplexity | — | — | — |

*(Results filled after training)*

## Trade-offs Discussion

### Model Size vs Quality
- **Dropped**: Removing 1 layer reduces parameters by ~2.4%
- **Pruned**: Removing 12 layers (keep 24) reduces parameters by ~28%
- Trade-off between size reduction and quality degradation

### What to Investigate Next
- Layer importance analysis (which layer matters most to drop)
- Knowledge distillation from full to pruned model
- Quantization comparisons (4-bit vs 8-bit)

## Dependencies

```
torch, unsloth, transformers, trl, peft
crawl4ai, playwright, pypdf
sentence-transformers, tensorboard, wandb
```

## License

For interview assignment purposes only.
