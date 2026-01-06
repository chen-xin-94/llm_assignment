# Evaluation

The evaluation pipeline computes metrics to compare model quality across different architectures and training configurations.

## Metrics

### 1. Perplexity

Measures how well the model predicts the evaluation dataset.

- **Implementation**: `src/llm_assignment/evaluation/perplexity.py`
- **Method**: Computes cross-entropy loss over all tokens in the eval set, then converts to perplexity via $\text{PPL} = e^{\text{avg\_loss}}$.
- **Lower is better**: Indicates the model assigns higher probability to correct tokens.

### 2. Semantic Entropy

Measures uncertainty and potential for hallucination by analyzing response consistency.

- **Implementation**: `src/llm_assignment/evaluation/semantic_entropy.py`
- **Method**:
  1. Generate $n$ responses (default: 8) per prompt using temperature sampling.
  2. Embed responses using Sentence-BERT (`all-MiniLM-L6-v2`).
  3. Cluster responses by cosine similarity (threshold: 0.8).
  4. Compute entropy over cluster distribution: $H = -\sum p_i \log(p_i)$
- **Lower is better**: Responses that cluster into fewer groups indicate more consistent, confident outputs.

### 3. Sample Generation

Generates responses to predefined BMW-related prompts for qualitative comparison.

- **Implementation**: `src/llm_assignment/evaluation/generate.py`
- **Prompts**: Questions about BMW Neue Klasse, electric vehicles, M division, sustainability, etc.
- **Supports Thinking Mode**: With `--enable-thinking`, captures Qwen3's reasoning process separately.

## Usage

```bash
# Evaluate a fine-tuned checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --model checkpoints/dropped_lora/final \
    --train-config configs/dropped_lora.yaml

# Evaluate base HuggingFace model
python scripts/evaluate.py --model Qwen/Qwen3-8B --max-seq-length 4096

# Skip slow metrics
python scripts/evaluate.py --model checkpoints/original_lora/final \
    --train-config configs/original_lora.yaml \
    --skip-entropy \
    --skip-generation
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--model` | Path to checkpoint or HuggingFace model ID |
| `--train-config` | Training config YAML (provides model_type, layer_to_drop, etc.) |
| `--dataset-path` | Path to eval dataset (default: `data/processed`) |
| `--output-dir` | Results directory (default: `results/`) |
| `--skip-entropy` | Skip semantic entropy (faster) |
| `--skip-generation` | Skip sample generation |
| `--enable-thinking` | Enable Qwen3 reasoning mode |
| `--max-tokens` | Max tokens for generation (default: 512) |
| `--max-seq-length` | Max sequence length for perplexity (default: from config) |

## Output

Results are saved as JSON:

```
results/
├── evaluation_results_dropped_lora.json
├── evaluation_results_original_lora.json
└── evaluation_results_Qwen3-8B.json
```

Example output structure:
```json
{
  "model_path": "checkpoints/dropped_lora/final",
  "config": "configs/dropped_lora.yaml",
  "perplexity": {
    "perplexity": 3.2145,
    "avg_loss": 1.1678,
    "total_tokens": 125000
  },
  "semantic_entropy": {
    "mean_entropy": 0.45,
    "max_entropy": 1.2,
    "min_entropy": 0.0
  },
  "generations": [...]
}
```
