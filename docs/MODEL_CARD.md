---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: transformers
tags:
- unsloth
- qwen
- qwen3
- fine-tuning
- lora
- bmw
datasets:
- Moonxc/bmw-press-1k
model-index:
- name: Qwen3-8B-bmw-press
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: BMW Press 1K
      type: Moonxc/bmw-press-1k
    metrics:
    - type: perplexity
      value: 5.59
      name: Perplexity
---

# Qwen3-8B-BMW-Press

## Model Description

This is a fine-tuned version of **Qwen/Qwen3-8B** trained on the [BMW Press Releases Dataset (1K)](https://huggingface.co/datasets/Moonxc/bmw-press-1k). The model has been adapted to understand and generate text in the specific corporate style of BMW Group press releases.

It features multiple variants based on parameter-efficient fine-tuning (LoRA) and experimental architecture modifications (Layer Dropping and Pruning).

*   **Developed by:** [Moonxc](https://huggingface.co/Moonxc)
*   **Base model:** Qwen/Qwen3-8B
*   **Library:** Unsloth / Transformers / TRL
*   **License:** Apache 2.0

## Intended Use

This model is designed for:
*   **Corporate Content Generation**: Drafting press-release style announcements.
*   **Domain Knowledge**: Answering questions about BMW's specific vehicle lineup (up to 2025/2026), sustainability goals (`Neue Klasse`), and financial results.

## Results

### Training and Evaluation Loss

![Training Loss](results/train_loss.png)

![Evaluation Loss](results/eval_loss.png)

### Evaluation Metrics Summary

| Model | Test Perplexity ↓ | Mean Entropy ↓ |
|-------|--------------|----------------|
| Qwen3-8B (Baseline) | 9.49 | 0.00 |
| **original** | **5.59** | 0.73 |
| original_lora | 11.40 | 0.51 |
| dropped | 6.11 | 1.14 |
| dropped_lora | 15.59 | **0.50** |
| pruned | 11.60 | 1.93 |
| pruned_lora | 14.96 | 1.61 |

> **Note:** Lower perplexity indicates better language modeling. Lower mean entropy typically indicates more deterministic/confident generation. The baseline Qwen3-8B has 0.00 mean entropy because all generated responses fell into single semantic clusters.

### Qualitative Examples

**Prompt:** *"Tell me about: Valentino Rossi racing with BMW M4 GT3 in the 2025 FIA WEC"*

<details>
<summary><b>Pretrained Qwen3-8B (Baseline)</b> — ❌ Incorrect answer due to earlier knowledege cutoff</summary>

> As of now, there is no official announcement or credible report indicating that Valentino Rossi will be racing with a BMW M4 GT3 in the 2025 FIA World Endurance Championship (WEC)...

</details>

<details>
<summary><b>Lora fine-tuned Qwen3-8B</b> — ✅ BMW press release style with correct information</summary>

> **Valentino Rossi racing with BMW M4 GT3 in the 2025 FIA WEC**
>
> Munich. Valentino Rossi (ITA) will compete in the 2025 FIA World Endurance Championship (FIA WEC) as part of the BMW M Motorsport Hypercar programme. The two‑time World Champion will join the BMW M Team WRT line‑up for the 2025 season. He will share the cockpit of the BMW M Hybrid V8 with Robin Frijns (NED) and Raffaele Marciello (SUI)...

</details>

<details>
<summary><b>Pruned Qwen3-8B</b> — ❌ Degenerate output (too aggressive pruning)</summary>

> \*\*\*\*\*\*\*\*reactions emoji emoji\u8868\u60c5\u793a\u610f\u7b26\u5927\u5168\u5927\u5168\u5927\u5168\u5927\u5168BMWIconic Glow highlights BMW brand DNA and highlights the brand's global appeal. BMW M4 GT3: Powerful, super‑efficient, and versatile race car +++ BMW Individual paint finishes in the BMW iX2-60...

</details>

## Training Details

### Training Data
*   **Dataset**: ~1,000 BMW press releases.
*   **Format**: ChatML (Instruction Tuning).

### Hyperparameters
*   **Optimization**: Unsloth (FlashAttention-2, gradient checkpointing).
*   **LoRA**: Rank=16, Alpha=16.
*   **Learning Rate**: 2e-4.
*   **Warmup**: 3% of steps.
*   **Weight Decay**: 0.01.
*   **Packing**: Enabled.



## Usage

### Loading with Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Moonxc/Qwen3-8B-bmw-press"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {"role": "user", "content": "What is the Neue Klasse?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# Generate with parameters matching training/evaluation scripts
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)

# Decode only the new tokens
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## Hosting

This model was trained and uploaded using custom scripts in the [llm_assignment](https://github.com/chen-xin-94/llm_assignment) repository.
