# Configuration System

The project uses a hierarchical configuration system to manage hyperparameters and model settings efficiently. This system allows for sharing common defaults while enabling specific model variants to override only what is necessary.

## inheritance Mechanism

The configuration system supports inheritance via the special `_extends` key.

- **`_extends`**: Specifies the relative path to a base configuration file.
- **Merging Logic**:
    1. The base configuration is loaded first.
    2. The current configuration is merged on top of it.
    3. **Shallow Merge**: For most top-level keys, the current config overwrites the base config.
    4. **Nested Merge (`lora`)**: The `lora` section is handled specially. It performs a merge where keys present in the current config's `lora` section override those in the base, but other `lora` keys from the base are preserved.

### Example

**Base Config (`configs/base.yaml`)**:

```yaml
learning_rate: 2e-4
num_train_epochs: 3
lora:
  r: 16
  alpha: 16
```

**Specific Config (`configs/dropped_lora.yaml`)**:

```yaml
_extends: "base.yaml"

model_type: "dropped"
# Overrides r, keeps alpha from base
lora:
  r: 64
```

## Configuration Schema

The configuration is parsed into a `TrainingConfig` dataclass defined in `src/llm_assignment/training/trainer.py`.

### Core Settings

| Key | Default | Description |
| :--- | :--- | :--- |
| `model_name` | `Qwen/Qwen3-8B` | Base model identifier (HuggingFace path). |
| `model_type` | `original` | Type of model variant (`original`, `dropped`, `pruned`). |
| `max_seq_length` | `4096` | Maximum sequence length for tokenization. |
| `output_dir` | `checkpoints` | Directory to save model checkpoints. |

### Model Variants

- **`dropped`**: Uses `dropped_model.py`.
  - `layer_to_drop`: Index of the transformer layer to remove (default: 16).
- **`pruned`**: Uses `pruned_model.py`.
  - `keep_layers`: Number of initial layers to keep (default: 24).

### LoRA Settings

Configured under the `lora` key:

| Key | Default | Description |
| :--- | :--- | :--- |
| `use_lora` | `true` | Whether to apply LoRA adapters. |
| `lora.r` | `16` | LoRA rank. |
| `lora.alpha` | `16` | LoRA alpha scaling factor. |

### Training Hyperparameters

| Key | Default | Description |
| :--- | :--- | :--- |
| `learning_rate` | `2e-4` | Initial learning rate. |
| `num_train_epochs` | `3` | Number of training epochs. |
| `per_device_train_batch_size` | `4` | Batch size per GPU. |
| `gradient_accumulation_steps` | `4` | Number of steps to accumulate gradients. |
| `warmup_ratio` | `0.03` | Fraction of steps for warmup. |
| `weight_decay` | `0.01` | Weight decay factor. |
| `gradient_checkpointing` | `true` | Enable gradient checkpointing to save VRAM. |

### Logging & Evaluation

| Key | Default | Description |
| :--- | :--- | :--- |
| `logging_steps` | `10` | Frequency of logging to W&B/console. |
| `eval_steps` | `50` | Frequency of evaluation on validation set. |
| `save_steps` | `100` | Frequency of saving checkpoints. |
| `wandb_project` | `bmw-llm-finetuning`| Weights & Biases project name. |

## Parameter Loading

The loading logic is implemented in `TrainingConfig.from_yaml()`:

```python
@classmethod
def from_yaml(cls, yaml_path: str | Path) -> TrainingConfig:
    # ... loads yaml ...
    if "_extends" in config_dict:
        # ... loads base config ...
        # ... merges configs ...
        # ... special handling for 'lora' dict ...
    return cls(**config_dict)
```
