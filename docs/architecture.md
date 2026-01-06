# Architecture Documentation

## Model Factory Pattern

The repository uses a Factory Pattern to manage different model variants while maximizing code reuse.

### Core Components

1. **`src/llm_assignment/models/base.py`**:
    * Handles low-level model loading using `unsloth.FastLanguageModel`.
    * Applies LoRA adapters with configurable rank/alpha.

2. **`src/llm_assignment/models/factory.py`**:
    * **`ModelFactory`**: The central class for creating and loading models.
    * **`create_model()`**: Used during training. Loads a base model and applies structural modifications (dropping/pruning layers) *before* applying LoRA.
    * **`load_for_inference()`**: Used during evaluation. Handles loading adapters onto structurally modified base models.

3. **Wrappers (`model.py`, `dropped_model.py`, `pruned_model.py`)**:
    * Maintain backward compatibility with older scripts.
    * Delegate calls to `ModelFactory`.

### Structural Modifications

* **Dropped Model**: Removes a specific layer index from `model.layers`. Useful for analyzing single-layer importance.
* **Pruned Model**: Truncates the model after $N$ layers. Useful for significant parameter reduction.

## Configuration System

The configuration uses a simplified inheritance mechanism to avoid duplication. More on the configuration system can be found in [configuration.md](configuration.md).

1. **Base Config (`configs/base.yaml`)**:
    * Defines default hyperparameters (learning rate, batch size, epochs).
    * Sets common project settings (W&B project, data path).

2. **Specific Configs**:
    * Use `_extends: "base.yaml"` to inherit defaults.
    * Override only specific fields (e.g., `model_type`, `lora.r`).
    * The `TrainingConfig.from_yaml` method in `trainer.py` handles the merging logic.

## Training Flow

1. `scripts/train.py` loads the YAML config.
2. `TrainingConfig` parses arguments and merges with base config.
3. `trainer.py` calls `load_model(config)`.
4. `load_model` delegates to the appropriate wrapper module based on `config.model_type`.
5. Wrapper delegates to `ModelFactory`.
6. `SFTTrainer` initializes with the created model and tokenizer.
