"""Dropped model variant with one transformer block removed.

Creates a Qwen3 model with 35 layers instead of 36 for architecture comparison.
Uses Unsloth's FastLanguageModel for optimized training.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedTokenizer


def create_dropped_model(
    model_name: str = "Qwen/Qwen3-8B",
    max_seq_length: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
    layer_to_drop: int = 16,  # Drop middle layer
    use_lora: bool = True,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Create a Qwen3 model with one transformer block dropped.

    This creates a model with 35 layers instead of 36, reducing parameters
    by approximately 2.4%. Uses Unsloth's FastLanguageModel for optimized training.

    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        layer_to_drop: Index of the layer to drop (0-35)
        use_lora: Whether to use LoRA adapters

    Returns:
        Tuple of (dropped_model, tokenizer)
    """
    from unsloth import FastLanguageModel

    print(f"Creating dropped model (dropping layer {layer_to_drop})")

    # Load model with Unsloth optimizations (without LoRA first)
    # We need to modify the architecture before applying LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=False,
    )

    # Access the base model (unwrap if needed)
    base_model = model

    # Find the layers module list
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        layers = base_model.model.layers
        config = base_model.config
    elif hasattr(base_model, "layers"):
        layers = base_model.layers
        config = base_model.config
    else:
        raise ValueError(f"Could not find layers attribute in model {type(base_model)}")

    original_layer_count = len(layers)
    print(f"Original layers: {original_layer_count}")

    if layer_to_drop >= original_layer_count:
        raise ValueError(f"Layer index {layer_to_drop} out of bounds for model with {original_layer_count} layers")

    # Drop the layer from the ModuleList
    del layers[layer_to_drop]

    # Update config to reflect new layer count
    config.num_hidden_layers = len(layers)
    print(f"Layers after drop: {config.num_hidden_layers}")

    # Handle layer_types if present (e.g., Qwen3 has per-layer attention type config)
    if hasattr(config, "layer_types") and config.layer_types is not None:
        layer_types = list(config.layer_types)
        if len(layer_types) == original_layer_count:
            del layer_types[layer_to_drop]
            config.layer_types = layer_types
            print(f"Updated layer_types: {len(layer_types)} entries")

    # Adjust max_window_layers if it exceeds new layer count
    if hasattr(config, "max_window_layers") and config.max_window_layers > config.num_hidden_layers:
        config.max_window_layers = config.num_hidden_layers
        print(f"Adjusted max_window_layers to {config.max_window_layers}")

    if use_lora:
        # Now apply LoRA using Unsloth's optimized method
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Optimized gradient checkpointing
            random_state=42,
        )

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, tokenizer


def load_dropped_model_for_inference(
    model_path: str,
    base_model_name: str = "Qwen/Qwen3-8B",
    layer_to_drop: int = 16,
    max_seq_length: int = 4096,
) -> tuple[Any, PreTrainedTokenizer]:
    """Load a dropped model for inference.

    Handles both full fine-tuned models and LoRA adapters.

    Args:
        model_path: Path to the saved model checkpoint or adapter
        base_model_name: Base model name (used if loading LoRA adapters)
        layer_to_drop: Index of the layer that was dropped during training
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """

    from unsloth import FastLanguageModel

    # Check if it's a LoRA adapter (has adapter_config.json)
    is_lora = (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        print(f"Loading LoRA adapter from {model_path} onto dropped base model {base_model_name}")
        # 1. Load base model (without adapters initially)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )

        # 2. Drop the layer to match training architecture
        base_model = model
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            layers = base_model.model.layers
            config = base_model.config
        elif hasattr(base_model, "layers"):
            layers = base_model.layers
            config = base_model.config
        else:
            raise ValueError(f"Could not find layers attribute in model {type(base_model)}")

        if layer_to_drop < len(layers):
            print(f"Dropping layer {layer_to_drop} for inference")
            del layers[layer_to_drop]
            config.num_hidden_layers = len(layers)

            # Handle layer_types if present
            if hasattr(config, "layer_types") and config.layer_types is not None:
                layer_types = list(config.layer_types)
                if len(layer_types) > len(layers):  # Original count
                    del layer_types[layer_to_drop]
                    config.layer_types = layer_types

            if hasattr(config, "max_window_layers") and config.max_window_layers > config.num_hidden_layers:
                config.max_window_layers = config.num_hidden_layers

        # 3. Load the adapter
        model.load_adapter(model_path)

    else:
        print(f"Loading full fine-tuned dropped model from {model_path}")
        # Full fine-tuned model should have the correct config saved
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer
