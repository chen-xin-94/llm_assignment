"""Pruned model variant with layers truncated after a specified layer.

Creates a Qwen3 model with fewer layers by removing all layers after the cutoff.
For example, pruning at layer 24 creates a 24-layer model from the original 36.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedTokenizer


def create_pruned_model(
    model_name: str = "Qwen/Qwen3-8B",
    max_seq_length: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
    keep_layers: int = 24,  # Keep first N layers, prune the rest
    use_lora: bool = True,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Create a pruned Qwen3 model with layers truncated after a cutoff.

    This removes all layers after `keep_layers`, creating a shallower model.
    For example, keep_layers=24 on Qwen3-8B (36 layers) creates a 24-layer model,
    reducing parameters by approximately 33%.

    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        keep_layers: Number of layers to keep (prune all layers after this)
        use_lora: Whether to use LoRA adapters

    Returns:
        Tuple of (pruned_model, tokenizer)
    """
    from unsloth import FastLanguageModel

    print(f"Creating pruned model (keeping first {keep_layers} layers)")

    # Load model with Unsloth optimizations (without LoRA first)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=False,
    )

    # Access the base model layers
    base_model = model

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

    if keep_layers <= 0:
        raise ValueError(f"keep_layers must be positive, got {keep_layers}")
    if keep_layers >= original_layer_count:
        raise ValueError(f"keep_layers ({keep_layers}) must be less than original layer count ({original_layer_count})")

    # Remove all layers after keep_layers
    # We need to delete from the end to avoid index shifting issues
    layers_to_remove = original_layer_count - keep_layers
    for _ in range(layers_to_remove):
        del layers[keep_layers]  # Always delete at keep_layers index

    # Update config to reflect new layer count
    config.num_hidden_layers = len(layers)
    print(f"Pruned layers: {config.num_hidden_layers} (removed {layers_to_remove} layers)")

    # Handle layer_types if present (e.g., Qwen3 has per-layer attention type config)
    if hasattr(config, "layer_types") and config.layer_types is not None:
        layer_types = list(config.layer_types)
        if len(layer_types) == original_layer_count:
            config.layer_types = layer_types[:keep_layers]
            print(f"Updated layer_types: {len(config.layer_types)} entries")

    # Adjust max_window_layers if it exceeds new layer count
    if hasattr(config, "max_window_layers") and config.max_window_layers > config.num_hidden_layers:
        config.max_window_layers = config.num_hidden_layers
        print(f"Adjusted max_window_layers to {config.max_window_layers}")

    if use_lora:
        # Apply LoRA using Unsloth's optimized method
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
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    reduction_pct = (1 - total_params / 8_191_000_000) * 100  # Approx original size
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter reduction: ~{reduction_pct:.1f}%")

    return model, tokenizer


def load_pruned_model_for_inference(
    model_path: str,
    base_model_name: str = "Qwen/Qwen3-8B",
    keep_layers: int = 24,
    max_seq_length: int = 4096,
) -> tuple[Any, PreTrainedTokenizer]:
    """Load a pruned model for inference.

    Handles both full fine-tuned models and LoRA adapters.

    Args:
        model_path: Path to the saved model checkpoint or adapter
        base_model_name: Base model name (used if loading LoRA adapters)
        keep_layers: Number of layers to keep (matching training configuration)
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth import FastLanguageModel

    # Check if it's a LoRA adapter (has adapter_config.json)
    is_lora = (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        print(f"Loading LoRA adapter from {model_path} onto pruned base model {base_model_name}")
        # 1. Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )

        # 2. Prune layers
        base_model = model
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            layers = base_model.model.layers
            config = base_model.config
        elif hasattr(base_model, "layers"):
            layers = base_model.layers
            config = base_model.config
        else:
            raise ValueError(f"Could not find layers attribute in model {type(base_model)}")

        original_layer_count = len(layers)
        if keep_layers < original_layer_count:
            print(f"Pruning to {keep_layers} layers for inference")
            layers_to_remove = original_layer_count - keep_layers
            for _ in range(layers_to_remove):
                del layers[keep_layers]
            config.num_hidden_layers = len(layers)

            if hasattr(config, "layer_types") and config.layer_types is not None:
                layer_types = list(config.layer_types)
                if len(layer_types) > keep_layers:
                    config.layer_types = layer_types[:keep_layers]

            if hasattr(config, "max_window_layers") and config.max_window_layers > config.num_hidden_layers:
                config.max_window_layers = config.num_hidden_layers

        # 3. Load adapter
        model.load_adapter(model_path)

    else:
        print(f"Loading full fine-tuned pruned model from {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer
