"""Model loading utilities for Qwen3-8B with Unsloth.

Provides functions to load the base model with LoRA configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from peft import PeftModel
from transformers import PreTrainedTokenizer
from unsloth import FastLanguageModel

if TYPE_CHECKING:
    from llm_assignment.training.trainer import TrainingConfig


def load_model_for_training(
    model_name: str = "Qwen/Qwen3-8B",
    max_seq_length: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    use_lora: bool = True,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Load Qwen3-8B with Unsloth optimizations and LoRA.

    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length for training
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        use_lora: Whether to use LoRA adapters

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load base model with Unsloth optimizations
    # Using fp32 for mixed precision training (AMP handles bf16/fp16 during forward pass)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=False,
    )

    if use_lora:
        # Apply LoRA adapters
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
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Optimized gradient checkpointing
            random_state=42,
        )

    return model, tokenizer


def load_model_for_inference(
    model_path: str,
    max_seq_length: int = 4096,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Load a fine-tuned model for inference.

    Args:
        model_path: Path to the saved model checkpoint
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    # Enable faster inference
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def load_model_for_inference_auto(
    model_path: str,
    config: TrainingConfig | dict | str | None = None,
    max_seq_length: int | None = None,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Load a model for inference using training config for parameters.

    Args:
        model_path: Path to checkpoint or HuggingFace model ID
        config: TrainingConfig object, dict (from yaml.safe_load), or path to config yaml.
                If provided, reads model_type, model_name, layer_to_drop,
                keep_layers, max_seq_length from config.
        max_seq_length: Maximum sequence length (if provided, overrides config)

    Returns:
        Tuple of (model, tokenizer)
    """
    from pathlib import Path

    from llm_assignment.models.dropped_model import load_dropped_model_for_inference
    from llm_assignment.models.pruned_model import load_pruned_model_for_inference

    # Parse config if provided
    model_type = "original"
    base_model_name = "Qwen/Qwen3-8B"
    layer_to_drop = 16
    keep_layers = 24
    config_max_seq_length = None

    if config is not None:
        # If config is a string path, load it
        if isinstance(config, str):
            from llm_assignment.training.trainer import TrainingConfig

            config_obj = TrainingConfig.from_yaml(config)
            model_type = config_obj.model_type
            base_model_name = config_obj.model_name
            layer_to_drop = config_obj.layer_to_drop
            keep_layers = config_obj.keep_layers
            config_max_seq_length = config_obj.max_seq_length
        elif isinstance(config, dict):
            # Direct dict from yaml.safe_load
            model_type = config.get("model_type", "original")
            base_model_name = config.get("model_name", "Qwen/Qwen3-8B")
            layer_to_drop = config.get("layer_to_drop", 16)
            keep_layers = config.get("keep_layers", 24)
            config_max_seq_length = config.get("max_seq_length")
        else:
            # TrainingConfig object
            model_type = config.model_type
            base_model_name = config.model_name
            layer_to_drop = config.layer_to_drop
            keep_layers = config.keep_layers
            config_max_seq_length = config.max_seq_length

    # Priority for max_seq_length: argument > config > default 4096
    if max_seq_length is None:
        max_seq_length = config_max_seq_length or 4096

    model_path_obj = Path(model_path)
    is_local = model_path_obj.exists()

    # For HuggingFace repo IDs (not local paths), always use original loader
    if not is_local:
        print(f"Loading from HuggingFace: {model_path}")
        return load_model_for_inference(model_path, max_seq_length)

    print(f"Loading local checkpoint: {model_path}")
    print(f"Model type: {model_type}")

    # Dispatch to appropriate loader based on model type
    if model_type == "original":
        return load_model_for_inference(model_path, max_seq_length)

    if model_type == "dropped":
        return load_dropped_model_for_inference(
            model_path=model_path,
            base_model_name=base_model_name,
            layer_to_drop=layer_to_drop,
            max_seq_length=max_seq_length,
        )

    if model_type == "pruned":
        return load_pruned_model_for_inference(
            model_path=model_path,
            base_model_name=base_model_name,
            keep_layers=keep_layers,
            max_seq_length=max_seq_length,
        )

    raise ValueError(f"Unknown model type '{model_type}'. Must be 'original', 'dropped', or 'pruned'.")
