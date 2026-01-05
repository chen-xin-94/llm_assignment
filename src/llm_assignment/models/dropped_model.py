"""Dropped model variant with one transformer block removed.

Wrapper around ModelFactory for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_assignment.models.factory import ModelFactory

if TYPE_CHECKING:
    pass


def create_dropped_model(
    model_name: str = "Qwen/Qwen3-8B",
    max_seq_length: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
    layer_to_drop: int = 16,
    use_lora: bool = True,
) -> tuple[Any, Any]:
    """Create a Qwen3 model with one transformer block dropped.

    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        layer_to_drop: Index of the layer to drop
        use_lora: Whether to use LoRA adapters

    Returns:
        Tuple of (model, tokenizer)
    """
    return ModelFactory.create_model(
        model_type="dropped",
        model_name=model_name,
        max_seq_length=max_seq_length,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        layer_to_drop=layer_to_drop,
    )


def load_dropped_model_for_inference(
    model_path: str,
    base_model_name: str = "Qwen/Qwen3-8B",
    layer_to_drop: int = 16,
    max_seq_length: int = 4096,
) -> tuple[Any, Any]:
    """Load a dropped model for inference.

    Args:
        model_path: Path to the saved model checkpoint or adapter
        base_model_name: Base model name (used if loading LoRA adapters)
        layer_to_drop: Index of the layer that was dropped during training
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    return ModelFactory.load_for_inference(
        model_path=model_path,
        model_type="dropped",
        base_model_name=base_model_name,
        layer_to_drop=layer_to_drop,
        max_seq_length=max_seq_length,
    )
