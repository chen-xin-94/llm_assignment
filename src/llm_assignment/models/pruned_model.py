"""Pruned model variant with layers truncated after a specified layer.

Wrapper around ModelFactory for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llm_assignment.models.factory import ModelFactory

if TYPE_CHECKING:
    pass


def create_pruned_model(
    model_name: str = "Qwen/Qwen3-8B",
    max_seq_length: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 16,
    keep_layers: int = 24,
    use_lora: bool = True,
) -> tuple[Any, Any]:
    """Create a pruned Qwen3 model with layers truncated after a cutoff.

    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        keep_layers: Number of layers to keep
        use_lora: Whether to use LoRA adapters

    Returns:
        Tuple of (model, tokenizer)
    """
    return ModelFactory.create_model(
        model_type="pruned",
        model_name=model_name,
        max_seq_length=max_seq_length,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        keep_layers=keep_layers,
    )


def load_pruned_model_for_inference(
    model_path: str,
    base_model_name: str = "Qwen/Qwen3-8B",
    keep_layers: int = 24,
    max_seq_length: int = 4096,
) -> tuple[Any, Any]:
    """Load a pruned model for inference.

    Args:
        model_path: Path to the saved model checkpoint or adapter
        base_model_name: Base model name (used if loading LoRA adapters)
        keep_layers: Number of layers to keep (matching training configuration)
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    return ModelFactory.load_for_inference(
        model_path=model_path,
        model_type="pruned",
        base_model_name=base_model_name,
        keep_layers=keep_layers,
        max_seq_length=max_seq_length,
    )
