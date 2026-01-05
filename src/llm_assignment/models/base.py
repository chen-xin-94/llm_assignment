"""Shared model loading utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from unsloth import FastLanguageModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_base_model(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool = False,
    dtype: Any = None,
) -> tuple[Any, Any]:  # pragma: no cover
    """Load the base model using Unsloth.

    Args:
        model_name: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit precision
        dtype: Data type (None for auto)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def apply_lora(
    model: Any,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> Any:  # pragma: no cover
    """Apply LoRA adapters to the model.

    Args:
        model: Base model
        r: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout probability
        target_modules: List of modules to target (default: Qwen dict)

    Returns:
        Model with LoRA adapters
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    logger.info(f"Applying LoRA (r={r}, alpha={alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model
