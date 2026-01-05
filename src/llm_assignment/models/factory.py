"""Model factory for creating and loading different model variants."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from llm_assignment.models.base import apply_lora
from llm_assignment.models.base import load_base_model

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

ModelType = Literal["original", "dropped", "pruned"]


def _modify_layers_dropped(model: Any, layer_to_drop: int) -> None:
    """Remove a specific layer from the model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        config = model.config
    elif hasattr(model, "layers"):
        layers = model.layers
        config = model.config
    else:
        raise ValueError(f"Could not find layers attribute in model {type(model)}")

    original_count = len(layers)
    if layer_to_drop >= original_count:
        raise ValueError(f"Layer {layer_to_drop} out of bounds ({original_count} layers)")

    logger.info(f"Dropping layer {layer_to_drop} (of {original_count})")
    del layers[layer_to_drop]
    config.num_hidden_layers = len(layers)

    # Clean up layer_types if present
    if hasattr(config, "layer_types") and config.layer_types is not None:
        layer_types = list(config.layer_types)
        if len(layer_types) == original_count:
            del layer_types[layer_to_drop]
            config.layer_types = layer_types

    if hasattr(config, "max_window_layers") and config.max_window_layers > config.num_hidden_layers:
        config.max_window_layers = config.num_hidden_layers


def _modify_layers_pruned(model: Any, keep_layers: int) -> None:
    """Truncate layers after keep_layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        config = model.config
    elif hasattr(model, "layers"):
        layers = model.layers
        config = model.config
    else:
        raise ValueError(f"Could not find layers attribute in model {type(model)}")

    original_count = len(layers)
    if keep_layers >= original_count or keep_layers <= 0:
        raise ValueError(f"Invalid keep_layers={keep_layers} (total={original_count})")

    logger.info(f"Pruning to keep first {keep_layers} layers (removing {original_count - keep_layers})")

    # Remove from end
    del layers[keep_layers:]
    config.num_hidden_layers = len(layers)

    if hasattr(config, "layer_types") and config.layer_types is not None:
        layer_types = list(config.layer_types)
        config.layer_types = layer_types[:keep_layers]

    if hasattr(config, "max_window_layers") and config.max_window_layers > config.num_hidden_layers:
        config.max_window_layers = config.num_hidden_layers


class ModelFactory:
    """Factory for creating and loading models."""

    @staticmethod
    def create_model(
        model_type: ModelType,
        model_name: str = "Qwen/Qwen3-8B",
        max_seq_length: int = 4096,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        layer_to_drop: int = 16,
        keep_layers: int = 24,
    ) -> tuple[Any, Any]:  # pragma: no cover
        """Create a model for training with specified architecture adjustments."""
        model, tokenizer = load_base_model(model_name, max_seq_length)

        # Apply structural changes BEFORE LoRA
        if model_type == "dropped":
            _modify_layers_dropped(model, layer_to_drop)
        elif model_type == "pruned":
            _modify_layers_pruned(model, keep_layers)
        elif model_type != "original":
            raise ValueError(f"Unknown model type: {model_type}")

        # Apply LoRA
        if use_lora:
            model = apply_lora(model, r=lora_r, alpha=lora_alpha)

        # Log parameter counts
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total:,}, Trainable: {trainable:,}")

        return model, tokenizer

    @staticmethod
    def load_for_inference(
        model_path: str,
        model_type: ModelType = "original",
        base_model_name: str = "Qwen/Qwen3-8B",
        max_seq_length: int = 4096,
        layer_to_drop: int = 16,
        keep_layers: int = 24,
    ) -> tuple[Any, Any]:  # pragma: no cover
        """Load a model for inference, handling structural changes if loading adapters."""
        from unsloth import FastLanguageModel

        path_obj = Path(model_path)
        is_adapter = (path_obj / "adapter_config.json").exists()

        if is_adapter:
            logger.info(f"Loading adapter from {model_path} onto {model_type} base")
            # Load base
            model, tokenizer = load_base_model(base_model_name, max_seq_length)

            # Apply structural changes to base
            if model_type == "dropped":
                _modify_layers_dropped(model, layer_to_drop)
            elif model_type == "pruned":
                _modify_layers_pruned(model, keep_layers)

            # Load adapter
            model.load_adapter(model_path)
        else:
            logger.info(f"Loading full model from {model_path}")
            # Full model should have correct config saved
            model, tokenizer = load_base_model(model_path, max_seq_length)

        FastLanguageModel.for_inference(model)
        return model, tokenizer
