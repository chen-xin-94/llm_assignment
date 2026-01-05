#!/usr/bin/env python3
"""Analyze parameter counts for each layer/block in a model.

Usage:
    python scripts/analyze_params.py --model Qwen/Qwen3-8B
    python scripts/analyze_params.py --model Qwen/Qwen3-8B --detailed
"""

import argparse

import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM


def count_parameters(module: torch.nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def format_params(num: int) -> str:
    """Format parameter count with appropriate suffix."""
    if num >= 1e9:
        return f"{num / 1e9:.3f}B"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def analyze_model(model_name: str, detailed: bool = False):
    """Analyze parameter distribution in a model."""

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")
    print(f"{'=' * 60}\n")

    # Load config first to get info without loading weights
    config = AutoConfig.from_pretrained(model_name)
    print("Model config:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Num key-value heads: {config.num_key_value_heads}")
    print(f"  - Num hidden layers: {config.num_hidden_layers}")
    print(f"  - Vocab size: {config.vocab_size}")
    print()

    # Load model (meta tensors to avoid loading weights into memory)
    print("Loading model structure...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # Analyze top-level components
    print(f"\n{'=' * 60}")
    print("TOP-LEVEL COMPONENTS")
    print(f"{'=' * 60}")

    component_params = {}

    # Get main components
    if hasattr(model, "model"):
        base_model = model.model

        # Embeddings
        if hasattr(base_model, "embed_tokens"):
            params = count_parameters(base_model.embed_tokens)
            component_params["embed_tokens"] = params
            print(f"  embed_tokens:     {format_params(params):>12}")

        # Layers
        if hasattr(base_model, "layers"):
            total_layer_params = count_parameters(base_model.layers)
            component_params["layers (total)"] = total_layer_params
            print(f"  layers (total):   {format_params(total_layer_params):>12}  ({len(base_model.layers)} layers)")

        # Final norm
        if hasattr(base_model, "norm"):
            params = count_parameters(base_model.norm)
            component_params["norm"] = params
            print(f"  norm:             {format_params(params):>12}")

    # LM head
    if hasattr(model, "lm_head"):
        params = count_parameters(model.lm_head)
        component_params["lm_head"] = params
        print(f"  lm_head:          {format_params(params):>12}")

    total_params = count_parameters(model)
    print(f"\n  TOTAL:            {format_params(total_params):>12}")

    # Analyze per-layer breakdown
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers

        print(f"\n{'=' * 60}")
        print("PER-LAYER BREAKDOWN")
        print(f"{'=' * 60}")

        # Analyze first layer in detail
        first_layer = layers[0]
        layer_params = count_parameters(first_layer)

        print(f"\nSingle layer parameters: {format_params(layer_params)}")
        print(f"Layer params x {len(layers)} = {format_params(layer_params * len(layers))}")
        print()

        # Breakdown of single layer
        print("Components in each decoder layer:")
        layer_breakdown = {}

        for name, module in first_layer.named_children():
            params = count_parameters(module)
            layer_breakdown[name] = params
            print(f"  {name:20s}: {format_params(params):>12}")

        if detailed:
            print(f"\n{'=' * 60}")
            print("DETAILED LAYER-BY-LAYER")
            print(f"{'=' * 60}")

            for i, layer in enumerate(layers):
                params = count_parameters(layer)
                print(f"  Layer {i:2d}: {format_params(params):>12}")

        # Self-attention breakdown
        if hasattr(first_layer, "self_attn"):
            attn = first_layer.self_attn
            print(f"\n{'=' * 60}")
            print("SELF-ATTENTION BREAKDOWN")
            print(f"{'=' * 60}")

            for name, module in attn.named_children():
                if hasattr(module, "weight"):
                    params = module.weight.numel()
                    shape = tuple(module.weight.shape)
                    print(f"  {name:12s}: {format_params(params):>10}  shape={shape}")

        # MLP breakdown
        if hasattr(first_layer, "mlp"):
            mlp = first_layer.mlp
            print(f"\n{'=' * 60}")
            print("MLP BREAKDOWN")
            print(f"{'=' * 60}")

            for name, module in mlp.named_children():
                if hasattr(module, "weight"):
                    params = module.weight.numel()
                    shape = tuple(module.weight.shape)
                    print(f"  {name:12s}: {format_params(params):>10}  shape={shape}")

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_params = count_parameters(layers[0])
        num_layers = len(layers)
        non_layer_params = total_params - (layer_params * num_layers)

        print(f"\n  Total parameters:     {format_params(total_params)}")
        print(f"  Parameters per layer: {format_params(layer_params)}")
        print(f"  Number of layers:     {num_layers}")
        print(
            f"  Layer params total:   {format_params(layer_params * num_layers)} ({layer_params * num_layers / total_params * 100:.1f}%)"
        )
        print(
            f"  Non-layer params:     {format_params(non_layer_params)} ({non_layer_params / total_params * 100:.1f}%)"
        )
        print("\n  If you remove 1 layer:")
        print(f"    New total:          {format_params(total_params - layer_params)}")
        print(f"    Reduction:          {format_params(layer_params)} ({layer_params / total_params * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze model parameter distribution")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-layer breakdown",
    )

    args = parser.parse_args()
    analyze_model(args.model, args.detailed)


if __name__ == "__main__":
    main()
