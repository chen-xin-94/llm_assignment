#!/usr/bin/env python
"""Evaluate fine-tuned BMW LLM models.

Usage:
    # Fine-tuned local checkpoint with training config
    python scripts/evaluate.py --model checkpoints/dropped/final --train-config configs/dropped.yaml

    # HuggingFace model (no config needed)
    python scripts/evaluate.py --model Qwen/Qwen3-8B

    # Original model checkpoint
    python scripts/evaluate.py --model checkpoints/original/final --train-config configs/original.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from llm_assignment.evaluation.generate import generate_samples
from llm_assignment.evaluation.generate import get_default_prompts
from llm_assignment.evaluation.perplexity import compute_perplexity
from llm_assignment.evaluation.semantic_entropy import evaluate_semantic_entropy
from llm_assignment.evaluation.semantic_entropy import get_bmw_prompts
from llm_assignment.models import load_model_for_inference_auto


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BMW LLM models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Path to training config YAML (provides model_type, model_name, layer_to_drop, etc.)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/processed",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--skip-entropy",
        action="store_true",
        help="Skip semantic entropy evaluation (faster)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip sample generation",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking/reasoning mode. If enabled, reasoning is captured separately.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate for samples (default: 512)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length for evaluation (default: from config or model default)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BMW LLM MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")

    # Load config if provided
    config = None
    if args.train_config:
        config = load_config(args.train_config)
        print(f"Config: {args.train_config}")
        print(f"  model_type: {config.get('model_type', 'original')}")
        print(f"  model_name: {config.get('model_name', 'Qwen/Qwen3-8B')}")

    # Load model
    print("\nLoading model...")

    # Priority for max_seq_length: CLI arg > Training Config > Default 4096
    max_seq_length = args.max_seq_length
    if max_seq_length is None:
        max_seq_length = config["max_seq_length"] if config and "max_seq_length" in config else 4096

    model, tokenizer = load_model_for_inference_auto(
        model_path=args.model,
        config=config,
        max_seq_length=max_seq_length,
    )

    results = {"model_path": args.model}
    if config:
        results["config"] = args.train_config

    # Perplexity evaluation
    print("\n[1/3] Computing perplexity...")
    try:
        from datasets import load_from_disk

        dataset = load_from_disk(args.dataset_path)
        eval_dataset = dataset["eval"]
        print(f"Evaluation samples: {len(eval_dataset)}")

        # Decide max_length for perplexity
        # Priority: CLI arg > Config > Model Config > Default 4096
        max_length = args.max_seq_length
        if max_length is None:
            if config and "max_seq_length" in config:
                max_length = config["max_seq_length"]
            elif hasattr(model.config, "max_position_embeddings"):
                max_length = model.config.max_position_embeddings
                print(f"Detected max_position_embeddings from model: {max_length}")
            else:
                max_length = 4096

        print(f"Using max_length: {max_length}")
        ppl_results = compute_perplexity(model, tokenizer, eval_dataset, max_length=max_length)
        results["perplexity"] = ppl_results
        print(f"Perplexity: {ppl_results['perplexity']:.4f}")
    except Exception as e:
        print(f"Perplexity evaluation failed: {e}")
        results["perplexity"] = {"error": str(e)}

    # Semantic entropy evaluation
    if not args.skip_entropy:
        print("\n[2/3] Computing semantic entropy...")
        try:
            prompts = get_bmw_prompts()
            entropy_results = evaluate_semantic_entropy(model, tokenizer, prompts)
            results["semantic_entropy"] = {
                "mean_entropy": entropy_results["mean_entropy"],
                "max_entropy": entropy_results["max_entropy"],
                "min_entropy": entropy_results["min_entropy"],
            }
            print(f"Mean entropy: {entropy_results['mean_entropy']:.4f}")
        except Exception as e:
            print(f"Semantic entropy evaluation failed: {e}")
            results["semantic_entropy"] = {"error": str(e)}
    else:
        print("\n[2/3] Skipping semantic entropy...")

    # Sample generations
    if not args.skip_generation:
        print("\n[3/3] Generating samples...")
        try:
            # prompts = get_default_prompts()[:3]  # Use first 3 prompts for testing
            prompts = get_default_prompts()
            # Only use thinking tokens for pretrained models (no train-config)
            is_pretrained = args.train_config is None
            generations = generate_samples(
                model,
                tokenizer,
                prompts,
                max_new_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
                is_pretrained=is_pretrained,
            )
            results["generations"] = generations

            # Print a sample
            if generations:
                print(f"\nSample prompt: {generations[0]['prompt'][:80]}...")
                print(f"Response: {generations[0]['responses'][0][:200]}...")
        except Exception as e:
            print(f"Generation failed: {e}")
            results["generations"] = {"error": str(e)}
    else:
        print("\n[3/3] Skipping generation...")

    # Save results with model-specific filename
    if args.train_config:
        # Use config name (e.g., "dropped_lora" from "configs/dropped_lora.yaml")
        config_name = Path(args.train_config).stem
        output_file = output_dir / f"evaluation_results_{config_name}.json"
    else:
        # Use HuggingFace repo name (e.g., "Qwen3-8B" from "Qwen/Qwen3-8B")
        repo_name = args.model.split("/")[-1]
        output_file = output_dir / f"evaluation_results_{repo_name}.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if "perplexity" in results and "perplexity" in results["perplexity"]:
        print(f"Perplexity: {results['perplexity']['perplexity']:.4f}")
    if "semantic_entropy" in results and "mean_entropy" in results["semantic_entropy"]:
        print(f"Mean Semantic Entropy: {results['semantic_entropy']['mean_entropy']:.4f}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
