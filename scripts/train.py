#!/usr/bin/env python
"""Training script for BMW LLM fine-tuning.

Usage:
    python scripts/train.py --config configs/original.yaml
    python scripts/train.py --config configs/reduced.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BMW LLM model with specified configuration")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/original.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for training."""
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"Loading configuration from: {config_path}")

    # Import trainer components
    from llm_assignment.training.trainer import TrainingConfig
    from llm_assignment.training.trainer import train_model

    # Load configuration from YAML
    config = TrainingConfig.from_yaml(config_path)

    print(f"Model type: {config.model_type}")
    print(f"Model name: {config.model_name}")
    print(f"Output directory: {config.output_dir}/{config.model_type}")

    # Start training
    print("\n" + "=" * 50)
    print(f"Starting training for {config.model_type.upper()} model")
    print("=" * 50)

    metrics = train_model(config)

    # Print results
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 45)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:<30} {value:>15.4f}")
        else:
            print(f"{metric:<30} {value:>15}")


if __name__ == "__main__":
    main()
