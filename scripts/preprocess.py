#!/usr/bin/env python
"""Preprocess BMW PDFs for Qwen3 training.

This script extracts text from downloaded PDFs, cleans it, filters it using an LLM (optional),
and formats it into Qwen3 ChatML format for training.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_assignment.data_engine.pipeline import preprocess_pdfs


def main():
    parser = argparse.ArgumentParser(description="Preprocess BMW PDFs for Qwen3 training")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory containing pdfs/ subdirectory (default: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Proportion for training set (default: 0.9)",
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["instruct", "article", "qa"],
        default="instruct",
        help="Format style for training (default: instruct)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["regex", "llm", "format", "all", "no-llm"],
        default="no-llm",
        help="Processing phase to run (default: no-llm)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai/gpt-oss-120b",
        help="LLM model for filtering (default: openai/gpt-oss-120b)",
    )
    args = parser.parse_args()

    preprocess_pdfs(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        style=args.style,
        seed=args.seed,
        phase=args.phase,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
