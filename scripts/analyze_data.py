#!/usr/bin/env python
"""Analyze token lengths of the processed dataset."""

import argparse
from pathlib import Path

from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def analyze_dataset(dataset_path: str, model_name: str, output_dir: str):
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    print(f"Loading tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    all_lengths = []

    # Process both train and eval splits
    for split in ["train", "eval"]:
        if split not in dataset:
            continue

        print(f"Processing split: {split}")
        data = dataset[split]
        lengths = []

        for item in tqdm(data, desc=f"Tokenizing {split}"):
            if "text" in item:
                text = item["text"]
                tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
                lengths.append(len(tokens))

        all_lengths.extend(lengths)

        # Split stats
        print(f"\nStats for {split} split:")
        print(f"  Count: {len(lengths)}")
        print(f"  Min: {np.min(lengths)}")
        print(f"  Max: {np.max(lengths)}")
        print(f"  Mean: {np.mean(lengths):.2f}")
        print(f"  Median: {np.median(lengths):.2f}")

    # Overall stats
    print(f"\nOverall Stats (Total {len(all_lengths)} samples):")
    print(f"  Min: {np.min(all_lengths)}")
    print(f"  Max: {np.max(all_lengths)}")
    print(f"  Mean: {np.mean(all_lengths):.2f}")
    print(f"  Median: {np.median(all_lengths):.2f}")

    # Percentiles
    for p in [90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(all_lengths, p):.2f}")

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=50, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of Token Lengths ({model_name})")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Add vertical lines for Mean and Max
    plt.axvline(
        np.mean(all_lengths), color="red", linestyle="dashed", linewidth=1, label=f"Mean: {np.mean(all_lengths):.0f}"
    )
    plt.axvline(
        np.median(all_lengths),
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"Median: {np.median(all_lengths):.0f}",
    )
    if np.max(all_lengths) > 0:
        plt.axvline(
            np.max(all_lengths), color="orange", linestyle="dotted", linewidth=1, label=f"Max: {np.max(all_lengths)}"
        )

    plt.legend()

    output_path = Path(output_dir) / "token_length_histogram.png"
    plt.savefig(output_path)
    print(f"\nHistogram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset token lengths")
    parser.add_argument("--dataset-path", default="data/processed", help="Path to processed dataset")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B", help="Model name for tokenizer")
    parser.add_argument("--output-dir", default="results", help="Directory to save histogram")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    analyze_dataset(args.dataset_path, args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
