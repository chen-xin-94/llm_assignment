"""Semantic Entropy computation for hallucination detection.

Uses Sentence-BERT embeddings and NLI clustering to measure
uncertainty in model outputs (NOT LLM-as-a-judge).
"""

from __future__ import annotations

from collections import Counter
import math

import torch
from tqdm import tqdm


def compute_semantic_entropy(
    responses: list[str],
    embedder,
    nli_model=None,
    entailment_threshold: float = 0.8,
) -> float:
    """Compute semantic entropy over a set of responses.

    Clusters responses by semantic similarity and computes
    entropy over the cluster distribution.

    Args:
        responses: List of generated responses
        embedder: SentenceTransformer model for embeddings
        nli_model: Optional NLI pipeline for entailment checking
        entailment_threshold: Threshold for entailment classification

    Returns:
        Semantic entropy value (higher = more uncertain)
    """
    if len(responses) <= 1:
        return 0.0

    # Get embeddings
    embeddings = embedder.encode(responses, convert_to_tensor=True)

    # Compute pairwise cosine similarity
    similarities = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2)

    # Simple clustering: group responses with similarity > threshold
    n = len(responses)
    clusters = list(range(n))  # Each response starts in its own cluster

    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i, j] > entailment_threshold:
                # Merge clusters
                old_cluster = clusters[j]
                new_cluster = clusters[i]
                for k in range(n):
                    if clusters[k] == old_cluster:
                        clusters[k] = new_cluster

    # Count cluster sizes
    cluster_counts = Counter(clusters)
    total = len(responses)

    # Compute entropy
    entropy = 0.0
    for count in cluster_counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log(prob)

    return entropy


def evaluate_semantic_entropy(
    model,
    tokenizer,
    prompts: list[str],
    n_samples: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> dict:  # pragma: no cover
    """Evaluate semantic entropy for a set of prompts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to evaluate
        n_samples: Number of responses to generate per prompt
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary with entropy scores per prompt
    """
    from sentence_transformers import SentenceTransformer

    # Load embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    model.eval()
    device = next(model.parameters()).device

    results = []

    for prompt in tqdm(prompts, desc="Evaluating semantic entropy"):
        # Generate multiple responses
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        responses = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            response = response[len(prompt) :].strip()
            responses.append(response)

        # Compute semantic entropy
        entropy = compute_semantic_entropy(responses, embedder)

        results.append(
            {
                "prompt": prompt,
                "responses": responses,
                "semantic_entropy": entropy,
            }
        )

    # Aggregate statistics
    entropies = [r["semantic_entropy"] for r in results]
    return {
        "prompts_evaluated": len(prompts),
        "samples_per_prompt": n_samples,
        "mean_entropy": sum(entropies) / len(entropies) if entropies else 0,
        "max_entropy": max(entropies) if entropies else 0,
        "min_entropy": min(entropies) if entropies else 0,
        "results": results,
    }


def get_bmw_prompts() -> list[str]:
    """Get a set of BMW-related prompts for evaluation."""
    return [
        "<|im_start|>user\nWhat is the BMW Neue Klasse?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nTell me about BMW's electric vehicle strategy.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is BMW M division known for?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe BMW's approach to sustainable manufacturing.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the BMW iX3?<|im_end|>\n<|im_start|>assistant\n",
    ]
