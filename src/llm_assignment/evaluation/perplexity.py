"""Perplexity evaluation for fine-tuned models.

Computes perplexity on the evaluation set for model comparison.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling


def compute_perplexity(
    model,
    tokenizer,
    eval_dataset,
    batch_size: int = 8,
    max_length: int = 4096,
) -> dict:
    """Compute perplexity on evaluation dataset.

    Args:
        model: The language model
        tokenizer: The tokenizer
        eval_dataset: HuggingFace dataset with 'text' field
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length

    Returns:
        Dictionary with perplexity and related metrics
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize dataset
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    tokenized.set_format("torch")

    # Create dataloader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized, batch_size=batch_size, collate_fn=data_collator)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch_gpu)
            loss = outputs.loss

            # Count non-padding tokens
            num_tokens = (batch_gpu["labels"] != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
    }
