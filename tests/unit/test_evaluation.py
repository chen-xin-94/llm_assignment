"""Unit tests for evaluation module."""

from unittest.mock import MagicMock

import torch

from llm_assignment.evaluation.generate import _parse_thinking_response
from llm_assignment.evaluation.generate import get_default_prompts
from llm_assignment.evaluation.semantic_entropy import compute_semantic_entropy
from llm_assignment.evaluation.semantic_entropy import get_bmw_prompts


def test_compute_semantic_entropy_identical():
    """Test entropy with identical responses."""
    responses = ["The car is red.", "The car is red."]
    mock_embedder = MagicMock()
    # Mock cosine similarity = 1.0
    with torch.no_grad():
        mock_embedder.encode.return_value = torch.ones((2, 10))

    entropy = compute_semantic_entropy(responses, mock_embedder, entailment_threshold=0.8)
    assert entropy == 0.0


def test_compute_semantic_entropy_diverse():
    """Test entropy with diverse responses."""
    responses = ["Red car.", "Blue bike."]
    mock_embedder = MagicMock()
    # Mock embeddings that result in low similarity
    # similarity between [1,0] and [0,1] is 0
    mock_embedder.encode.return_value = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    entropy = compute_semantic_entropy(responses, mock_embedder, entailment_threshold=0.8)
    assert entropy > 0.0


def test_parse_thinking_response():
    """Test parsing thinking blocks."""
    text = "<think>I should say hello.</think> Hello world!"
    thinking, final = _parse_thinking_response(text)
    assert thinking == "I should say hello."
    assert final == "Hello world!"

    text_no_think = "Just a normal response."
    thinking, final = _parse_thinking_response(text_no_think)
    assert thinking is None
    assert final == "Just a normal response."


def test_get_default_prompts():
    """Test default prompt generation."""
    prompts = get_default_prompts(enable_thinking=True, is_pretrained=True)
    assert len(prompts) > 0
    assert "/think" in prompts[0]

    prompts_no_think = get_default_prompts(enable_thinking=False, is_pretrained=True)
    assert "/no_think" in prompts_no_think[0]

    prompts_ft = get_default_prompts(is_pretrained=False)
    assert "/think" not in prompts_ft[0]
    assert "/no_think" not in prompts_ft[0]


def test_get_bmw_prompts():
    """Test BMW prompt list."""
    prompts = get_bmw_prompts()
    assert len(prompts) > 0
    assert any("BMW Neue Klasse" in p for p in prompts)
