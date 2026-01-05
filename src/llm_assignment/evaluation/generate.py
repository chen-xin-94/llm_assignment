"""Sample generation for BMW-related prompts.

Generates sample responses from fine-tuned models for qualitative evaluation.
"""

from __future__ import annotations

import re

import torch


def _parse_thinking_response(text: str) -> tuple[str | None, str]:
    """Parse thinking blocks from Qwen3 response.

    Args:
        text: Raw model output text

    Returns:
        Tuple of (thinking_content, final_response)
        thinking_content is None if no thinking block found
    """
    # Pattern to match <think>...</think> blocks
    think_pattern = r"<think>(.*?)</think>"
    match = re.search(think_pattern, text, re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        # Remove the thinking block from the response
        response = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()
        return thinking, response
    return None, text.strip()


def generate_samples(
    model,
    tokenizer,
    prompts: list[str] | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    num_return_sequences: int = 1,
    enable_thinking: bool = False,
    is_pretrained: bool = True,
) -> list[dict]:
    """Generate sample responses for prompts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts (uses default BMW prompts if None)
        max_new_tokens: Maximum tokens to generate (default 512 to avoid truncation)
        temperature: Sampling temperature
        num_return_sequences: Number of responses per prompt
        enable_thinking: Whether to enable Qwen3's thinking/reasoning mode.
            If False (default), reasoning is disabled for direct answers.
            If True, reasoning is captured in separate 'thinking' field.
        is_pretrained: Whether this is a pretrained model (not fine-tuned).
            If True, adds /think or /no_think tokens to control reasoning.
            If False (fine-tuned), prompts match training format exactly.

    Returns:
        List of dictionaries with prompts and responses.
        If enable_thinking=True, each response dict also has a 'thinking' field.
    """
    if prompts is None:
        prompts = get_default_prompts(
            enable_thinking=enable_thinking,
            is_pretrained=is_pretrained,
        )

    model.eval()
    device = next(model.parameters()).device

    results = []

    for prompt in prompts:
        # Prompts from get_default_prompts() already include special tokens
        # (<|im_start|>, <|im_end|>), so we tokenize directly without apply_chat_template
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = []
        thinking_list = []

        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the response
            # Since skip_special_tokens=True, we need to find where the prompt ends
            # The prompt ends with "assistant\n" after token removal
            response = response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :].strip()

            # Parse thinking blocks if present
            thinking, final_response = _parse_thinking_response(response)
            responses.append(final_response)
            thinking_list.append(thinking)

        result = {
            "prompt": prompt,
            "responses": responses,
        }

        # Add thinking field if any thinking was captured
        if enable_thinking or any(t is not None for t in thinking_list):
            result["thinking"] = thinking_list

        results.append(result)

    return results


def get_default_prompts(enable_thinking: bool = False, is_pretrained: bool = True) -> list[str]:
    """Default BMW-related prompts for evaluation.

    These prompts:
    1. Match the exact training data format for maximum performance
    2. Ask about 2024-2025 specific facts that pre-trained models cannot answer
    3. Have verifiable answers in the BMW press releases dataset

    Args:
        enable_thinking: Controls Qwen3's thinking mode via soft switch tokens.
            - True: Adds /think token to enable reasoning (for pretrained Qwen3)
            - False: Adds /no_think token to disable reasoning (default)
        is_pretrained: Whether this is a pretrained model.
            - True: Adds thinking tokens (/think or /no_think)
            - False: No tokens added, prompts match fine-tuned training format
    """
    # Topics that are well-represented in the 2024-2025 press releases
    topics = [
        # Valentino Rossi + BMW partnership (2024-2025 specific, 92 mentions)
        "Valentino Rossi racing with BMW M4 GT3 in the 2025 FIA WEC",
        # Julie Mehretu Art Car (Le Mans 2024, 151 mentions)
        "Julie Mehretu and the BMW M Hybrid V8 Art Car for Le Mans 2024",
        # BMW M Hybrid V8 hypercar (2024+ knowledge, 512 mentions)
        "The BMW M Hybrid V8 hypercar competing in the FIA World Endurance Championship",
        # Neue Klasse platform (2024+ announcements, 232 mentions)
        "BMW Neue Klasse platform and the new BMW iX3",
        # BMW M2 Racing customer car (106 mentions)
        "The BMW M2 Racing customer race car specifications and features",
        # Oliver Zipse statements (151 mentions)
        "Oliver Zipse and BMW Group's electric vehicle strategy",
        # MINI John Cooper Works (2024 model updates)
        "The new MINI John Cooper Works with 170 kW and 231 PS",
        # Plant investments (recent news)
        "BMW Group investments at Plant Regensburg and Plant Landshut",
    ]

    # Only add thinking tokens for pretrained models
    # Fine-tuned models should see prompts matching their training format
    think_token = ("/think " if enable_thinking else "/no_think ") if is_pretrained else ""

    return [
        (
            "<|im_start|>system\n"
            "You are a BMW automotive expert assistant. "
            "Provide accurate, detailed information about BMW vehicles, technology, and company news.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{think_token}Tell me about: {topic}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for topic in topics
    ]
