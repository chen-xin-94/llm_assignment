"""LLM-based content filtering for PDF preprocessing.

Uses openai/gpt-oss-120b to clean and validate extracted PDF text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Pipeline

logger = logging.getLogger(__name__)


# Default system prompt for content filtering
SYSTEM_PROMPT = """You are a document cleaning assistant for BMW press releases. Your task is to clean the text content extracted from PDF.

Instructions:
1. FIND the article title in the content and REMOVE everything before it (headers like "Topic", "Media Information", "Press Information", "MINI", "Company", company headers, dates, etc.)
2. REMOVE any remaining boilerplate: contact info, social media links, page numbers, "If you have any questions" sections, spokesperson info, repeated headers
3. REMOVE pdf header/footer patterns like random appearance of "Topic xxx" "Subject xxx" "Date xxx" "Page xxx" in the content
4. VALIDATE the content is:
   - Not empty or just bullet points (like "• \\n• \\n• \\n•")
   - Not garbled/corrupted text (random unicode, encoding errors)
   - In the expected language (English for English titles, German for German titles)
   - Has meaningful content (more than just a few words)

If content is VALID after cleaning, return ONLY the cleaned text, nothing else.
If content is INVALID after cleaning, return exactly: REJECT: [reason]
"""

USER_MSG_TEMPLATE = """Title: {title}

Raw Content:
{content}
"""


class LLMFilter:
    """LLM-based content filtering using openai/gpt-oss-120b.

    Cleans and validates extracted PDF text by:
    - Removing header noise before the article title
    - Removing remaining boilerplate not caught by regex
    - Detecting and rejecting low-quality content
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        device: str = "auto",
        max_new_tokens: int = 4096,
    ) -> None:
        """Initialize the LLM filter.

        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('auto', 'cuda', 'cpu')
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline: Pipeline | None = None

    def load_model(self) -> Pipeline:
        """Lazy load the model pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers package required. Install with: pip install transformers accelerate torch"
            ) from e

        logger.info(f"Loading model: {self.model_name}")

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            dtype="auto",
            device_map=self.device,
            trust_remote_code=True,
        )

        logger.info("Model loaded successfully")
        return self._pipeline

    def filter_content(
        self,
        text: str,
        title: str,
    ) -> dict:
        """Filter and clean content using LLM.

        Args:
            text: Raw extracted text from PDF
            title: Article title

        Returns:
            Dict with keys:
                - "content": Cleaned text or None if rejected
                - "reasoning": Reasoning text (analysis) or None
                - "rejected": Boolean indicating if content was rejected
                - "reason": Rejection reason if rejected, else None
        """
        if not text or not text.strip():
            return {
                "content": None,
                "rejected": True,
                "reason": "Empty content",
            }

        # Truncate very long content to avoid token limits
        max_input_chars = 8000
        truncated = text[:max_input_chars] if len(text) > max_input_chars else text

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_MSG_TEMPLATE.format(title=title, content=truncated),
            },
        ]

        pipe = self.load_model()

        result = pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )

        generated = result[0]["generated_text"].strip()
        reasoning = None

        # Parse reasoning (analysis) and final content (assistantfinal)
        # Expected format: "analysis ... assistantfinal ..."
        if "assistantfinal" in generated:
            parts = generated.split("assistantfinal", 1)
            raw_reasoning = parts[0]
            final_content = parts[1].strip()

            # Clean up reasoning (remove "analysis" keyword if present)
            if "analysis" in raw_reasoning:
                reasoning = raw_reasoning.split("analysis", 1)[1].strip()
            else:
                reasoning = raw_reasoning.strip()

            generated = final_content

        # Check if content was rejected
        if generated.startswith("REJECT:"):
            reason = generated[7:].strip()
            return {
                "content": None,
                "reasoning": reasoning,
                "rejected": True,
                "reason": reason,
            }

        # Append any remaining content that was truncated
        if len(text) > max_input_chars:
            # Try to find where the cleaned content ends in original
            # and append the rest
            generated = generated + text[max_input_chars:]

        return {
            "content": generated,
            "reasoning": reasoning,
            "rejected": False,
            "reason": None,
        }

    def filter_batch(
        self,
        items: list[dict],
        batch_size: int = 4,
    ) -> list[dict]:
        """Filter a batch of items.

        Args:
            items: List of dicts with 'text' and 'title' keys
            batch_size: Batch size for inference

        Returns:
            List of result dicts (content, reasoning, rejected, reason)
        """
        if not items:
            return []

        pipe = self.load_model()

        # Prepare inputs
        inputs = []
        indices_to_process = []
        results = [None] * len(items)
        max_input_chars = 8000

        for idx, item in enumerate(items):
            text = item.get("text", "")
            title = item.get("title", "")

            if not text or not text.strip():
                results[idx] = {
                    "content": None,
                    "reasoning": None,
                    "rejected": True,
                    "reason": "Empty content",
                }
                continue

            truncated = text[:max_input_chars] if len(text) > max_input_chars else text

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_MSG_TEMPLATE.format(title=title, content=truncated),
                },
            ]
            inputs.append(messages)
            indices_to_process.append(idx)

        if not inputs:
            return results

        # Run batch inference
        # Note: pipeline works as iterator or list
        outputs = pipe(
            inputs,
            batch_size=batch_size,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )

        for i, output in enumerate(outputs):
            idx = indices_to_process[i]
            original_text = items[idx]["text"]

            generated = output[0]["generated_text"].strip()
            reasoning = None

            # Parse reasoning (analysis) and final content (assistantfinal)
            if "assistantfinal" in generated:
                parts = generated.split("assistantfinal", 1)
                raw_reasoning = parts[0]
                final_content = parts[1].strip()

                if "analysis" in raw_reasoning:
                    reasoning = raw_reasoning.split("analysis", 1)[1].strip()
                else:
                    reasoning = raw_reasoning.strip()

                generated = final_content

            # Check rejection
            if generated.startswith("REJECT:"):
                results[idx] = {
                    "content": None,
                    "reasoning": reasoning,
                    "rejected": True,
                    "reason": generated[7:].strip(),
                }
            else:
                # Append truncated content if needed
                if len(original_text) > max_input_chars:
                    generated = generated + original_text[max_input_chars:]

                results[idx] = {
                    "content": generated,
                    "reasoning": reasoning,
                    "rejected": False,
                    "reason": None,
                }
        return results


def create_llm_filter(
    model_name: str = "openai/gpt-oss-120b",
    device: str = "auto",
) -> LLMFilter:
    """Factory function to create an LLMFilter instance.

    Args:
        model_name: HuggingFace model name
        device: Device to run on

    Returns:
        LLMFilter instance
    """
    return LLMFilter(model_name=model_name, device=device)
