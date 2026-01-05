from datasets import load_from_disk
from transformers import AutoTokenizer

try:
    dataset = load_from_disk("data/processed_regex")
    train = dataset["train"]

    print(f"Train rows: {len(train)}")

    # Simple whitespace token estimation first to be fast
    total_chars = sum(len(x["text"]) for x in train)
    print(f"Total characters: {total_chars}")
    print(f"Estimated tokens (char/4): {total_chars / 4}")

    # If small enough, use actual tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    total_tokens = 0
    for item in train:
        total_tokens += len(tokenizer.encode(item["text"]))

    print(f"Actual Total Tokens: {total_tokens}")

except Exception as e:
    print(f"Error: {e}")
