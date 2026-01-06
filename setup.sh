#!/bin/bash
# BMW LLM Pipeline - Environment Setup
# Run: chmod +x setup.sh && ./setup.sh

set -e

echo "🔧 Installing uv (if needed)..."
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "🐍 Creating Python 3.11 virtual environment..."
uv venv --python 3.11.12 .venv

echo "📦 Installing dependencies..."
uv sync --all-groups

echo "🎭 Installing Playwright browsers..."
uv run playwright install-deps
uv run playwright install

echo "✅ Setup complete! Activate with: source .venv/bin/activate"
