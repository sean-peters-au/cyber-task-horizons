# Quick Start Guide

## Prerequisites
- Python 3.9+
- Docker (for challenge environments)
- Git
- API keys for model providers

## Setup

```bash
# Install uv and dependencies
curl -fsSL https://get.uv.dev | bash
uv sync

# Set up environment variables
cp .env.template .env
# Edit .env with your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)

# Clone external datasets and METR analysis tools
make third-party
```

## Quick Start

### Test with Fast Model
```bash
# Start local gpt2-xl server (required for default model)
make start-local-model-server MODEL=openai/gpt2-xl

# In a new terminal, run the evaluation
make repro TIER=1
```

### Model Tiers
- **TIER=1**: openai/gpt2-xl, openai/gpt-3.5-turbo, anthropic/claude-3-5-haiku-20241022, openai/o4-mini-2025-04-16
- **TIER=2**: openai/davinci-002, openai/gpt-4-0314, openai/gpt-4-1106-preview, anthropic/claude-3-5-sonnet-20241022, anthropic/claude-3-7-sonnet-20250219, google/gemini-2.5-flash-preview-20250520
- **TIER=3**: google/gemini-2.5-pro-20250605, anthropic/claude-opus-4-20250514, openai/o3-2025-04-16, openai/gpt-4-0613, openai/gpt-4-32k-0613

### Single Dataset Evaluation
```bash
# Evaluate specific model on specific dataset
make bench DATASET=cybench MODEL=openai/o4-mini-2025-04-16
```

### Available Datasets
- **cybashbench**: Command reflexes (0.6s-15s)
- **nl2bash**: Bash translation (4s-4min)
- **intercode-ctf**: Interactive challenges (10s-10min)
- **nyuctf**: University CTF challenges (2min-6h) 
- **cybench**: Professional CTF challenges (2min-25h)

## More Options

For additional commands and options:
```bash
make help
```

This shows all available targets including dataset processing, custom model evaluation, and plotting options.