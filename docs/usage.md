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

### Full Reproduction Pipeline
```bash
# Start local model servers (required for gpt2-xl and davinci-002)
make start-local-model-servers

# In a new terminal, run fast tier evaluation
make repro TIER=1
```

## Model Tiers

- **TIER=1**: Fast and cheap models for quick checks
  - openai/gpt2-xl, openai/gpt-3.5-turbo, anthropic/claude-3-5-haiku-20241022, openai/o4-mini-2025-04-16

- **TIER=2**: Mid-range models balancing cost and capability  
  - Includes TIER=1 plus: openai/davinci-002, openai/gpt-4-0314, openai/gpt-4-1106-preview, anthropic/claude-3-5-sonnet-20241022, anthropic/claude-3-7-sonnet-20250219, google/gemini-2.5-flash-preview-20250520

- **TIER=3**: Most powerful models for deep analysis
  - openai/gpt2-xl, openai/gpt-3.5-turbo, openai/o4-mini-2025-04-16, anthropic/claude-3-5-sonnet-20241022, anthropic/claude-opus-4-20250514, openai/o3-2025-04-16, openai/gpt-4-0613, openai/gpt-4-32k-0613

- **TIER=publication**: Models used in the blog post
  - openai/gpt2-xl, openai/davinci-002, openai/gpt-3.5-turbo, anthropic/claude-3-5-sonnet-20240620, anthropic/claude-3-5-haiku-20241022, anthropic/claude-3-5-sonnet-20241022, openai/o4-mini-2025-04-16, openai/o3-2025-04-16, google/gemini-2.5-pro-preview-06-05

## Available Datasets

- **cybashbench**: Command reflexes (1s-30s)
- **nl2bash**: Bash translation (4s-4min)
- **intercode-ctf**: Interactive challenges (10s-10min)
- **nyuctf**: University CTF challenges (2min-6h) 
- **cybench**: Professional CTF challenges (2min-25h)

## Commands

### Data Pipeline
```bash
# Process all datasets
make retrieve-all      # Retrieve all raw datasets
make prepare-all       # Prepare all datasets for evaluation  
make describe-all      # Generate descriptions for all datasets

# Process specific dataset
make retrieve DATASET=cybench
make prepare DATASET=nl2bash
make describe DATASET=intercode-ctf
```

### Benchmarking
```bash
# Run single benchmark
make bench DATASET=cybench MODEL=openai/gpt-4

# Run full reproduction pipeline
make repro TIER=1                    # Fast models
make repro TIER=2                    # Mid-range models  
make repro TIER=3                    # Powerful models
make repro TIER=publication          # Publication models

# Check benchmark progress
make progress                        # Check progress against publication models
```

### Local Model Servers

For models like gpt2-xl and davinci-002 that require local serving:

```bash
# Start all local servers in background
make start-local-model-servers

# Stop all local servers
make stop-local-model-servers

# Run individual servers in foreground (for debugging)
make run-gpt2xl-local              # GPT2-XL server on port 8000
make run-davinci-local             # Davinci-002 proxy on port 8001
```

### Analysis and Plotting
```bash
# Generate horizon plots
make plot                          # Default 50% success rate
make plot SUCCESS_RATE=25          # Custom success rate

# Review tasks (for development)
make review-cybashbench            # Interactive task review tool
```

### Utilities
```bash
# Get full help
make help                          # Shows all available commands

# Clean generated files
make clean                         # Clean all generated files
make clean_datasets               # Clean only dataset files
make clean_benchmarks             # Clean only benchmark results
make clean_plots                  # Clean only generated plots

# Run tests
make test                         # Run unit tests
```

## Examples

```bash
# Quick evaluation on fast models
make repro TIER=1

# Single model evaluation  
make bench DATASET=cybench MODEL=anthropic/claude-3-5-haiku-20241022

# Process specific dataset
make prepare DATASET=nl2bash

# Generate plots with custom success rate
make plot SUCCESS_RATE=75
```