# Running Benchmarks with Local Models

This guide explains how to run benchmarks using locally-served models like GPT-2.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for vLLM, but CPU mode is supported)
- ~2-4GB of free RAM for GPT-2

## Setup

1. **Install vLLM** (one-time setup):
   ```bash
   make setup-vllm
   ```

2. **Start the model server**:
   ```bash
   make start-gpt2
   ```
   
   This will start a vLLM server on `http://localhost:8000` serving GPT-2.
   Keep this terminal open - the server needs to stay running.

3. **Run benchmarks** (in a new terminal):
   ```bash
   # Run NL2Bash benchmark with GPT-2
   MODEL=openai/gpt2 make bench-nl2bash
   
   # Or with a specific number of tasks
   MODEL=openai/gpt2 NUM_TASKS=10 make bench-nl2bash
   ```

4. **Stop the server** when done:
   ```bash
   make stop-vllm
   ```

## Available Local Models

Currently configured models:
- `openai/gpt2` - GPT-2 base (124M parameters)
- `openai/gpt2-medium` - GPT-2 medium (355M parameters)
- `openai/gpt2-large` - GPT-2 large (774M parameters)
- `openai/gpt2-xl` - GPT-2 XL (1.5B parameters)

To use larger variants, modify the `make start-gpt2` command or create new targets:
```bash
# Start GPT-2 medium
PYTHONPATH=src: uv run python -m vllm.entrypoints.openai.api_server \
    --model gpt2-medium \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 1024
```

## How It Works

1. **vLLM** provides an OpenAI-compatible API endpoint for serving HuggingFace models
2. Our `local_models.py` module configures inspect_ai to use the local endpoint
3. When you specify `MODEL=openai/gpt2`, the system:
   - Detects it's a local model
   - Sets the appropriate environment variables
   - Validates the server is running
   - Routes requests to `http://localhost:8000`

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use: `lsof -i :8000`
- Try a different port by modifying the `--port` argument

### Out of memory errors
- Reduce `--max-model-len` (default 1024)
- Use a smaller model variant
- Add `--gpu-memory-utilization 0.8` to leave some GPU memory free

### Slow performance on CPU
- vLLM is optimized for GPU; CPU inference will be slower
- Consider using smaller models or fewer tasks for testing

### Model not found
- vLLM automatically downloads models from HuggingFace on first use
- Ensure you have internet connection for the first run
- Models are cached in `~/.cache/huggingface/`

## Adding New Local Models

To add support for a new model:

1. Edit `src/human_ttc_eval/core/local_models.py`
2. Add a new entry to `LOCAL_MODEL_CONFIGS`
3. Create a new Makefile target if desired

Example for LLaMA:
```python
"openai/llama-7b": {
    "base_url": "http://localhost:8000/v1",
    "api_key": "dummy",
    "model": "meta-llama/Llama-2-7b-hf",
    "provider": "openai"
}
``` 