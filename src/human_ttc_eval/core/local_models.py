"""
Configuration and utilities for running local models with inspect_ai.

This module provides support for:
1. Running models like GPT-2 locally through OpenAI-compatible API servers (vLLM, etc)
2. Configuring OpenAI completion models (like davinci-002) as "local" models to work
   around inspect_ai's lack of native support for the completions API endpoint
"""

import os
import logging
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# Mapping of model aliases to their configurations
LOCAL_MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "openai/gpt2": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "dummy",  # vLLM doesn't require real API key
        "model": "gpt2",  # Actual HuggingFace model name
        "provider": "openai"  # Use OpenAI provider with custom base URL
    },
    "openai/gpt2-medium": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "dummy",
        "model": "gpt2-medium",
        "provider": "openai"
    },
    "openai/gpt2-large": {
        "base_url": "http://localhost:8000/v1", 
        "api_key": "dummy",
        "model": "gpt2-large",
        "provider": "openai"
    },
    "openai/gpt2-xl": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "dummy",
        "model": "gpt2-xl",
        "provider": "openai"
    },
    # OpenAI completion models configured as "local" to work around inspect_ai limitations
    "openai/davinci-002": {
        "base_url": "http://localhost:8001/v1",  # Our proxy server
        "api_key": "dummy",  # Proxy doesn't need real key (it uses OPENAI_API_KEY internally)
        "model": "davinci-002",
        "provider": "openai"
    }
}


def configure_local_model(model_name: str) -> bool:
    """
    Configure environment variables for local model if needed.
    
    Args:
        model_name: Model identifier (e.g., "openai/gpt2")
        
    Returns:
        True if local model was configured, False otherwise
    """
    if model_name not in LOCAL_MODEL_CONFIGS:
        return False
    
    config = LOCAL_MODEL_CONFIGS[model_name]
    
    # Set environment variables that inspect_ai will use
    os.environ["OPENAI_BASE_URL"] = config["base_url"]
    os.environ["OPENAI_API_KEY"] = config["api_key"]
    
    logger.info(f"Configured local model endpoint for {model_name}")
    logger.info(f"Base URL: {config['base_url']}")
    
    return True


def is_local_model(model_name: str) -> bool:
    """Check if a model name refers to a locally-served model."""
    return model_name in LOCAL_MODEL_CONFIGS


def get_actual_model_name(model_name: str) -> str:
    """
    Get the actual model name to use with the API.
    
    For local models, this returns just the model name without provider prefix.
    """
    if model_name in LOCAL_MODEL_CONFIGS:
        return LOCAL_MODEL_CONFIGS[model_name]["model"]
    return model_name


def validate_local_server(model_name: str) -> bool:
    """
    Check if the local server for a model is running.
    
    Args:
        model_name: Model identifier
        
    Returns:
        True if server is accessible, False otherwise
    """
    if model_name not in LOCAL_MODEL_CONFIGS:
        return True  # Not a local model, assume it's fine
    
    # Special case for OpenAI API models configured as "local"
    if model_name == "openai/davinci-002":
        # Check if proxy server is running
        import requests
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Davinci-002 proxy server is running for {model_name}")
                return True
            else:
                logger.error(f"Davinci-002 proxy server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException:
            logger.error("Davinci-002 proxy server not running. Please start it with: uv run python scripts/davinci_proxy_server.py")
            return False
    
    import requests
    
    config = LOCAL_MODEL_CONFIGS[model_name]
    base_url = config["base_url"].rstrip("/v1")  # Remove /v1 for health check
    
    try:
        # Try to hit the models endpoint
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            logger.info(f"Local server at {base_url} is running")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot connect to local server at {base_url}: {e}")
    
    return False 