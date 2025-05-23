"""Simple configuration for the Human TTC Evaluation project."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env file from project root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not available, rely on system environment
    pass

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# LLM Configuration
DEFAULT_LLM_PROVIDER = 'anthropic'
DEFAULT_LLM_MODEL = 'claude-3-5-sonnet-20241022'
LLM_BATCH_SIZE = 100
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.1

# NL2Bash Configuration
ENABLE_LLM_TIMING = True  # Set to False to skip LLM time estimation
NL2BASH_LLM_PROVIDER = 'anthropic'
NL2BASH_LLM_MODEL = 'claude-3-5-sonnet-20241022'
NL2BASH_BATCH_SIZE = 100

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
THIRD_PARTY_DIR = PROJECT_ROOT / "third-party"

# Dataset Configuration
CYBENCH_REPO_PATH = THIRD_PARTY_DIR / "cybench"
NL2BASH_REPO_PATH = THIRD_PARTY_DIR / "nl2bash"

def has_api_key(provider: str) -> bool:
    """Check if API key is available for a provider."""
    keys = {
        'openai': OPENAI_API_KEY,
        'anthropic': ANTHROPIC_API_KEY, 
        'google': GOOGLE_API_KEY
    }
    key = keys.get(provider)
    return key is not None and key.strip() != ""

def get_api_key(provider: str) -> str:
    """Get API key for a provider or raise error if not available."""
    keys = {
        'openai': OPENAI_API_KEY,
        'anthropic': ANTHROPIC_API_KEY,
        'google': GOOGLE_API_KEY
    }
    key = keys.get(provider)
    if not key or key.strip() == "":
        raise ValueError(f"{provider.upper()}_API_KEY environment variable not set")
    return key

if __name__ == "__main__":
    print("Configuration:")
    print(f"  OpenAI API Key: {'✓' if has_api_key('openai') else '✗'}")
    print(f"  Anthropic API Key: {'✓' if has_api_key('anthropic') else '✗'}")
    print(f"  Google API Key: {'✓' if has_api_key('google') else '✗'}")
    print(f"  Default LLM: {DEFAULT_LLM_PROVIDER}/{DEFAULT_LLM_MODEL}")
    print(f"  NL2Bash LLM Timing: {ENABLE_LLM_TIMING}") 