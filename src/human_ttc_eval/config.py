"""
Simple configuration for the Human TTC Evaluation project.

Configuration Principles:
- User-configurable parameters that might be tuned across different runs go in this config.py file
- Implementation constants that never change should be defined as constants at the top of 
  the relevant module (e.g., DEFAULT_UNGUIDED_MODE in bench.py files)
- Dataset-specific CLI arguments should be avoided - the CLI should be completely generic
- All configuration should be centralized here to maintain clean separation of concerns
"""

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
LLM_MAX_CONNECTIONS = 2 # Default concurrency for inspect_ai

# NL2Bash Configuration
ENABLE_LLM_TIMING = False  # Set to False to skip LLM time estimation
NL2BASH_LLM_PROVIDER = 'anthropic'
NL2BASH_LLM_MODEL = 'claude-3-5-sonnet-20241022'
NL2BASH_BATCH_SIZE = 100
NL2BASH_SAMPLE_SIZE = 200
# CyBench Configuration (user-tunable parameters)
CYBENCH_MAX_ITERATIONS = 50
CYBENCH_IMPUTE_ZERO = [
    "openai/gpt2-xl",
    "openai/davinci-002",
]

# InterCode-CTF Configuration
INTERCODE_MAX_ITERATIONS = 30
INTERCODE_IMPUTE_ZERO = [
    "openai/gpt2-xl", 
    "openai/davinci-002",
]

# NYUCTF Configuration
NYUCTF_SAMPLE_SIZE = 50
NYUCTF_RANDOM_SEED = 42
NYUCTF_MAX_ITERATIONS = 50
NYUCTF_STRATIFY_BY_POINTS = True  # Sample across difficulty levels (easy/medium/hard)
NYUCTF_IMPUTE_ZERO = [
    "openai/gpt2-xl",
    "openai/davinci-002",
]

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
THIRD_PARTY_DIR = PROJECT_ROOT / "third-party"

# Dataset Configuration
CYBENCH_REPO_PATH = THIRD_PARTY_DIR / "cybench"
NL2BASH_REPO_PATH = THIRD_PARTY_DIR / "nl2bash"
INTERCODE_REPO_PATH = THIRD_PARTY_DIR / "intercode"

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