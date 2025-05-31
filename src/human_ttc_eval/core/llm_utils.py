"""General LLM utilities for calling different model providers."""

import time
import concurrent.futures
from typing import List, Any, Optional
from dataclasses import dataclass
import anthropic
from openai import OpenAI
from ..config import get_api_key, LLM_MAX_TOKENS, LLM_TEMPERATURE, DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM calls."""
    provider: str  # 'anthropic', 'openai'
    model: str
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass

class LLMClient:
    """Unified client for calling different LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._setup_client()
    
    def _setup_client(self):
        """Initialize the appropriate client based on provider."""
        if self.config.provider == 'anthropic':
            api_key = get_api_key('anthropic')
            if not api_key:
                raise LLMError("ANTHROPIC_API_KEY environment variable not set")
            self.client = anthropic.Anthropic(api_key=api_key)
            
        elif self.config.provider == 'openai':
            api_key = get_api_key('openai')
            if not api_key:
                raise LLMError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
            
        else:
            raise LLMError(f"Unsupported provider: {self.config.provider}")
    
    def call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a single call to the LLM with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                return self._make_call(prompt, system_prompt)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise LLMError(f"Failed after {self.config.max_retries} attempts: {str(e)}")
                
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise LLMError("Unexpected error in retry logic")
    
    def _make_call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make the actual API call based on provider."""
        if self.config.provider == 'anthropic':
            return self._call_anthropic(prompt, system_prompt)
        elif self.config.provider == 'openai':
            return self._call_openai(prompt, system_prompt)
        else:
            raise LLMError(f"Unsupported provider: {self.config.provider}")
    
    def _call_anthropic(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call Anthropic Claude API."""
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
    
    def _call_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
        )
        
        return response.choices[0].message.content

def create_default_llm_client() -> LLMClient:
    """Create a default LLM client using config defaults."""
    config = LLMConfig(
        provider=DEFAULT_LLM_PROVIDER,
        model=DEFAULT_LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        timeout=60,
        max_retries=3,
        retry_delay=1.0
    )
    return LLMClient(config)

def batch_process(
    items: List[Any], 
    processor_func: callable, 
    batch_size: int = 100,
    progress_callback: Optional[callable] = None
) -> List[Any]:
    """Process items in batches with optional progress reporting.
    
    Args:
        items: List of items to process
        processor_func: Function that takes a batch and returns results
        batch_size: Number of items per batch
        progress_callback: Optional function called with (current, total) after each batch
    
    Returns:
        List of results from all batches
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = processor_func(batch)
        results.extend(batch_results)
        
        if progress_callback:
            current_batch = (i // batch_size) + 1
            progress_callback(current_batch, total_batches)
    
    return results

def batch_process_parallel(
    items: List[Any], 
    processor_func: callable, 
    batch_size: int = 100,
    max_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> List[Any]:
    """Process items in batches with parallel execution.
    
    Args:
        items: List of items to process
        processor_func: Function that takes a batch and returns results
        batch_size: Number of items per batch
        max_workers: Maximum number of parallel workers
        progress_callback: Optional function called with (current, total) after each batch
    
    Returns:
        List of results from all batches
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    # Create batches
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append(batch)
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(processor_func, batch): i for i, batch in enumerate(batches)}
        
        completed_batches = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_results = future.result()
            results.extend(batch_results)
            
            completed_batches += 1
            if progress_callback:
                progress_callback(completed_batches, total_batches)
    
    return results

def get_real_time_pricing() -> dict:
    """Get current pricing from provider websites - uses latest 2025 rates."""
    # Using current rates from web search (May 2025)
    return {
        'anthropic': {
            "provider": "anthropic",
            "models": {
                "claude-3-5-sonnet-20241022": {"input_price_per_million": 3.0, "output_price_per_million": 15.0},
                "claude-3-5-haiku-20241022": {"input_price_per_million": 0.8, "output_price_per_million": 4.0},
                "claude-3-opus-20240229": {"input_price_per_million": 15.0, "output_price_per_million": 75.0},
                "claude-sonnet-4": {"input_price_per_million": 3.0, "output_price_per_million": 15.0},
                "claude-opus-4": {"input_price_per_million": 15.0, "output_price_per_million": 75.0},
                "claude-haiku-3.5": {"input_price_per_million": 0.8, "output_price_per_million": 4.0}
            }
        },
        'openai': {
            "provider": "openai", 
            "models": {
                "gpt-4o": {"input_price_per_million": 2.5, "output_price_per_million": 5.0},
                "gpt-4": {"input_price_per_million": 30.0, "output_price_per_million": 60.0},
                "gpt-3.5-turbo": {"input_price_per_million": 3.0, "output_price_per_million": 6.0},
                "gpt-4-turbo": {"input_price_per_million": 10.0, "output_price_per_million": 30.0},
                "o3": {"input_price_per_million": 1.0, "output_price_per_million": 4.0}
            }
        },
        'google': {
            "provider": "google",
            "models": {
                "gemini-1.5-flash": {"input_price_per_million": 0.075, "output_price_per_million": 0.30},
                "gemini-1.5-pro": {"input_price_per_million": 1.25, "output_price_per_million": 5.0},
                "gemini-2.0-flash": {"input_price_per_million": 0.10, "output_price_per_million": 0.40},
                "gemini-2.5-flash": {"input_price_per_million": 0.15, "output_price_per_million": 0.60},
                "gemini-2.5-pro": {"input_price_per_million": 2.50, "output_price_per_million": 15.0}
            }
        }
    }

def estimate_batch_cost_realtime(
    num_items: int, 
    batch_size: int, 
    provider: str = 'anthropic',
    model: str = None
) -> dict:
    """Estimate the cost of processing items in batches using real-time pricing.
    
    Args:
        num_items: Total number of items to process
        batch_size: Items per batch
        provider: LLM provider ('anthropic', 'openai', 'google')
        model: Specific model name (if None, uses provider default)
    
    Returns:
        Dictionary with cost estimates
    """
    num_batches = (num_items + batch_size - 1) // batch_size
    
    # Get real-time pricing
    pricing_data = get_real_time_pricing()
    
    if provider not in pricing_data:
        raise ValueError(f"Pricing not available for provider: {provider}")
    
    provider_data = pricing_data[provider]
    
    # Select model
    if model is None:
        # Use first model as default
        model = list(provider_data['models'].keys())[0]
    
    if model not in provider_data['models']:
        # Try to find a similar model or use first available
        available_models = list(provider_data['models'].keys())
        logger.warning(f"Model {model} not found. Available models: {available_models}")
        model = available_models[0]
        logger.warning(f"Using {model} instead.")
    
    model_pricing = provider_data['models'][model]
    
    # Rough token estimates (will vary based on actual content)
    input_tokens_per_batch = 2000  # System prompt + batch content
    output_tokens_per_batch = 500  # JSON response
    
    total_input_tokens = num_batches * input_tokens_per_batch
    total_output_tokens = num_batches * output_tokens_per_batch
    
    # Convert pricing from per million tokens to per token
    cost_per_input_token = model_pricing['input_price_per_million'] / 1_000_000
    cost_per_output_token = model_pricing['output_price_per_million'] / 1_000_000
    
    input_cost = total_input_tokens * cost_per_input_token
    output_cost = total_output_tokens * cost_per_output_token
    total_cost = input_cost + output_cost
    
    return {
        'num_batches': num_batches,
        'estimated_input_tokens': total_input_tokens,
        'estimated_output_tokens': total_output_tokens,
        'estimated_input_cost': input_cost,
        'estimated_output_cost': output_cost,
        'estimated_total_cost': total_cost,
        'provider': provider,
        'model': model,
        'input_price_per_million': model_pricing['input_price_per_million'],
        'output_price_per_million': model_pricing['output_price_per_million']
    }

if __name__ == "__main__":
    # Test the LLM client - these prints are for user interaction so keeping them
    try:
        client = create_default_llm_client()
        response = client.call(
            "What is 2+2?", 
            system_prompt="You are a helpful assistant that gives brief answers."
        )
        print(f"Test response: {response}")
        
        # Test real-time cost estimation
        cost_estimate = estimate_batch_cost_realtime(12607, 100, 'anthropic', 'claude-3-5-sonnet-20241022')
        print(f"\nCost estimate for NL2Bash dataset:")
        print(f"  Provider: {cost_estimate['provider']}")
        print(f"  Model: {cost_estimate['model']}")
        print(f"  Batches: {cost_estimate['num_batches']}")
        print(f"  Estimated cost: ${cost_estimate['estimated_total_cost']:.2f}")
        
    except LLMError as e:
        print(f"LLM Error: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in your environment")
    except Exception as e:
        print(f"Unexpected error: {e}") 
