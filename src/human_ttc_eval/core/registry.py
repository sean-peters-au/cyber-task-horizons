from typing import Dict, Type, Callable, List
from .base_parser import BaseParser
from .base_summariser import BaseSummariser
from .base_retriever import BaseRetriever
from .base_bench import BaseBench
import logging

logger = logging.getLogger(__name__)

# Global registries
_parsers: Dict[str, Type[BaseParser]] = {}
_summarisers: Dict[str, Type[BaseSummariser]] = {}
_retrievers: Dict[str, Type[BaseRetriever]] = {}
_benches: Dict[str, Type[BaseBench]] = {}

def register_parser(name: str) -> Callable[[Type[BaseParser]], Type[BaseParser]]:
    """Decorator to register a parser class."""
    def decorator(cls: Type[BaseParser]) -> Type[BaseParser]:
        if name in _parsers:
            logger.warning(f"Parser '{name}' already registered. Overwriting.")
        _parsers[name] = cls
        return cls
    return decorator

def get_parser(name: str) -> Type[BaseParser]:
    """Get a registered parser class by name."""
    if name not in _parsers:
        raise ValueError(f"Parser '{name}' not found. Available parsers: {list(_parsers.keys())}")
    return _parsers[name]

def list_parsers() -> List[str]:
    """List all registered parser names."""
    return list(_parsers.keys())

def register_summariser(name: str) -> Callable[[Type[BaseSummariser]], Type[BaseSummariser]]:
    """Decorator to register a summariser class."""
    def decorator(cls: Type[BaseSummariser]) -> Type[BaseSummariser]:
        if name in _summarisers:
            logger.warning(f"Summariser '{name}' already registered. Overwriting.")
        _summarisers[name] = cls
        return cls
    return decorator

def get_summariser(name: str) -> Type[BaseSummariser]:
    """Get a registered summariser class by name."""
    if name not in _summarisers:
        raise ValueError(f"Summariser '{name}' not found. Available summarisers: {list(_summarisers.keys())}")
    return _summarisers[name]

def list_summarisers() -> List[str]:
    """List all registered summariser names."""
    return list(_summarisers.keys())

def register_retriever(name: str) -> Callable[[Type[BaseRetriever]], Type[BaseRetriever]]:
    """Decorator to register a retriever class."""
    def decorator(cls: Type[BaseRetriever]) -> Type[BaseRetriever]:
        if name in _retrievers:
            logger.warning(f"Retriever '{name}' already registered. Overwriting.")
        _retrievers[name] = cls
        return cls
    return decorator

def get_retriever(name: str) -> Type[BaseRetriever]:
    """Get a registered retriever class by name."""
    if name not in _retrievers:
        raise ValueError(f"Retriever '{name}' not found. Available retrievers: {list(_retrievers.keys())}")
    return _retrievers[name]

def list_retrievers() -> List[str]:
    """List all registered retriever names."""
    return list(_retrievers.keys())

def register_bench(name: str) -> Callable[[Type[BaseBench]], Type[BaseBench]]:
    """Decorator to register a bench class."""
    def decorator(cls: Type[BaseBench]) -> Type[BaseBench]:
        if name in _benches:
            logger.warning(f"Bench '{name}' already registered. Overwriting.")
        _benches[name] = cls
        return cls
    return decorator

def get_bench(name: str) -> Type[BaseBench]:
    """Get a registered bench class by name."""
    if name not in _benches:
        raise ValueError(f"Bench '{name}' not found. Available benches: {list(_benches.keys())}")
    return _benches[name]

def list_benches() -> List[str]:
    """List all registered bench names."""
    return list(_benches.keys()) 