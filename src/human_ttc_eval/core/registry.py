from typing import Dict, Type, Callable, List, Union
from .prepare import Prepare
from .describe import Describe
from .retrieve import Retrieve
from .bench import Bench
import logging

logger = logging.getLogger(__name__)

# Global registries
_preparers: Dict[str, Type[Prepare]] = {}
_describers: Dict[str, Type[Describe]] = {}
_retrievers: Dict[str, Type[Retrieve]] = {}
_benches: Dict[str, Type[Bench]] = {}

def register_preparer(name: str) -> Callable[[Type[Prepare]], Type[Prepare]]:
    """Decorator to register a preparer class."""
    def decorator(cls: Type[Prepare]) -> Type[Prepare]:
        if name in _preparers:
            logger.warning(f"Preparer '{name}' already registered. Overwriting.")
        _preparers[name] = cls
        return cls
    return decorator

def get_preparer(name: str) -> Type[Prepare]:
    """Get a registered preparer class by name."""
    if name not in _preparers:
        raise ValueError(f"Preparer '{name}' not found. Available preparers: {list(_preparers.keys())}")
    return _preparers[name]

def list_preparers() -> List[str]:
    """List all registered preparer names."""
    return list(_preparers.keys())

# Keep old names for CLI compatibility during transition
def register_parser(name: str) -> Callable[[Type[Prepare]], Type[Prepare]]:
    """Decorator to register a parser class. Alias for register_preparer."""
    return register_preparer(name)

def get_parser(name: str) -> Type[Prepare]:
    """Get a registered parser class by name. Alias for get_preparer."""
    return get_preparer(name)

def list_parsers() -> List[str]:
    """List all registered parser names. Alias for list_preparers."""
    return list_preparers()

def register_describer(name: str) -> Callable[[Type[Describe]], Type[Describe]]:
    """Decorator to register a describer class."""
    def decorator(cls: Type[Describe]) -> Type[Describe]:
        if name in _describers:
            logger.warning(f"Describer '{name}' already registered. Overwriting.")
        _describers[name] = cls
        return cls
    return decorator

def get_describer(name: str) -> Type[Describe]:
    """Get a registered describer class by name."""
    if name not in _describers:
        raise ValueError(f"Describer '{name}' not found. Available describers: {list(_describers.keys())}")
    return _describers[name]

def list_describers() -> List[str]:
    """List all registered describer names."""
    return list(_describers.keys())

# Keep old names for CLI compatibility during transition
def register_summariser(name: str) -> Callable[[Type[Describe]], Type[Describe]]:
    """Decorator to register a summariser class. Alias for register_describer."""
    return register_describer(name)

def get_summariser(name: str) -> Type[Describe]:
    """Get a registered summariser class by name. Alias for get_describer."""
    return get_describer(name)

def list_summarisers() -> List[str]:
    """List all registered summariser names. Alias for list_describers."""
    return list_describers()

def register_retriever(name: str) -> Callable[[Type[Retrieve]], Type[Retrieve]]:
    """Decorator to register a retriever class."""
    def decorator(cls: Type[Retrieve]) -> Type[Retrieve]:
        if name in _retrievers:
            logger.warning(f"Retriever '{name}' already registered. Overwriting.")
        _retrievers[name] = cls
        return cls
    return decorator

def get_retriever(name: str) -> Type[Retrieve]:
    """Get a registered retriever class by name."""
    if name not in _retrievers:
        raise ValueError(f"Retriever '{name}' not found. Available retrievers: {list(_retrievers.keys())}")
    return _retrievers[name]

def list_retrievers() -> List[str]:
    """List all registered retriever names."""
    return list(_retrievers.keys())

def register_bench(name: str) -> Callable[[Type[Bench]], Type[Bench]]:
    """Decorator to register a bench class."""
    def decorator(cls: Type[Bench]) -> Type[Bench]:
        if name in _benches:
            logger.warning(f"Bench '{name}' already registered. Overwriting.")
        _benches[name] = cls
        return cls
    return decorator

def get_bench(name: str) -> Type[Bench]:
    """Get a registered bench class by name."""
    if name not in _benches:
        raise ValueError(f"Bench '{name}' not found. Available benches: {list(_benches.keys())}")
    return _benches[name]

def list_benches() -> List[str]:
    """List all registered bench names."""
    return list(_benches.keys()) 