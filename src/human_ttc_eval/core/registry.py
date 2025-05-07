from typing import Dict, Type, Callable, List
from .base_parser import BaseParser
from .base_summariser import BaseSummariser

# Global registries
_PARSERS: Dict[str, Type[BaseParser]] = {}
_SUMMARISERS: Dict[str, Type[BaseSummariser]] = {}

def register_parser(name: str) -> Callable[[Type[BaseParser]], Type[BaseParser]]:
    """Decorator to register a new parser class."""
    def decorator(cls: Type[BaseParser]) -> Type[BaseParser]:
        if name in _PARSERS:
            print(f"Warning: Parser '{name}' already registered. Overwriting.") # Replace with logging
        _PARSERS[name] = cls
        return cls
    return decorator

def get_parser(name: str) -> Type[BaseParser]:
    """Retrieve a parser class from the registry."""
    if name not in _PARSERS:
        raise ValueError(f"No parser registered for '{name}'. Available: {list(_PARSERS.keys())}")
    return _PARSERS[name]

def list_parsers() -> List[str]:
    """Return a list of names of all registered parsers."""
    return list(_PARSERS.keys())

def register_summariser(name: str) -> Callable[[Type[BaseSummariser]], Type[BaseSummariser]]:
    """Decorator to register a new summariser class."""
    def decorator(cls: Type[BaseSummariser]) -> Type[BaseSummariser]:
        if name in _SUMMARISERS:
            print(f"Warning: Summariser '{name}' already registered. Overwriting.") # Replace with logging
        _SUMMARISERS[name] = cls
        return cls
    return decorator

def get_summariser(name: str) -> Type[BaseSummariser]:
    """Retrieve a summariser class from the registry."""
    if name not in _SUMMARISERS:
        raise ValueError(f"No summariser registered for '{name}'. Available: {list(_SUMMARISERS.keys())}")
    return _SUMMARISERS[name]

def list_summarisers() -> List[str]:
    """Return a list of names of all registered summarisers."""
    return list(_SUMMARISERS.keys()) 