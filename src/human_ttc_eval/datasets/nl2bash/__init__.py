"""NL2Bash dataset integration."""

# Import modules to ensure they register themselves with the registry
try:
    from . import nl2bash_retrieve # noqa
    from . import nl2bash_prepare # noqa
    from . import nl2bash_describe # noqa
    from . import nl2bash_bench # noqa
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Could not import all NL2Bash modules: {e}") 