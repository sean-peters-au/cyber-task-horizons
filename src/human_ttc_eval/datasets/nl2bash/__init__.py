"""NL2Bash dataset integration."""

# Import modules to ensure they register themselves with the registry
try:
    from . import parser as nl2bash_parser_module  # noqa
    from . import metr_parser as nl2bash_metr_parser_module  # noqa
    from . import metr_summariser as nl2bash_metr_summariser_module  # noqa
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Could not import all NL2Bash modules: {e}") 