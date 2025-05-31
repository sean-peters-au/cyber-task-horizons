"""NL2Bash dataset integration."""

# Import modules to ensure they register themselves with the registry
try:
    from . import parser as nl2bash_parser_module  # noqa
    from . import summariser as nl2bash_summariser_module  # noqa
    from . import bench as nl2bash_bench_module # noqa
    from . import retrieve as nl2bash_retrieve_module # noqa
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Could not import all NL2Bash modules: {e}") 