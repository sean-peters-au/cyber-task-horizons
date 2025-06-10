"""
Registers the CyBashBench dataset components with the system.
This ensures that the retriever, preparer, describer, and bench
harness are discoverable by the CLI.
"""

from . import cybashbench_retrieve
from . import cybashbench_prepare
from . import cybashbench_describe
from . import cybashbench_bench 