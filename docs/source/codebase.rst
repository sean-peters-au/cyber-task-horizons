.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

Codebase Overview
=================

Simple, modular architecture for cybersecurity dataset processing and evaluation.

Directory Structure
-------------------

.. code-block:: text

    human-ttc-eval/
    ├── src/                    # Source code
    │   └── human_ttc_eval/
    │       ├── core/           # Shared utilities (LLM, config)
    │       └── datasets/       # Dataset-specific modules
    │           ├── nl2bash/
    │           └── cybench/
    ├── data/                   # Dataset storage
    │   ├── raw/                # Original downloaded data
    │   ├── processed/          # Cleaned, standardized tasks
    │   └── external/           # Third-party datasets
    ├── third-party/            # External repositories (nl2bash, cybench)
    ├── results/                # Evaluation outputs
    ├── docs/                   # Documentation
    └── Makefile               # Main entry point

Entry Points
------------

**Primary Interface: Make**

.. code-block:: bash

    make help                   # List all available targets
    make nl2bash-parse          # Download and process NL2Bash
    make nl2bash-summarise      # Generate dataset statistics

Dataset Module Pattern
----------------------

Each dataset follows a consistent 4-module pattern:

**1. Retriever** (`retrieve.py`)
Downloads and caches source data.

.. code-block:: python

    def get_data_files() -> tuple[Path, Path]:
        """Auto-download if missing, return file paths."""

**2. Parser** (`parser.py`)  
Processes raw data into standardized tasks.

.. code-block:: python

    @dataclass
    class DatasetTask:
        id: str
        description: str
        estimated_time_seconds: float
        # ... dataset-specific fields

    def parse_dataset() -> Iterator[DatasetTask]:
        """Yield standardized task objects."""

**3. Summariser** (`summariser.py`)
Generates statistics and analysis.

.. code-block:: python

    def generate_summary(tasks: List[DatasetTask]) -> dict:
        """Compute dataset statistics."""

**4. Bench** (`bench.py`) *(Future)*
Agent evaluation harness.

.. code-block:: python

    def evaluate_model(model: str, tasks: List[DatasetTask]) -> Results:
        """Run model against task set."""

Core Utilities
--------------

**Configuration** (`src/human_ttc_eval/config.py`)
Centralized settings and API key management.

**LLM Utils** (`src/human_ttc_eval/core/llm_utils.py`)
- Multi-provider LLM client (Anthropic, OpenAI)
- Parallel batch processing with cost estimation
- Real-time pricing from provider APIs

Adding New Datasets
--------------------

1. Create directory: `src/human_ttc_eval/datasets/newdataset/`
2. Implement the 4 modules: `retrieve.py`, `parser.py`, `summariser.py`, `bench.py`
3. Add makefile targets:

.. code-block:: make

    newdataset-parse:
        uv run python -m src.human_ttc_eval.datasets.newdataset.parser

4. Follow common task schema in output JSONL:

.. code-block:: json

    {
      "id": "unique_id",
      "description": "task description",
      "estimated_time_seconds": 180.0,
      "timing_source": "llm|heuristic|empirical"
    }

Development Workflow
--------------------

.. code-block:: bash

    # Install dependencies
    uv sync

    # Process datasets
    make nl2bash-parse

    # Check outputs
    ls data/processed/nl2bash/

    # Generate documentation
    cd docs && make html

Configuration
-------------

Settings in `src/human_ttc_eval/config.py`:

- API keys loaded from `.env` file
- LLM provider and model selection
- Batch sizes and processing parameters
- File paths and directory structure

**Environment Setup:**

.. code-block:: bash

    # .env file
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=...

That's it. Simple modular design focused on getting things done. 