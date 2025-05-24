.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

Codebase Overview
=================

Human-TTC-Eval is designed for evaluating AI models on cybersecurity tasks, drawing inspiration from METR's horizon-curve methodology. It emphasizes a simple, modular architecture to process diverse datasets and benchmark AI capabilities against human performance.

**Core Philosophy:**

*   **Simplicity over Premature Abstraction:** Prioritize straightforward implementations. Avoid unnecessary flexibility that could complicate the core goal of clear, reproducible evaluation.
*   **Configuration-Driven Tuning:** Keep the Command Line Interface (CLI) simple with minimal arguments. Most run-specific parameters (e.g., model names, iteration counts, API keys) are managed in a central `config.py` file.
*   **Standardized Dataset Pipeline:** Each dataset follows a consistent workflow:
    1.  **Retrieve:** Fetch raw data or metadata.
    2.  **Parse:** Convert raw data into a standardized format (typically a JSONL file like `all_runs.jsonl` or task-specific schemas). This standardized output is crucial for consistent downstream processing.
    3.  **Summarise:** Generate dataset-specific statistics and visualizations from the parsed data.
    4.  **Bench:** Run AI models against the parsed tasks using appropriate evaluation harnesses.
*   **Leverage Existing Work:** Integrate robust third-party tools where appropriate. Examples include:
    *   Using CyBench's native benchmarking harness for CyBench tasks.
    *   Employing METR's logistic regression and plotting code for horizon analysis via the `analysis` module.
*   **Modular and Extensible:** New datasets and models can be added by implementing defined interfaces and registering them with the core system, making them automatically available via the CLI.

Directory Structure
-------------------

.. code-block:: text

    human-ttc-eval/
    ├── src/                        # Source code
    │   └── human_ttc_eval/
    │       ├── __init__.py         # Main package initializer
    │       ├── cli.py              # Command Line Interface (Typer/Click)
    │       ├── config.py           # Centralized configuration and API keys
    │       ├── models.json         # Registry for model metadata (release dates)
    │       │
    │       ├── core/               # Core abstractions and utilities
    │       │   ├── __init__.py
    │       │   ├── base_parser.py      # Base class for dataset parsers
    │       │   ├── base_summariser.py  # Base class for dataset summarisers
    │       │   ├── base_retriever.py   # Base class for dataset retrievers
    │       │   ├── base_bench.py       # Base class for benchmark harnesses
    │       │   ├── inspect_bench.py    # Base for inspect_ai powered benchmarks
    │       │   ├── llm_utils.py        # LLM provider integration
    │       │   ├── registry.py         # Decorator-based registry for components
    │       │   └── utils.py            # Common utility functions
    │       │
    │       ├── datasets/           # Dataset-specific modules
    │       │   ├── __init__.py         # Imports submodules to register components
    │       │   ├── kypo/               # KYPO Cyber Range Dataset
    │       │   │   ├── __init__.py
    │       │   │   ├── parser.py
    │       │   │   └── summariser.py
    │       │   ├── cybench/            # CyBench Dataset
    │       │   │   ├── __init__.py
    │       │   │   ├── retrieve.py
    │       │   │   ├── parser.py
    │       │   │   ├── summariser.py
    │       │   │   └── bench.py
    │       │   └── nl2bash/            # NL2Bash Dataset
    │       │       ├── __init__.py
    │       │       ├── retrieve.py
    │       │       ├── parser.py
    │       │       ├── summariser.py
    │       │       └── bench.py
    │       │
    │       └── analysis/           # Horizon plotting and METR integration
    │           ├── __init__.py
    │           ├── transform.py      # Converts benchmark results to METR format
    │           └── plotter.py        # Generates horizon plots using METR's code
    │
    ├── data/                       # Dataset storage
    │   ├── raw/                    # Original downloaded data (e.g., from retrieve step)
    │   ├── processed/              # Cleaned, standardized tasks (e.g., from parse step)
    │   └── cybench_human_runs.jsonl # Example human baseline data
    │
    ├── third-party/                # External repositories (e.g., nl2bash, cybench, METR's eval-analysis-public)
    ├── results/                    # Evaluation outputs
    │   ├── benchmarks/             # Raw results from AI model benchmark runs
    │   ├── dataset-summaries/      # Outputs from the summarise step
    │   └── plots/                  # Generated horizon plots and other visuals
    │
    ├── docs/                       # Documentation
    ├── Makefile                    # Main operational entry point
    ├── .env.template               # Template for API keys
    └── pyproject.toml              # Project metadata and dependencies (uv)

Entry Points
------------

**Primary Interface: `Makefile`**

The `Makefile` serves as the main entry point for most operations, providing convenient targets for common workflows. It abstracts the underlying CLI calls.

.. code-block:: bash

    make help                   # List all available Make targets
    make datasets               # Process all datasets (retrieve, parse, summarise)
    make benchmark MODEL=...    # Run benchmarks for a specified model
    make plot                   # Generate horizon plots from benchmark results
    make docs                   # Build Sphinx documentation

**Underlying Interface: `human_ttc_eval.cli`**

The `Makefile` targets typically invoke the project's Command Line Interface, built with Typer (using Click).

.. code-block:: bash

    # Example CLI invocations (usually run via Make)
    uv run python -m human_ttc_eval.cli parse cybench --input-dir data/raw/cybench --output-file data/cybench_human_runs.jsonl
    uv run python -m human_ttc_eval.cli benchmark --dataset cybench --model openai/gpt-4o-2024-05-13 --output-dir results/benchmarks/cybench
    uv run python -m human_ttc_eval.cli plot --results-dir results/benchmarks --output-dir results/plots

Dataset Module Pattern
----------------------

Each supported dataset (e.g., `kypo`, `cybench`, `nl2bash`) typically follows a consistent module structure within `src/human_ttc_eval/datasets/`. This promotes modularity and ease of adding new datasets.

**1. Retriever (`retrieve.py`)**
   - **Purpose:** Downloads, fetches, or ensures the availability of the raw dataset files or metadata.
   - **Output:** Raw data stored in `data/raw/<dataset_name>/`.
   - **Example:** For `nl2bash`, it clones the GitHub repository. For `cybench`, it processes metadata from a local CyBench repo clone.

**2. Parser (`parser.py`)**  
   - **Purpose:** Processes the raw data obtained by the retriever and transforms it into a standardized format. This often involves creating a JSONL file (e.g., `all_tasks.jsonl` or `cybench_human_runs.jsonl`) that represents tasks or human baseline performance.
   - **Input:** Raw data from `data/raw/<dataset_name>/`.
   - **Output:** Standardized data in `data/processed/<dataset_name>/` or directly into files like `data/cybench_human_runs.jsonl`.
   - **Schema:** The output aims for a common schema that can be understood by downstream analysis and benchmarking tools.

**3. Summariser (`summariser.py`)**
   - **Purpose:** Reads the parsed, standardized data and generates descriptive statistics, summaries, and dataset-specific plots.
   - **Input:** Parsed data (e.g., JSONL files).
   - **Output:** CSV files and plots in `results/dataset-summaries/<dataset_name>/`.

**4. Bench (`bench.py`)**
   - **Purpose:** Provides the harness for running AI models against the tasks from the parsed dataset. This module is responsible for interacting with the AI model, presenting tasks, collecting responses, and scoring.
   - **Integration:**
     - For datasets like CyBench, it might wrap an external evaluation script (`third-party/cybench/run_benchmark.py`).
     - For others like NL2Bash, it might use an internal framework like `inspect_ai` (via `core.inspect_bench.py`).
   - **Output:** Raw benchmark results (often JSON files) in `results/benchmarks/<dataset_name>/`.

These modules are made discoverable to the CLI via registration decorators in `src/human_ttc_eval/core/registry.py`.

Core Utilities
--------------

The `src/human_ttc_eval/core/` directory contains shared utilities and base classes:

*   **`config.py`**: Centralizes user-configurable parameters (e.g., default model names, API keys loaded from `.env`, batch sizes, file paths). This keeps the CLI lean and allows for easy tuning of experiments.
*   **`llm_utils.py`**: Provides a unified client (`LLMClient`) for interacting with multiple LLM providers (Anthropic, OpenAI, Google). Handles API key management, retry logic, and offers batch processing utilities. Includes functions for real-time pricing estimation.
*   **`registry.py`**: Implements a decorator-based registry system. Parsers, summarisers, retrievers, and benchmark harnesses register themselves, making them discoverable by the CLI. This allows for easy extension of the system.
*   **Base Classes (`base_parser.py`, `base_summariser.py`, `base_retriever.py`, `base_bench.py`)**: Define abstract base classes for each component in the dataset module pattern. This enforces a consistent interface and promotes code reuse.
*   **`inspect_bench.py`**: A specialized base class for benchmarks implemented using the `inspect_ai` evaluation framework.
*   **`utils.py`**: Contains common utility functions, like `slugify`.

Analysis Module
---------------
The `src/human_ttc_eval/analysis/` directory is dedicated to replicating METR's horizon analysis:

*   **`transform.py`**: Transforms the raw benchmark results (from `results/benchmarks/`) and human baseline data into the specific JSONL format required by METR's analysis scripts (`all_runs.jsonl`, `release_dates.yaml`).
*   **`plotter.py`**: Acts as a wrapper around METR's `run_logistic_regressions` and `plot_horizon_graph` functions (imported from `third-party/eval-analysis-public`). It takes the transformed data and generates the final horizon plots.

Adding New Datasets
--------------------

1.  Create a new directory: `src/human_ttc_eval/datasets/newdataset/`.
2.  Implement the necessary modules (e.g., `retriever.py`, `parser.py`, `summariser.py`, `bench.py`), inheriting from the base classes in `src/human_ttc_eval/core/`.
3.  Use the appropriate registration decorator (e.g., `@register_parser("newdataset")`) in each module.
4.  Import your new dataset module in `src/human_ttc_eval/datasets/__init__.py` so its components are registered at startup.
5.  Add corresponding targets to the `Makefile` for the new dataset's processing steps (retrieve, parse, summarise, benchmark).
6.  Ensure the `parser.py` module outputs data in a schema that `analysis.transform.py` can understand or adapt `transform.py` accordingly.

Development Workflow
--------------------

.. code-block:: bash

    # 1. Initial Setup (first time)
    # Create a virtual environment (e.g., using uv)
    uv venv
    source .venv/bin/activate
    # Install dependencies
    uv pip install -r requirements.txt # Or uv sync if pyproject.toml is primary

    # 2. Create .env file from .env.template for API keys
    cp .env.template .env
    # Edit .env with your API keys

    # 3. Process a dataset (example: NL2Bash)
    make nl2bash-retrieve
    make nl2bash-parse
    make nl2bash-summarise

    # 4. Check outputs
    ls data/processed/nl2bash/
    ls results/dataset-summaries/nl2bash/

    # 5. Run a benchmark (example: NL2Bash with a specific model)
    make nl2bash-benchmark MODEL=openai/gpt-4o-2024-05-13

    # 6. Transform results and generate plots
    make plot SUCCESS_RATE=0.5

    # 7. View plots
    ls results/plots/

    # 8. Build documentation
    make docs
    # Open docs/build/html/index.html in your browser

Configuration
-------------

Key configuration options are managed in `src/human_ttc_eval/config.py`.
API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are loaded from an `.env` file in the project root.

**Environment Setup:**

Create an `.env` file in the project root:
.. code-block:: bash

    # .env (example)
    OPENAI_API_KEY="sk-..."
    ANTHROPIC_API_KEY="sk-ant-..."
    GOOGLE_API_KEY="..."

This modular design, combined with a clear CLI and Make-based workflow, aims to provide a robust and extensible framework for evaluating AI models on cybersecurity tasks. 