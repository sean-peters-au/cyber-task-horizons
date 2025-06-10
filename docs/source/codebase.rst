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
    2.  **Prepare:** Convert raw data into a standardized format, creating `Run` objects for human baselines and `Task` objects for benchmarking.
    3.  **Describe:** Generate dataset-specific statistics and visualizations from the prepared data.
    4.  **Bench:** Run AI models against the prepared tasks using appropriate evaluation harnesses.
*   **Leverage Existing Work:** Integrate robust third-party tools where appropriate. Examples include:
    *   Using `inspect_ai` for sandboxed evaluation of interactive tasks (CyBench, InterCode-CTF).
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
    │       │   ├── retrieve.py     # Base class for dataset retrievers
    │       │   ├── prepare.py      # Base class for dataset preparers
    │       │   ├── describe.py     # Base class for dataset describers
    │       │   ├── bench.py        # Base class for benchmark harnesses
    │       │   ├── run.py          # Dataclass for a single evaluation run (METR schema)
    │       │   ├── task.py         # Dataclass for a single task definition
    │       │   ├── llm_utils.py      # LLM provider integration
    │       │   └── registry.py       # Decorator-based registry for components
    │       │
    │       ├── datasets/           # Dataset-specific modules
    │       │   ├── __init__.py       # Imports submodules to register components
    │       │   ├── cybench/          # CyBench Dataset
    │       │   │   ├── __init__.py
    │       │   │   ├── cybench_retrieve.py
    │       │   │   ├── cybench_prepare.py
    │       │   │   ├── cybench_describe.py
    │       │   │   └── cybench_bench.py
    │       │   ├── nl2bash/          # NL2Bash Dataset
    │       │   │   ├── __init__.py
    │       │   │   ├── nl2bash_retrieve.py
    │       │   │   ├── nl2bash_prepare.py
    │       │   │   ├── nl2bash_describe.py
    │       │   │   └── nl2bash_bench.py
    │       │   └── intercode_ctf/    # InterCode-CTF Dataset
    │       │       ├── __init__.py
    │       │       ├── intercode_ctf_retrieve.py
    │       │       ├── intercode_ctf_prepare.py
    │       │       ├── intercode_ctf_describe.py
    │       │       └── intercode_ctf_bench.py
    │       │
    │       └── analysis/           # Horizon plotting and METR integration
    │           ├── __init__.py
    │           ├── transform.py    # Converts benchmark results to METR format
    │           └── plotter.py      # Generates horizon plots using METR's code
    │
    ├── data/                       # Dataset storage
    │   ├── raw/                    # Original downloaded data (e.g., from retrieve step)
    │   └── processed/              # Cleaned, standardized tasks (e.g., from prepare step)
    │
    ├── third-party/                # External repositories (e.g., nl2bash, cybench, METR's eval-analysis-public)
    ├── results/                    # Evaluation outputs
    │   ├── benchmarks/             # Raw results from AI model benchmark runs
    │   ├── dataset-summaries/      # Outputs from the describe step
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
    make describe-all           # Process all datasets (retrieve, prepare, describe)
    make repro TIER=1           # Run all benchmarks for a specified tier of models
    make plot                   # Generate horizon plots from benchmark results
    make docs                   # Build Sphinx documentation

**Underlying Interface: `human_ttc_eval.cli`**

The `Makefile` targets typically invoke the project's Command Line Interface, built with Click.

.. code-block:: bash

    # Example CLI invocations (usually run via Make)
    uv run python -m human_ttc_eval.cli retrieve run cybench
    uv run python -m human_ttc_eval.cli prepare cybench
    uv run python -m human_ttc_eval.cli describe cybench
    uv run python -m human_ttc_eval.cli benchmark cybench --model openai/gpt-4o-2024-05-13
    uv run python -m human_ttc_eval.cli plot

Dataset Module Pattern
----------------------

Each supported dataset (e.g., `cybench`, `nl2bash`) follows a consistent module structure within `src/human_ttc_eval/datasets/`. This promotes modularity and ease of adding new datasets.

**1. Retriever (`<dataset>_retrieve.py`)**
   - **Purpose:** Downloads, fetches, or ensures the availability of the raw dataset files or metadata.
   - **Output:** Raw data stored in `data/raw/<dataset_name>/`.
   - **Example:** For `nl2bash`, it clones the GitHub repository. For `cybench`, it processes ``challenge.yaml`` files from the `inspect_evals` repository.

**2. Preparer (`<dataset>_prepare.py`)**  
   - **Purpose:** Processes the raw data obtained by the retriever and transforms it into a standardized format. It creates a list of `Run` objects representing the human baseline and a list of `Task` objects containing all metadata needed for benchmarking.
   - **Input:** Raw data from `data/raw/<dataset_name>/`.
   - **Output:** Standardized data files in `data/processed/<dataset_name>/` (e.g., `cybench_human_runs.jsonl`, `cybench_tasks.jsonl`).
   - **Schema:** The output strictly adheres to the `Run` and `Task` dataclasses defined in `src/human_ttc_eval/core/`.

**3. Describer (`<dataset>_describe.py`)**
   - **Purpose:** Reads the prepared, standardized data and generates descriptive statistics, summaries, and dataset-specific plots.
   - **Input:** Prepared data from `data/processed/<dataset_name>/`.
   - **Output:** CSV files and plots in `results/dataset-summaries/<dataset_name>/`.

**4. Bench (`<dataset>_bench.py`)**
   - **Purpose:** Provides the harness for running AI models against the tasks from the prepared dataset. This module is responsible for interacting with the AI model, presenting tasks, collecting responses, and scoring.
   - **Integration:** For all interactive datasets (`cybench`, `intercode-ctf`, `nl2bash`), it uses `inspect_ai` for sandboxed evaluation.
   - **Output:** Raw benchmark results (JSON files) in `results/benchmarks/<dataset_name>/`.

These modules are made discoverable to the CLI via registration decorators in `src/human_ttc_eval/core/registry.py`.

Core Utilities
--------------

The `src/human_ttc_eval/core/` directory contains shared utilities and base classes:

*   **`config.py`**: Centralizes user-configurable parameters (e.g., default model names, API keys loaded from `.env`, batch sizes, file paths). This keeps the CLI lean and allows for easy tuning of experiments.
*   **`llm_utils.py`**: Provides a unified client (`LLMClient`) for interacting with multiple LLM providers (Anthropic, OpenAI, Google). Handles API key management, retry logic, and offers batch processing utilities. Includes functions for real-time pricing estimation.
*   **`registry.py`**: Implements a decorator-based registry system. Retrievers, preparers, describers, and benchmark harnesses register themselves, making them discoverable by the CLI. This allows for easy extension of the system.
*   **Base Classes (`retrieve.py`, `prepare.py`, `describe.py`, `bench.py`)**: Define abstract base classes for each component in the dataset module pattern. This enforces a consistent interface and promotes code reuse.
*   **Data Models (`run.py`, `task.py`)**: Define the canonical `Run` and `Task` dataclasses that are used throughout the pipeline, ensuring data consistency.

Analysis Module
---------------
The `src/human_ttc_eval/analysis/` directory is dedicated to replicating METR's horizon analysis:

*   **`transform.py`**: Transforms the raw benchmark results (from `results/benchmarks/`) and human baseline data into the specific JSONL format required by METR's analysis scripts (`all_runs.jsonl`, `release_dates.yaml`).
*   **`plotter.py`**: Acts as a wrapper around METR's `run_logistic_regressions` and `plot_horizon_graph` functions (imported from `third-party/eval-analysis-public`). It takes the transformed data and generates the final horizon plots.

Adding New Datasets
--------------------

1.  Create a new directory: `src/human_ttc_eval/datasets/newdataset/`.
2.  Implement the necessary modules (e.g., `newdataset_retrieve.py`, `newdataset_prepare.py`), inheriting from the base classes in `src/human_ttc_eval/core/`.
3.  Use the appropriate registration decorator (e.g., `@register_retriever("newdataset")`) in each module.
4.  Import your new dataset module in `src/human_ttc_eval/datasets/__init__.py` so its components are registered at startup.
5.  Update the `DATASETS` variable in the `Makefile` to include `newdataset`. The generic targets will then apply to it.
6.  Ensure the `prepare.py` module outputs data consistent with the `Run` and `Task` schemas.

Development Workflow
--------------------

.. code-block:: bash

    # 1. Initial Setup (first time)
    # Create a virtual environment (e.g., using uv)
    uv venv
    source .venv/bin/activate
    # Install dependencies
    uv sync

    # 2. Create .env file from .env.template for API keys
    cp .env.template .env
    # Edit .env with your API keys

    # 3. Process a dataset (example: NL2Bash)
    make retrieve DATASET=nl2bash
    make prepare DATASET=nl2bash
    make describe DATASET=nl2bash

    # 4. Check outputs
    ls data/processed/nl2bash/
    ls results/dataset-summaries/nl2bash/

    # 5. Run a benchmark (example: NL2Bash with a specific model)
    make bench DATASET=nl2bash MODEL=openai/gpt-4o

    # 6. Transform results and generate plots
    make plot

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