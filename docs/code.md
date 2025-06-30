# Technical Architecture

> **Note**: This document was primarily created for prompting code assistance tools during development, but should be useful for humans wanting a quick rundown of the codebase architecture and implementation patterns. The content is AI-generated in its entirety.

## Design Philosophy

### Core Principles

**Simplicity Over Premature Abstraction**  
The framework prioritizes clear, straightforward implementations over flexibility that might complicate the core goal of reproducible evaluation. Every abstraction must justify its complexity with clear benefits.

**METR Schema Compatibility**  
Strict adherence to METR's `all_runs.jsonl` format ensures our results can be directly used with METR's analysis tools and compared against their findings. This compatibility is non-negotiable.

**Configuration-Driven Operation**  
The CLI remains minimal while most parameters (model names, API keys, iteration counts) are managed through central configuration. This separates operational concerns from interface complexity.

**Modular Extensibility**  
New datasets and models can be added through well-defined interfaces without modifying core framework code. The registry pattern makes components discoverable automatically.

## System Architecture Overview

### Four-Stage Pipeline

Every dataset follows the same processing pipeline, ensuring consistent outputs suitable for METR analysis:

```
Raw Data → [Retrieve] → [Prepare] → [Describe] → [Bench] → METR Analysis
```

**Stage Isolation**: Each stage can be run independently, enabling iterative development and debugging.  
**Standard Outputs**: All stages produce predictable file structures for downstream consumption.  
**Error Handling**: Failures at any stage don't compromise earlier work.

### Directory Structure Logic

```
cyber-task-horizons/
├── data/
│   ├── raw/                    # Retrieve outputs: unmodified source data
│   └── processed/              # Prepare outputs: METR-format runs and tasks
├── results/
│   ├── benchmarks/             # Bench outputs: AI evaluation results
│   ├── dataset-summaries/      # Describe outputs: analysis and plots
│   └── plots/                  # METR horizon curves and analysis
├── third-party/               # External tools (METR analysis, datasets)
└── src/human_ttc_eval/
    ├── core/                  # Base classes and utilities
    ├── datasets/              # Dataset-specific implementations
    └── analysis/              # METR integration and plotting
```

**Separation of Concerns**: Raw data preservation, processed data standardization, evaluation results, and analysis outputs are cleanly separated.

## Core Abstractions

### Base Classes (src/human_ttc_eval/core/)

**Retrieve** (`retrieve.py`):
- Fetches raw data from external sources
- Stores unmodified data in `data/raw/<dataset>/`
- No transformation or interpretation
- Handles source authentication, rate limiting, error recovery

**Prepare** (`prepare.py`):  
- Transforms raw data into standardized METR format
- Produces `Run` objects for human baselines
- Produces `Task` objects for AI evaluation
- Calculates statistical weights for analysis

**Describe** (`describe.py`):
- Generates dataset-specific statistics and visualizations
- Works with processed `Run` objects
- Creates standard plots plus custom analyses
- Outputs CSV summaries and PNG visualizations

**Bench** (`bench.py`):
- Evaluates AI models on prepared tasks
- Produces `Run` objects compatible with human baselines  
- Integrates with sandboxed evaluation environments
- Handles model-specific formatting and scoring

### Data Models (src/human_ttc_eval/core/)

**Run** (`run.py`): METR-compatible evaluation record
```python
@dataclass
class Run:
    task_id: str                    # Unique task identifier
    task_family: str               # Category for analysis grouping
    run_id: str                    # Unique run identifier  
    alias: str                     # Display name
    model: str                     # Model identifier or "human"
    score_binarized: int           # 0/1 success indicator
    score_cont: float              # Continuous score (0.0-1.0)
    human_minutes: float           # Expert completion time
    human_source: str              # Baseline methodology
    task_source: str               # Dataset origin
    # ... timing, cost, error fields
```

**Task** (`task.py`): Task definition with metadata
```python
@dataclass  
class Task:
    task_id: str                    # Matches Run.task_id
    task_family: str               # Matches Run.task_family
    human_minutes: float           # Expert baseline time
    equal_task_weight: float       # Statistical weighting
    invsqrt_task_weight: float     # Alternative weighting
    dataset_task_metadata: dict    # Dataset-specific fields
```

## Registry Pattern Implementation

### Automatic Discovery
Components register themselves using decorators, making them discoverable by the CLI without manual registration:

```python
# In dataset implementation
@register_retriever("cybench")
class CyBenchRetrieve(Retrieve):
    # Implementation

# In CLI
retriever = get_retriever("cybench")  # Automatic lookup
```

### Registry Benefits
- **Zero Configuration**: New datasets work immediately after import
- **Type Safety**: Registry enforces base class compliance
- **Introspection**: CLI can list all available components
- **Modularity**: Components can be developed independently

### Implementation Pattern
```python
# Registry storage
_retrievers: Dict[str, Type[Retrieve]] = {}

# Registration decorator
def register_retriever(name: str):
    def decorator(cls: Type[Retrieve]):
        _retrievers[name] = cls
        return cls
    return decorator

# Lookup function
def get_retriever(name: str) -> Type[Retrieve]:
    if name not in _retrievers:
        raise ValueError(f"No retriever registered for '{name}'")
    return _retrievers[name]
```

## Evaluation Environment Design

### inspect_ai Integration

**Sandboxed Execution**: All model evaluations run in isolated environments  
**Tool Access**: Models can execute bash commands, interact with web services  
**Resource Limits**: CPU, memory, network, and time constraints  
**Security Isolation**: Prevents information leakage between evaluations

## METR Analysis Integration

### Data Transformation (`analysis/transform.py`)

Converts framework outputs to METR's expected format:
- **Run Aggregation**: Combines human baselines and AI results
- **Schema Validation**: Ensures strict METR compatibility
- **Metadata Preservation**: Maintains provenance and quality indicators

### Plotting Integration (`analysis/plotter.py`)

Direct integration with METR's analysis tools:
- **Logistic Regression**: Uses METR's curve-fitting methodology
- **Horizon Plots**: Generates publication-quality visualizations  
- **Release Date Integration**: Tracks capability progression over time

## Configuration Management

### Centralized Configuration (`config.py`)

**Environment Variables**: API keys, debug flags, operational parameters  
**Path Management**: Consistent directory structure across all components  
**Model Defaults**: Standard model configurations and parameters  
**Evaluation Parameters**: Timeout values, iteration limits, sampling rates

## Extension Points

### Adding New Datasets

1. **Create Dataset Directory**: `src/human_ttc_eval/datasets/newdataset/`
2. **Implement Four Components**: 
   - `newdataset_retrieve.py` with `@register_retriever("newdataset")`
   - `newdataset_prepare.py` with `@register_preparer("newdataset")`
   - `newdataset_describe.py` with `@register_describer("newdataset")`
   - `newdataset_bench.py` with `@register_bench("newdataset")`
3. **Update Configuration**: Add to `DATASETS` list in Makefile
4. **Test Integration**: Verify CLI discovers and operates on new dataset

### Adding New Models

1. **Model Metadata**: Add entry to `models.json`
2. **Evaluation Compatibility**: Ensure model works with inspect_ai harnesses
3. **Configuration**: Add any model-specific parameters to config
