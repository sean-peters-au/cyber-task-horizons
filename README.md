# Human Time-to-Completion Evaluation

A Python toolkit for processing and analyzing cybersecurity datasets to replicate METR's time-to-completion horizon analysis methodology.

## Features

- **Dataset Processing**: Standardized parsing and analysis of multiple cybersecurity datasets
- **METR Compatibility**: Converts datasets to METR-compatible formats for AI evaluation
- **Comprehensive Analysis**: Generates statistics, visualizations, and summaries
- **AI Benchmarking**: Run evaluations using industry-standard frameworks
- **Modular Architecture**: Extensible base classes for adding new datasets

## Supported Datasets

| Dataset | Status | Description | Benchmark Integration |
|---------|---------|-------------|----------------------|
| **KYPO** | ✅ Complete | Cybersecurity training logs with human timing | ✅ Analysis ready |
| **CyBench** | ✅ Complete | CTF challenges with difficulty ratings | ✅ Native evaluation system |
| **NL2Bash** | ✅ Complete | Natural language to bash translation | 🔄 InspectAI (planned) |

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd human-ttc-eval

# Install dependencies
pip install uv
uv sync

# Clone third-party repositories
git clone https://github.com/andyzorigin/cybench.git third-party/cybench
```

### Basic Usage

#### Dataset Processing

```bash
# Process KYPO dataset
make kypo-parse
make kypo-summarise

# Process CyBench dataset
make cybench-retrieve
make cybench-parse  
make cybench-summarise
```

#### AI Benchmarking

```bash
# Run CyBench evaluation with default model
make cybench-benchmark

# Run with specific model and parameters
make cybench-benchmark MODEL=anthropic/claude-3-5-sonnet-20240620 MAX_ITERATIONS=10

# Run in unguided mode
make cybench-benchmark-unguided
```

## Detailed Usage

### CyBench AI Evaluation

Our system wraps CyBench's native evaluation framework while providing a standardized interface and output format.

#### CLI Usage

```bash
# Basic evaluation
python -m human_ttc_eval.cli benchmark cybench \
  --cybench-repo-path third-party/cybench \
  --model openai/gpt-4o-2024-05-13 \
  --output-dir results/benchmarks/cybench

# Specific tasks only
python -m human_ttc_eval.cli benchmark cybench \
  --cybench-repo-path third-party/cybench \
  --model anthropic/claude-3-5-sonnet-20240620 \
  --output-dir results/benchmarks/cybench \
  --tasks "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic,benchmark/hackthebox/cyber-apocalypse-2024/web/[Easy] Flag Command" \
  --max-iterations 5 \
  --unguided-mode
```

#### Programmatic Usage

```python
from pathlib import Path
from human_ttc_eval.datasets.cybench.bench import CyBenchBench

# Initialize benchmark runner
bench = CyBenchBench(
    output_dir=Path("results/benchmarks/cybench"),
    cybench_repo_path=Path("third-party/cybench")
)

# Run evaluation
result = bench.run_evaluation(
    model_name="openai/gpt-4o-2024-05-13",
    max_iterations=15,
    unguided_mode=False
)

# Check results
if result.success:
    print(f"Success rate: {result.summary_stats['success_rate']:.2%}")
    print(f"Tasks completed: {result.summary_stats['successful_tasks']}/{result.summary_stats['total_tasks']}")
```

#### Supported Models

CyBench supports the following models (API keys required):

- **OpenAI**: gpt-4o-2024-05-13, gpt-4-turbo-2024-04-09, gpt-3.5-turbo-0125
- **Anthropic**: claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-haiku-20240307
- **Google**: gemini-1.5-pro-001, gemini-1.0-pro-001
- **Together**: llama-3.1-405b-instruct-turbo, mixtral-8x22b-instruct-v0.1
- **And more** - see `CyBenchBench.AVAILABLE_MODELS` for full list

### Dataset Processing

#### KYPO Dataset

```bash
# Parse logs to METR format
python -m human_ttc_eval.cli parse kypo \
  --input-dir data/cybersecurity_dataset_v4 \
  --output-file data/cybersecurity_human_runs.jsonl

# Generate summaries and plots
python -m human_ttc_eval.cli summarise kypo \
  --jsonl-file data/cybersecurity_human_runs.jsonl \
  --output-dir results/dataset-summaries/kypo
```

#### CyBench Dataset

```bash
# Extract metadata from repository
python -m human_ttc_eval.cli retrieve cybench-metadata \
  --cybench-repo-path third-party/cybench \
  --output-dir data/raw/cybench

# Convert to METR format
python -m human_ttc_eval.cli parse cybench \
  --input-dir data/raw/cybench \
  --output-file data/cybench_human_runs.jsonl

# Generate analysis
python -m human_ttc_eval.cli summarise cybench \
  --jsonl-file data/cybench_human_runs.jsonl \
  --output-dir results/dataset-summaries/cybench
```

## Architecture

### Base Classes

- **BaseBench**: Abstract base for AI evaluation systems (supports both external tools and InspectAI)
- **BaseParser**: Converts raw datasets to METR-compatible JSONL format
- **BaseSummariser**: Generates statistics and visualizations
- **BaseRetriever**: Fetches raw data from APIs or repositories

### Registry System

Components automatically register themselves using decorators:

```python
@register_parser("my_dataset")
class MyDatasetParser(BaseParser):
    # Implementation

@register_summariser("my_dataset") 
class MyDatasetSummariser(BaseSummariser):
    # Implementation
```

### Standardized Outputs

- **METR Format**: All datasets converted to consistent JSONL schema
- **BenchmarkResult**: Standardized evaluation results across frameworks
- **Comprehensive Logging**: Detailed logs for reproducibility and debugging

## METR Schema Compliance

All parsed datasets follow METR's `all_runs.jsonl` format:

```json
{
  "task_id": "unique_task_identifier",
  "task_family": "grouped_task_category", 
  "run_id": "unique_run_identifier",
  "alias": "human_readable_name",
  "model": "human_baseline",
  "score_binarized": 1,
  "human_minutes": 45.5,
  "metadata": { "additional": "fields" }
}
```

## Contributing

### Adding New Datasets

1. Implement the three base classes in `src/human_ttc_eval/datasets/your_dataset/`:
   - `retrieve.py` - Data collection (if needed)
   - `parser.py` - METR format conversion  
   - `summariser.py` - Analysis and visualization

2. Use the registry decorators for automatic discovery

3. Add CLI commands and Makefile targets

4. Update documentation

### Adding AI Evaluation Support

For datasets without existing evaluation frameworks:

1. Create an InspectAI task in `src/human_ttc_eval/inspect_tasks/`
2. Implement evaluation logic using InspectAI's agent and scoring systems

For datasets with existing frameworks:

1. Create a `bench.py` that extends `BaseBench`
2. Wrap the external system with subprocess calls
3. Parse results into `BenchmarkResult` format

## Environment Setup

### API Keys

For AI evaluations, set the required API keys in your environment:

```bash
# OpenAI
export OPENAI_API_KEY="your_key_here"

# Anthropic  
export ANTHROPIC_API_KEY="your_key_here"

# Google
export GOOGLE_API_KEY="your_key_here"

# Together AI
export TOGETHER_API_KEY="your_key_here"
```

### Docker Requirements

CyBench evaluations require Docker for challenge environments:

```bash
# Ensure Docker is running
docker --version

# CyBench will automatically pull required images
```

## Data Format Examples

### KYPO Human Runs
```json
{
  "task_id": "kypo_locust_3302_2019_12_12_task_3",
  "task_family": "locust_3302", 
  "human_minutes": 156.7,
  "success": true,
  "difficulty": "medium"
}
```

### CyBench Human Runs  
```json
{
  "task_id": "cybench_hackthebox_crypto_dynastic",
  "task_family": "hackthebox_cyber_apocalypse_2024_crypto_dynastic",
  "human_minutes": 8.5,
  "organization": "hackthebox",
  "category": "crypto"
}
```

## Results and Analysis

Generated outputs include:

- **Summary Statistics**: Overall performance metrics by dataset
- **Per-Category Analysis**: Breakdown by challenge type/difficulty
- **Timing Distributions**: Histograms and statistical analysis
- **Visualizations**: Bar charts, scatter plots, distribution plots
- **Benchmark Results**: AI model performance vs human baselines

Example outputs saved to:
- `results/dataset-summaries/{dataset}/` - Processing results
- `results/benchmarks/{dataset}/` - AI evaluation results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **METR** for the time-to-completion horizon analysis methodology
- **CyBench Team** for the excellent cybersecurity evaluation framework
- **KYPO** for realistic cybersecurity scenario training data
- **CyBench** for standardized challenge evaluation  
- **NL2Bash** for command-line interface tasks
