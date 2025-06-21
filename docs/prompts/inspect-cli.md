# Inspect AI CLI Usage Guide

## Overview

The Inspect AI CLI is a powerful tool for evaluating AI models on various tasks and analyzing the results. In the context of the Human TTC Eval project, it's used to run evaluations on cybersecurity benchmarks and review the resulting `.eval` files that contain detailed logs of model performance.

## Installation and Basic Usage

All Inspect commands in this project are run through `uv`:

```bash
uv run inspect [COMMAND] [OPTIONS]
```

To see available commands and options:

```bash
uv run inspect --help
```

## Main Commands Overview

The Inspect CLI provides several commands for different purposes:

- **`cache`**: Manage the inspect model output cache
- **`eval`**: Evaluate tasks
- **`eval-retry`**: Retry failed evaluation(s)
- **`eval-set`**: Evaluate a set of tasks with retries
- **`info`**: Read configuration and log info
- **`list`**: List tasks on the filesystem
- **`log`**: Query, read, and convert logs (focus of this guide)
- **`sandbox`**: Manage Sandbox Environments
- **`score`**: Score a previous evaluation run
- **`trace`**: List and read execution traces
- **`view`**: Inspect log viewer

## Log Command Deep Dive

The `log` command is essential for reviewing evaluation results. Inspect supports two log formats:
- **`eval`**: A compact, high-performance binary format (default)
- **`json`**: Logs represented as JSON

### Available Log Subcommands

```bash
uv run inspect log --help
```

#### `inspect log list`
Lists all logs in the log directory:

```bash
uv run inspect log list
```

#### `inspect log dump`
Prints log file contents as JSON. This is the primary command for reviewing `.eval` files:

```bash
# Dump full log contents
uv run inspect log dump [PATH_TO_EVAL_FILE]

# Dump only the header (useful for large files)
uv run inspect log dump --header-only [PATH_TO_EVAL_FILE]
```

Example:
```bash
uv run inspect log dump results/benchmarks/nyuctf/inspect_logs/2025-06-18T18-33-22+10-00_nyuctf-task_KH3MX8nXtPMhCiej7QbJMS.eval
```

#### `inspect log convert`
Converts between log file formats:

```bash
uv run inspect log convert [SOURCE] [DESTINATION]
```

#### `inspect log schema`
Prints the JSON schema for log files:

```bash
uv run inspect log schema
```

## Understanding .eval Files

When you run `inspect log dump` on an `.eval` file, it returns a large JSON object with the following structure:

### Top-Level Structure

```json
{
  "version": 2,
  "status": "success",
  "eval": {...},
  "plan": {...},
  "reductions": {...},
  "results": {...},
  "samples": [...],
  "stats": {...}
}
```

### Key Sections

#### `eval` Section
Contains metadata about the evaluation run:
- `run_id`: Unique identifier for this run
- `created`: Timestamp when evaluation started
- `task`: Name of the task being evaluated
- `task_id`: Unique task identifier
- `dataset`: Information about samples (count, IDs)
- `model`: Model being evaluated
- `config`: Evaluation configuration

#### `samples` Section
An array containing detailed information for each evaluated sample:
- `id`: Sample identifier (e.g., "2017f-pwn-humm_sch_t")
- `epoch`: Sample number in the evaluation
- `input`: The prompt/task given to the model
- `target`: Expected output or goal
- `output`: Model's actual output
- `scores`: Scoring results for this sample
- `messages`: Full conversation history
- `model_usage`: Token usage for this sample
- `total_time`: Total execution time
- `sandbox`: Sandbox environment details

#### `results` Section
Summary of evaluation results:
- `total_samples`: Number of samples evaluated
- `completed_samples`: Number of samples completed
- `scores`: Aggregated scoring results

#### `stats` Section
Overall evaluation statistics:
- `started_at`: Evaluation start time
- `completed_at`: Evaluation end time
- `model_usage`: Aggregated token usage across all samples

## Using jq for Navigation

Since `.eval` files can be extremely large, `jq` is essential for extracting specific information. Here are useful queries:

### Basic Navigation

```bash
# Get all top-level keys
uv run inspect log dump [EVAL_FILE] | jq 'keys'

# Get evaluation metadata
uv run inspect log dump [EVAL_FILE] | jq '.eval'

# Get overall statistics
uv run inspect log dump [EVAL_FILE] | jq '.stats'

# Get results summary
uv run inspect log dump [EVAL_FILE] | jq '.results'
```

### Working with Samples

```bash
# Count number of samples
uv run inspect log dump [EVAL_FILE] | jq '.samples | length'

# Get first sample
uv run inspect log dump [EVAL_FILE] | jq '.samples[0]'

# Get all sample IDs
uv run inspect log dump [EVAL_FILE] | jq '.samples[].id'

# Get specific sample by ID
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.id == "2017f-pwn-humm_sch_t")'
```

### Extracting Specific Information

```bash
# Get all sample scores
uv run inspect log dump [EVAL_FILE] | jq '.samples[].scores'

# Get samples that passed (score = 1)
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.scores.score == 1) | .id'

# Get model usage per sample
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | {id: .id, tokens: .model_usage.total_tokens}'

# Get execution times
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | {id: .id, time: .total_time}'
```

### Aggregating Data

```bash
# Calculate average execution time
uv run inspect log dump [EVAL_FILE] | jq '[.samples[].total_time] | add / length'

# Sum total tokens used
uv run inspect log dump [EVAL_FILE] | jq '[.samples[].model_usage.total_tokens] | add'

# Count passed vs failed samples
uv run inspect log dump [EVAL_FILE] | jq '.samples | group_by(.scores.score) | map({score: .[0].scores.score, count: length})'
```

### Filtering and Analysis

```bash
# Get failed samples with their error messages
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.scores.score == 0) | {id: .id, error: .output}'

# Find samples that took longest to execute
uv run inspect log dump [EVAL_FILE] | jq '.samples | sort_by(.total_time) | reverse | .[0:5] | .[] | {id: .id, time: .total_time}'

# Extract specific message from conversation history
uv run inspect log dump [EVAL_FILE] | jq '.samples[0].messages[] | select(.role == "assistant") | .content'
```

### Combining with Other Tools

```bash
# Save specific sample to file
uv run inspect log dump [EVAL_FILE] | jq '.samples[0]' > sample_0.json

# Create CSV of results
uv run inspect log dump [EVAL_FILE] | jq -r '.samples[] | [.id, .scores.score, .total_time] | @csv' > results.csv

# Pretty print with less
uv run inspect log dump --header-only [EVAL_FILE] | jq '.' | less
```