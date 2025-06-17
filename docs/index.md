# Human Time-to-Completion Evaluation Framework

## Research Motivation

This project implements a comprehensive evaluation framework for measuring AI capabilities on cybersecurity tasks using METR's time-to-completion horizon methodology. The central question driving this work is: **When will AI systems cross critical capability thresholds in offensive cybersecurity that could precipitate misuse and/or existential AI safety risks?**

Speed is leverage in cybersecurity. An AI system that can reduce exploit development from hours to minutes enables low-skill actors to launch sophisticated attacks at machine scale, potentially overwhelming defensive capacity. By tracking when AI agents cross human expert timing thresholds on real cybersecurity tasks, we aim to provide early signals of capability-driven risk escalation—well before these skills appear in deployed systems or black-box services.

## Methodology Overview

### Time Horizon Analysis

Following METR's approach from "Measuring AI Ability to Complete Long Tasks," we measure task difficulty by expert human completion time and determine each AI model's performance across varying time budgets. This reveals **horizon curves**—the percentage of tasks an AI system can complete within different time constraints.

Key insight: The time budget that allows an AI to solve 50% of tasks provides a meaningful capability benchmark, directly comparable to human expert performance on the same task distribution.

### Five-Dataset Stack (0.6 seconds → 25 hours)

Our evaluation spans five complementary datasets covering different time horizons:

| Dataset | Time Range | Focus | Human Baseline Source |
|---------|------------|-------|----------------------|
| **CyBashBench** | 0.6s-15s | Command reflexes | AI-assisted estimates (author-calibrated) |
| **NL2Bash** | 4s-4min | Command synthesis | AI-assisted estimates (CyBashBench-anchored) |
| **InterCode-CTF** | 10s-10min | Interactive solving | AI-assisted estimates (author-calibrated) |
| **NYUCTF** | 2min-6h | CTF challenges | AI-assisted estimates (CyBench-anchored) |
| **CyBench** | 2min-25h | Professional exploits | Competition solve times (AI-validated) |

This distribution allows us to capture capability progression across multiple orders of magnitude in task complexity.

## Implementation Approach

### Core Design Philosophy

1. **Simplicity over premature abstraction**: Prioritize clear, reproducible implementations over flexibility
2. **METR compatibility**: Strict adherence to METR's schema and analysis methodology
3. **Modular extensibility**: Clean interfaces for adding new datasets and models
4. **Transparent limitations**: Honest assessment of annotation quality and scope constraints

### Standard Pipeline

Each dataset follows a consistent four-stage pipeline:

1. **Retrieve**: Fetch raw data from external sources
2. **Prepare**: Transform to standardized METR format with human baselines
3. **Describe**: Generate dataset-specific analysis and visualizations  
4. **Bench**: Evaluate AI models using sandboxed environments

This ensures all datasets produce comparable outputs suitable for METR's horizon analysis tools.

## Important Limitations and Caveats

### Human Time Annotations

**Critical disclaimer**: Most human time annotations in this framework are AI-assisted estimates rather than empirical measurements from human experts. The quality and reliability vary significantly by dataset:

- **CyBashBench & NL2Bash**: Primarily LLM estimates with heuristic validation
- **InterCode-CTF**: Author-reported averages (limited sample)
- **NYUCTF**: Competition timing data (but may include strategic delays)
- **CyBench**: Competition solve times (contaminated by multi-tasking effects)

### Scope Constraints

- **Domain limitation**: Focused exclusively on technical cybersecurity tasks
- **Task isolation**: Individual challenges don't capture multi-day campaign complexity
- **Static evaluation**: No adaptation to defensive countermeasures
- **Skill proxy**: Command-line and exploit tasks may not fully represent expert knowledge

### Research vs. Production

This is fundamentally a learning and research project, representing one person's attempt to understand AI capability progression in cybersecurity. It should not be treated as:
- An authoritative benchmark for AI safety decisions
- A comprehensive assessment of cybersecurity AI capabilities  
- A replacement for expert human evaluation

## Project Status

This framework represents a working implementation that successfully reproduces METR's methodology for cybersecurity tasks. While the core pipeline is functional and all datasets are integrated, the human baseline quality varies and would benefit from expert validation.

The goal is to provide a foundation for more rigorous cybersecurity capability assessment while being transparent about current limitations.

## Documentation Structure

- **[Methodology](methodology.md)**: Detailed approach to time horizon analysis and human baseline collection
- **[Datasets](datasets.md)**: Overview of all five datasets with quality assessments
- **[Architecture](architecture.md)**: Technical implementation and design decisions
- **[Usage](usage.md)**: Getting started guide and common workflows

Individual dataset documentation:
- [CyBashBench](cybashbench.md) | [NL2Bash](nl2bash.md) | [InterCode-CTF](intercode_ctf.md) | [NYUCTF](nyuctf.md) | [CyBench](cybench.md)