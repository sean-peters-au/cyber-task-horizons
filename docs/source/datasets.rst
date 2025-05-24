.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

Benchmark Datasets Overview
===========================

Time-to-completion horizon analysis across multiple cybersecurity datasets.

.. toctree::
   :maxdepth: 1
   :caption: Dataset Documentation:

   nl2bash
   cybench
   nyuctf

Dataset Portfolio
-----------------

Currently implemented and planned datasets, spanning a wide range of task complexities and completion times:

+--------------------+------------------------+---------------------------------------+-------------------------+
| Dataset            | Est. Time Range        | Tasks / Focus                         | Status                  |
+====================+========================+=======================================+=========================+
| **NL2Bash**        | 10s - 5 min            | ~4,000 atomic bash commands           | ✅ Implemented           |
+--------------------+------------------------+---------------------------------------+-------------------------+
| **CyBench**        | 30 min - 24 hours      | ~40 CTF exploits (web, pwn, RE etc.)  | ✅ Implemented           |
+--------------------+------------------------+---------------------------------------+-------------------------+
| **NYU-CTF Bench**  | 10 min - 2 hours (est.)| ~200 Dockerized CTF challenges        | 📝 Planned (Theoretical)|
+--------------------+------------------------+---------------------------------------+-------------------------+
| **InterCode-CTF**  | ~3.5 min avg (authors) | PicoCTF-based (High School level)     | 📝 Planned (Research)    |
+--------------------+------------------------+---------------------------------------+-------------------------+

Dataset Characteristics
-----------------------

**NL2Bash: Short Horizon Tasks**
- Natural language to bash command translation.
- Primarily atomic operations (no pipes, complex redirects, or chaining initially filtered for some analyses).
- Time estimation: Heuristic-based, with an option for LLM-based refinement.
- Deterministic grading via command output comparison possible (though current benchmarks might use LLM-based scorers for functional equivalence).
- Covers basic shell literacy and common command-line tool usage.

**CyBench: Medium to Long Horizon Tasks**  
- Professional Capture The Flag (CTF) challenges from recent competitions.
- Categories: Web, Pwn, Reverse Engineering, Cryptography, etc.
- Time estimation: Based on first-solve times from actual CTF competitions (typically 30 minutes to 24 hours).
- Execution: Docker-based environments specific to each challenge.
- Involves complex multi-step exploitation scenarios.
- Evaluation: Wraps CyBench's native evaluation harness.

**NYU-CTF Bench: Medium Horizon Tasks (Planned)**
- Source: `NYU CTF Dataset: A Scalable Open-Source Benchmark for Evaluating LLMs in Offensive Security (arXiv:2406.05590) <https://arxiv.org/abs/2406.05590>`_.
- Interactive cybersecurity tasks from the CSAW-CTF competition series (2011–2023).
- ~200 Dockerized challenges (test split) across web, pwn, rev, forensics, crypto, misc.
- Time estimation strategy: Plan to extract from CTFd solve logs, focusing on "first blood with zero prior solves" to establish a strict upper-bound human working time. Expected to yield ~45-60 tasks with high-confidence times in the 10 min – 2 hour range.
- Evaluation: Each challenge is self-contained with Docker Compose and an auto-grading flag checker.
- *Status: Theoretical planning stage. See the :doc:`nyuctf` page for the detailed (early-stage) plan.*

**InterCode-CTF: Short to Medium Horizon Tasks (Planned)**
- Source: `InterCode: Standardizing and Benchmarking Interactive Coding with Real-World Tools (arXiv:2306.14898) <https://arxiv.org/pdf/2306.14898>`_.
- Focuses on the CTF task environment within the broader InterCode benchmark, primarily using tasks from PicoCTF (targeting high school students).
- Tasks: Procedurally generated or adapted from PicoCTF, involving bash and python interaction within a minimalist CTF environment.
- Execution: Utilizes the InterCode execution framework.
- Time estimation strategy: 
    - The original InterCode paper focuses on agent success rates and command counts, not human completion times.
    - However, a related work (CyBench paper, ICLR 2025) notes that the *authors of InterCode-CTF reported an average of 3.5 minutes* to solve their tasks. This provides an initial, albeit rough, estimate.
    - For more granular or independently validated Human Time-to-Completion (HTC) data suitable for horizon analysis, a dedicated strategy (e.g., expert elicitation, targeted human studies, or adapting heuristics based on task complexity) would be required.
- Evaluation: Based on achieving goals within the interactive environment as defined by InterCode.
- *Status: Research and planning stage. The 3.5-minute average author solve time is a useful starting point, but a robust HTC collection methodology for our needs is TBD.*

Processing Architecture
-----------------------

Each dataset aims to follow a consistent processing pipeline (Retrieve → Parse → Summarise → Bench) as outlined in the :doc:`codebase` documentation. This ensures that data can be transformed into a standardized format suitable for the core analysis and plotting tools, particularly for METR-style horizon analysis.

Common Task Schema
------------------

While specific fields may vary, the goal of the `parse` step for each dataset is to produce data that can be mapped to a common conceptual schema for analysis, including:

- A unique task identifier.
- A textual description of the task.
- An estimated human completion time in seconds (`human_minutes` in some internal formats).
- The source or method of time estimation (e.g., `llm`, `heuristic`, `empirical_fst`).
- Relevant categories, difficulty levels, and other dataset-specific metadata.

This allows `analysis.transform` to create the `all_runs.jsonl` needed for plotting.

Usage
-----

Refer to the `Makefile` for common operations:

.. code-block:: bash

    # Process individual datasets (examples)
    make nl2bash-parse
    make cybench-summarise
    
    # Run benchmarks (example)
    make cybench-benchmark MODEL=openai/gpt-4o-2024-05-13

    # Generate plots
    make plot
    
    # View all available targets
    make help
