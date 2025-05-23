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

Dataset Portfolio
-----------------

Currently implemented datasets spanning 10 seconds to hours:

+------------------+------------------+------------------+-------------------------+
| Dataset          | Time Range       | Tasks            | Status                  |
+==================+==================+==================+=========================+
| **NL2Bash**      | 10s - 5 min      | 4,247 atomic     | ✅ Implemented           |
+------------------+------------------+------------------+-------------------------+
| **CyBench**      | 30 min - 4 hours | 40 CTF exploits  | 🔄 Integration started  |
+------------------+------------------+------------------+-------------------------+

Dataset Characteristics
-----------------------

**NL2Bash: Short Horizon Tasks**
- Natural language to bash command translation
- Atomic operations (no pipes, redirects, chaining)
- LLM + heuristic time estimation
- Deterministic grading via command output comparison
- Covers basic shell literacy and tool usage

**CyBench: Long Horizon Tasks**  
- Professional CTF challenges from recent competitions
- Categories: web, pwn, reverse engineering, crypto
- First-solve times from competition data (30min - 24h)
- Docker-based execution environment
- Complex multi-step exploitation scenarios

Processing Architecture
-----------------------

Each dataset follows a consistent pattern:

**1. Retriever Module**
Auto-downloads source data and handles caching.

**2. Parser Module**  
Processes raw data into standardized task format with metadata.

**3. Summariser Module**
Generates statistics and analysis of processed tasks.

**4. Bench Module** *(Future)*
Agent evaluation harness for running models against tasks.

Common Task Schema
------------------

All datasets export tasks in consistent JSONL format:

.. code-block:: json

    {
      "id": "unique_task_id",
      "description": "task description", 
      "expected_time_seconds": 180.0,
      "timing_source": "llm|heuristic|empirical",
      "category": "dataset_specific_category",
      "difficulty": "easy|medium|hard",
      "metadata": {}
    }

Usage
-----

.. code-block:: bash

    # Process individual datasets
    make nl2bash-parse
    make nl2bash-summarise
    
    # View available targets
    make help
