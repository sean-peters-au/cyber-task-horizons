.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

NL2Bash Dataset Integration
===========================

Natural language to bash command translation dataset providing short-horizon tasks (10 seconds - 5 minutes).

Overview
--------

+------------------+-------------------------------------------------------------------------------------------+
| **Source**       | TellinaTool/nl2bash - "Natural Language Interface to the Linux OS"                       |
+------------------+-------------------------------------------------------------------------------------------+
| **Size**         | 12,607 NL→bash pairs, filtered to 4,247 atomic tasks                                     |
+------------------+-------------------------------------------------------------------------------------------+
| **Time Range**   | 10 seconds - 5 minutes (estimated via LLM + heuristics)                                  |
+------------------+-------------------------------------------------------------------------------------------+
| **License**      | GPL-3.0                                                                                   |
+------------------+-------------------------------------------------------------------------------------------+
| **Repository**   | https://github.com/TellinaTool/nl2bash                                                   |
+------------------+-------------------------------------------------------------------------------------------+

Dataset Structure
-----------------

Raw data consists of parallel text files:
- `*.nl`: Natural language descriptions ("count lines in file.txt")  
- `*.cm`: Corresponding bash commands ("wc -l file.txt")

Processing Pipeline
-------------------

**1. Download & Retrieval**
Auto-downloads from GitHub if not present locally.

**2. Filtering** 
Atomic tasks only:
- No pipes, redirects, or command chaining
- Common utilities (find, grep, awk, sed, etc.)
- Simple, deterministic operations

**3. Time Estimation**
Two-phase approach:
- Heuristic baseline (complexity + word count)
- LLM refinement using Claude 3.5 Sonnet in parallel batches

**4. Complexity Analysis**
Tasks categorized by:
- Word count and utility types
- Complexity score (1.0-6.0 scale)
- Has pipes/redirects/subcommands flags

Task Categories
---------------

+-------------+-------+---------------------------+------------------+
| Category    | Count | Complexity Range          | Example          |
+=============+=======+===========================+==================+
| **Atomic**  | 4,247 | Simple, no composition    | `ls -la`         |
+-------------+-------+---------------------------+------------------+
| **Simple**  | 8,818 | Basic operations          | `find . -name x` |
+-------------+-------+---------------------------+------------------+
| **Medium**  | 3,541 | Moderate complexity       | `awk '{print $2}'`|
+-------------+-------+---------------------------+------------------+
| **Complex** | 248   | Advanced operations       | Multi-step chains|
+-------------+-------+---------------------------+------------------+

Output Format
-------------

Each task exported as JSONL with:

.. code-block:: json

    {
      "id": 1234,
      "nl_description": "list all .txt files",
      "bash_command": "ls *.txt",
      "word_count": 2,
      "complexity_score": 1.2,
      "estimated_time_seconds": 15.0,
      "timing_source": "llm",
      "utilities_used": ["ls"]
    }

Usage
-----

.. code-block:: bash

    # Download and process
    make nl2bash-parse
    
    # Generate summaries  
    make nl2bash-summarise
    
    # Output locations
    data/processed/nl2bash/atomic_tasks.jsonl
    data/processed/nl2bash/simple_tasks.jsonl
