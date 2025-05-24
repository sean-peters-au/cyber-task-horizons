.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

CyBench Dataset Integration
===========================

Professional CTF challenges providing long-horizon cybersecurity tasks (30 minutes - 4 hours).

Overview
--------

+--------------------+----------------------------------------------------------------------------------------------+
| **Source**         | CyBench: Framework for Evaluating Cybersecurity Capabilities of LMs                        |
+--------------------+----------------------------------------------------------------------------------------------+
| **Size**           | 40 professional CTF challenges from recent competitions                                     |
+--------------------+----------------------------------------------------------------------------------------------+
| **Categories**     | Web exploitation, binary pwn, reverse engineering, cryptography                             |
+--------------------+----------------------------------------------------------------------------------------------+
| **Time Range**     | 30 minutes - 4 hours (first-solve times from competition data)                              |
+--------------------+----------------------------------------------------------------------------------------------+
| **License**        | Apache-2.0                                                                                   |
+--------------------+----------------------------------------------------------------------------------------------+
| **Repository**     | https://github.com/andyzorigin/cybench                                                      |
+--------------------+----------------------------------------------------------------------------------------------+

Dataset Structure
-----------------

Each challenge includes:
- Docker environment with vulnerable services
- Challenge description and setup instructions  
- Flag checker for automated grading
- Optional subtasks for guided evaluation
- First-solve time metadata from original competitions

**⚠️ Critical Timing Data Issues**

The current timing data has significant limitations for horizon analysis:

**Contamination Problem:** 
"Fastest solve time" may represent time from competition start to first solve, not isolated task completion time. A team solving a problem at 130 minutes may have spent 120 minutes on other problems and only 10 minutes on this specific task.

**Competition Context:**
- Teams typically solve easier problems first (strategic ordering)
- Multiple team members may work on different problems simultaneously  
- Submission times reflect competition strategy, not focused expert time
- Are we sure these are team based competitions?

**Potential Solutions (Future Work):**
1. **Scoreboard Analysis:** Parse competition scoreboards to approximate individual task time as ``problem_solve_time - last_submission_time_other_problem``
2. **Percentile Filtering:** Use 90th percentile of solve times to reduce noise from lucky/strategic solves
3. **Team Dynamics:** Account for parallel work in team-based competitions
4. **Expert Validation:** Cross-reference with expert estimates for task complexity

**Current Status:** Using fastest solve times as directional proxy. Results should be interpreted as lower bounds on expert task completion time, not precise human baselines.

Competition Sources
-------------------

Challenges drawn from recent professional CTFs:
- **Hack The Box Cyber Apocalypse 2024**
- **SekaiCTF 2024** 
- **GlacierCTF 2023**
- **HKCERT CTF 2023**

Processing Pipeline
-------------------

**1. Download & Setup**
Clone repository and pull Docker images.

**2. Challenge Metadata**
Each task includes competition timing data:
- First-solve time (validated against scoreboard data)
- Solving team information
- Total number of solves
- Difficulty category and point value

**3. Execution Environment**
Docker-based sandboxed environments with:
- Isolated networking for multi-service challenges
- Resource limits and security constraints
- Automated flag validation

Current Status
--------------

**Implementation Status:** 🔄 Integration in progress

Challenges encountered:
- Docker image download timeouts during evaluation
- Nested Docker complexity on some platforms
- Network configuration requirements

**Next Steps:**
- Resolve Docker networking and download issues
- Implement retriever/parser/summariser modules
- Add to makefile automation

Integration Architecture
------------------------

Will follow standard dataset pattern:

**Retriever Module**
- Clone CyBench repository
- Pull required Docker images
- Validate challenge metadata

**Parser Module**  
- Process challenge descriptions and metadata
- Extract timing information from competition data
- Generate standardized task format

**Summariser Module**
- Analyze difficulty distribution
- Generate timing statistics
- Create category breakdowns

Usage *(Future)*
-----------------

.. code-block:: bash

    # Download and process
    make cybench-parse
    
    # Generate summaries  
    make cybench-summarise
    
    # Output locations
    data/processed/cybench/web_tasks.jsonl
    data/processed/cybench/pwn_tasks.jsonl
    data/processed/cybench/crypto_tasks.jsonl
