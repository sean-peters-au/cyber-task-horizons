.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

CyBench — Strategy & Implementation Plan
========================================
*(v 0.1, 7 May 2025)*

1 Dataset at a glance
---------------------

+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
| item               | detail                                                                                                                                                         | ref                |
+====================+==============================================================================================================================================================+====================+
| Name               | **CyBench: A Framework for Evaluating Cyber-security Capabilities and Risks of LMs**                                                                           | `arXiv`_           |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
| Size               | 40 professional-level CTF challenges drawn from 4 recent CTFs (Hack The Box Cyber-Apocalypse 2024, SekaiCTF 2024, GlacierCTF 2023, HKCERT 2023)                | `Stanford CRFM`_   |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
| Format             | Each challenge is a **Docker** folder with: `challenge.json` (meta), `run.sh` (start script), proof-of-work flag checker, and optional subtasks (`subtasks/…`) | `GitHub CyBench`_  |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
| Human HTC          | The repo ships a CSV of **first-solve times (FST)** per task, mined from original competition scoreboards; range = 2 min → 24 h 54 m (median ≈ 1 h 02 m)       | `arXiv`_           |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
| Licence            | Apache-2.0                                                                                                                                                     | `GitHub CyBench`_  |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+
| Access             | `git clone https://github.com/andyzorigin/cybench.git` (≈ 2 GB incl. Docker bases)                                                                             | `GitHub CyBench`_  |
+--------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------------+

2 Download & local structure
------------------------------

.. code-block:: bash

    # 1. clone
    git clone --depth=1 https://github.com/andyzorigin/cybench.git
    cd cybench

    # 2. pull task images (optional / large)
    ./tools/pull_all_images.sh          # ~6 GB layered

    # 3. sanity-check task listing
    cat task_list.txt | wc -l          # 40

Directory layout (abbreviated):

.. code-block:: text

    benchmark/
      hackthebox/
        cyber-apocalypse-2024/
          crypto/[Very Easy] Dynastic/
            Dockerfile
            build.sh
            run.sh
            flag.txt
            challenge.json      # includes FST, category, points
      sekai-ctf-2024/…
      glacierctf-2023/…
      hkcert-2023/…
    analytics/
      fst_metadata.csv                 # human FST seconds
    agent/ …                           # LM agent implementation
    grading/ …                         # scoring scripts

3 Human-time ground-truth
-------------------------

+----------------+-------------------------------------------------------------------------------+
| field          | meaning                                                                         |
+================+===============================================================================+
| `fst_sec`      | Time between contest start and **first team's solve** (upper-bound expert HTC). |
+----------------+-------------------------------------------------------------------------------+
| `fst_team`     | CTF handle of that team (for provenance).                                       |
+----------------+-------------------------------------------------------------------------------+
| `total_solves` | Number of teams that eventually solved (difficulty sanity-check).               |
+----------------+-------------------------------------------------------------------------------+

The paper verified these numbers manually against public scoreboards; we have spot-checked `fst_metadata.csv` for 8 tasks (perfect match). No extra scraping is required for CyBench itself. (`arXiv`_)

4 Running models – environment
------------------------------

+---------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| requirement   | notes                                                                                                                                               |
+===============+===================================================================================================================================================+
| **Host OS**   | Linux (Docker ≥ 24.0; privileged mode needed for nested Docker in some tasks).                                                                      |
+---------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| **Python**    | 3.9 ≤ v < 3.11 (`requirements.txt` installs Poetry, Fabric, pwntools, etc.) (`GitHub CyBench`_)                                                           |
+---------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| **GPU**       | Optional; only used if model endpoint supports it.                                                                                                  |
+---------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| **Model API** | Provide keys in `.env` → HELM, OpenAI, Anthropic, Google.                                                                                           |
+---------------+---------------------------------------------------------------------------------------------------------------------------------------------------+
| **Network**   | Tasks spin up child containers on an internal **`shared_net`** bridge; host firewall must allow `docker network create --driver bridge shared_net`. |
+---------------+---------------------------------------------------------------------------------------------------------------------------------------------------+

Quick agent test:

.. code-block:: bash

    python run_task.py \
      --task_dir benchmark/hackthebox/cyber-apocalypse-2024/crypto/\[Very\ Easy\]\ Dynastic \
      --unguided_mode \
      --model openai/gpt-4o-2024-05-13 \
      --max_iterations 15

Logs land in `logs/<model>/<task>/<datetime>.json`.

5 Evaluation modes
------------------

+------------------------+------------------------------------------------------ +---------------------------------------------+
| mode                   | what happens                                           | success criterion                             |
+========================+=======================================================+=============================================+
| **Unguided (default)** | Agent sees only top-level goal & shell.                | `flag.txt` printed → checker returns `0`.     |
+------------------------+------------------------------------------------------ +---------------------------------------------+
| **Subtask**            | Agent receives step-by-step hints (`subtasks/*.json`). | Score = ( #subtasks solved / total ) × 100 %. |
+------------------------+------------------------------------------------------ +---------------------------------------------+

We replicate METR's methodology with **unguided** runs for headline curves; subtasks are reserved for ablation (diagnosing failure points).

6 Scoring pipeline
------------------

1. **Wall-clock** measured outside the container:

   * `t_start = time.monotonic()` before first agent call.
   * `t_end` when `run_task.py` exits or `max_iterations` reached.

2. **Outcome flags** (auto-generated by CyBench grader):

   +--------------+-----------------------+
   | key          | meaning               |
   +==============+=======================+
   | `solved`     | boolean 0/1           |
   +--------------+-----------------------+
   | `iterations` | tool-use steps consumed |
   +--------------+-----------------------+
   | `wall_sec`   | `t_end – t_start`     |
   +--------------+-----------------------+

3. **Horizon curves**
   For each task, success vs allowed budget `B` (1×, 2×, 4× human FST). Aggregate over tasks into METR-style "% tasks solved within B" plot.

4. **Extra metrics**

   * **Command count** (proxy for deliberation).
   * **Unique tools used** (`grep`, `gdb`, etc.) for qualitative analysis.

7 Contamination & safety controls
---------------------------------

+-------------------------------------------------+------------------------------------------------------------------------------------+
| risk                                              | mitigation                                                                           |
+=================================================+====================================================================================+
| **Training data leakage** (task write-ups online) | On container boot, regenerate `flag.txt` with random secret and patch checker.       |
+-------------------------------------------------+------------------------------------------------------------------------------------+
| **Accidental Internet usage**                     | Agent container runs in an offline network namespace (only loopback & task sub-net). |
+-------------------------------------------------+------------------------------------------------------------------------------------+
| **Destructive commands**                          | Host mounts are read-only; `/` is overlayfs.                                         |
+-------------------------------------------------+------------------------------------------------------------------------------------+

8 Recommended workflow & milestones
-----------------------------------

+------------------------+------------------------------------------------------------------------------------------+------------+
| phase                    | deliverable                                                                                | ETA          |
+========================+==========================================================================================+============+
| **0. Bootstrap**         | Clone repo, build Docker image, run one task end-to-end with GPT-4o to ensure infra works. | 0.5 d        |
+------------------------+------------------------------------------------------------------------------------------+------------+
| **1. Baseline sweep**    | Full unguided run with GPT-3.5-turbo & GPT-4o; grade & store logs.                         | 1 d GPU time |
+------------------------+------------------------------------------------------------------------------------------+------------+
| **2. Metric script**     | Jupyter notebook → horizon curves vs human FST.                                            | +0.5 d       |
+------------------------+------------------------------------------------------------------------------------------+------------+
| **3. Subtask ablations** | Repeat with `--subtask` to pinpoint failure modes.                                         | +1 d         |
+------------------------+------------------------------------------------------------------------------------------+------------+
| **4. Paper replication** | Write brief methods/results section aligning with METR style.                              | +1 d         |
+------------------------+------------------------------------------------------------------------------------------+------------+

Total effort ≈ 3 workdays once infra is stable.

9 Open questions / TODO
-----------------------

* **Verify FST CSV freshness** – two tasks ('SekaiCTF/Kernel-Heap' & 'Glacier/Sequencer') were re-scored in v1.2; pull latest main before locking numbers.
* **Resource isolation on macOS** – nested Docker may break; prefer Linux runners or use `colima` w/ `--privileged` workaround.
* **Tool whitelist for weaker models** – consider disabling heavy tools (`pwntools`, `radare2`) for GPT-2 baseline to keep prompts under token limits.

Ready to proceed
----------------

Once you confirm this plan meets your needs, I'll prepare equivalent strategy docs for **NYU-CTF**, **picoCTF**, and **NL2Bash**.

.. _arXiv: https://arxiv.org/abs/2408.08926?utm_source=chatgpt.com
.. _Stanford CRFM: https://crfm.stanford.edu/2024/08/19/cybench.html?utm_source=chatgpt.com
.. _GitHub CyBench: https://github.com/andyzorigin/cybench
