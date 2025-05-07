.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

NYU-CTF Bench — Strategy & Implementation Plan
==============================================
*(v 0.1 · 7 May 2025)*

1 Dataset overview
------------------

+------------------+---------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| item             | detail                                                                                                                                        | source                 |
+==================+=============================================================================================================================================+========================+
| Name             | **NYU-CTF Bench** (a.k.a. “NYU CTF Dataset”)                                                                                                  | `arXiv NYU`_           |
+------------------+---------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| Purpose          | Evaluate LLM agents on *interactive* cybersecurity tasks drawn from the CSAW-CTF competition series (2011 – 2023).                            |                        |
+------------------+---------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| Size             | **200 Dockerised challenges** in the **test split**, plus a 55-challenge dev split. 6 categories: **web, pwn, rev, forensics, crypto, misc**. | `GitHub NYU`_          |
+------------------+---------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| Licence          | GPL-2.0                                                                                                                                       | `GitHub NYU`_          |
+------------------+---------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| Repo             | `https://github.com/NYU-LLM-CTF/NYU_CTF_Bench` and PyPI package `nyuctf`.                                                                     | `GitHub NYU`_, `PyPI NYU`_ |
+------------------+---------------------------------------------------------------------------------------------------------------------------------------------+------------------------+
| Associated paper | *“NYU CTF Dataset: A Scalable Open-Source Benchmark for Evaluating LLMs in Offensive Security”* (NeurIPS 2024 D&B).                          | `arXiv NYU`_           |
+------------------+---------------------------------------------------------------------------------------------------------------------------------------------+------------------------+

Folder layout (after install / clone):

.. code-block:: text

    test/
     └─ 2022/
         └─ CSAW-Finals/
             └─ pwn/
                 └─ "Strict Shell"/
                     ├─ challenge.json   # metadata, port, points
                     ├─ docker-compose.yaml
                     └─ run.sh           # boots server & checker
    development/           # 55-challenge “train” split
    scripts/
     nyu_scrape_scoreboard.py
     nyu_build_metadata.py

2 Obtaining the dataset
-----------------------

.. code-block:: bash

    # Option A – PyPI one-liner
    pip install -U nyuctf            # pulls code + lightweight index
    python -m nyuctf.download        # clones the ~3 GB repo

    # Option B – direct clone
    git clone --depth=1 https://github.com/NYU-LLM-CTF/NYU_CTF_Bench.git

Each challenge is self-contained: `docker compose up` brings up any servers plus an **auto-grading flag checker** referenced in `challenge.json` (`GitHub NYU`_).

3 Human time-to-completion (HTC) extraction
-------------------------------------------

3.1 Data source
~~~~~~~~~~~~~~~

All CSAW events run on **CTFd**, whose REST API exposes per-challenge solve logs:

.. code-block:: text

    GET https://ctf.csaw.io/api/v1/challenges/<id>/solves

→ returns ``[{"team_id": …, "date": "2024-09-15T14:22:11.000Z"}, …]`` (`JusCodin's Blog CTFdPy`_).
Scoreboards stay public after each contest ends.

3.2 Filter: *first_blood_zero_prior*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Goal = **strict upper-bound human working time**.
Algorithm for each challenge `c` within a given year:

1. Query `/challenges/<id>/solves` → list of solves `S`.
2. For every `(team, t_solve)` in `S`, fetch that team’s *earliest* solve in the contest:

   .. code-block:: python

      t_first_any = min(s["date"] for s in team_solves(team))

3. Keep a solve if `t_solve == t_first_any` (team had **no earlier solve**).
4. Take the *earliest* such `t_solve` ⇒ `htc_sec = (t_solve - T₀).total_seconds()`.

If no team passes the zero-prior check, mark `timing_quality="sparse"` and **omit** the challenge from timing-sensitive plots (still usable for pass/fail counts).

Skeleton scraper (≈120 LOC) is provided in `scripts/nyu_scrape_scoreboard.py`.

Expected retention ⬇

+----------------+----------------+------------------------------+
| contest year   | finals tasks   | tasks with ≥1 zero-prior solve |
+================+================+==============================+
| 2019–23        | \~20 / year    | 3 – 6 / year                 |
+----------------+----------------+------------------------------+

Across 10 years ≈ **45-60 tasks** with high-confidence upper-bound times, sufficient for the 10 min – 2 h horizon band.

3.3 Metadata file
~~~~~~~~~~~~~~~~~

.. code-block:: jsonc

    {
      "challenge_id": "2022f-pwn-strict-shell",
      "points": 300,
      "category": "pwn",
      "htc_sec": 2412,
      "timing_source": "first_blood_zero_prior",
      "year": 2022,
      "event": "CSAW-Finals",
      "total_solves": 118
    }

Stored at `metadata/human_time.jsonl` and cached by hash (so re-scrapes are incremental).

4 Model-execution environment
-----------------------------

+---------------------+-------------------------------------------------------------------------------------------------------------------------------+
| component           | requirement                                                                                                                     |
+=====================+===============================================================================================================================+
| **Host**            | Linux; Docker ≥ 24 with `docker compose`.                                                                                       |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------+
| **Network**         | Each challenge launches its own bridge (`ctf_net_<hash>`).  Agent container needs `--network host` or `docker exec` to connect. |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------+
| **Tools for agent** | BusyBox shell + pre-installed: `python3`, `pwntools`, `radare2`, `gdb`, `curl`, `jq`, `netcat`, `openssl`.                      |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------+
| **Model backend**   | Same abstraction layer as CyBench (`agent_runner.py`): supports OpenAI, Claude, Gemini via environment keys.                    |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------+

**Quick smoke test**

.. code-block:: bash

    python agent_runner.py \
      --challenge test/2022/CSAW-Finals/pwn/"Strict Shell" \
      --model openai/gpt-4o-2024-05-13 \
      --max_tokens 4096 --timeout 3600

The CLI spins the challenge services, drops the agent into `/workspace`, proxies tool calls, and captures transcripts.

5 Success checking & scoring
----------------------------

+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| step                | mechanism                                                                                                                                                   |
+===================+===========================================================================================================================================================+
| 1. **Flag check**   | Each challenge contains `flag.txt` or a network service returning the flag; `challenge.json["flag"]` encodes the regex. Agent must print a matching string. |
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| 2. **Runner exits** | `run_challenge.py` returns JSON: `{solved: 0/1, wall_sec, cmd_count}`.                                                                                      |
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+
| 3. **Aggregate**    | Per-challenge CSV → METR horizon curves: `% of tasks solved vs budget B ∈ {1×,2×,4×} human HTC`.                                                            |
+-------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------+

Flags are regenerated on container boot:

.. code-block:: bash

    FLAG=$(openssl rand -hex 16)
    sed -i "s/FLAG{[A-Za-z0-9]*}/FLAG{$FLAG}/" flag.txt checker.py

—preventing memorisation.

6 Contamination & safety
------------------------

* **Write-ups online** – new random flag, shuffled port numbers, randomise filenames (`secret.txt` → `s_9gdk.txt`) per boot.
* **Outbound Internet** – agent network is *isolated*; `/etc/resolv.conf` points to dummy DNS.
* **Resource abuse** – CPU/mem quotas passed to Docker; all writes confined to `/workspace/tmp`.

7 Milestones
------------

+-------------------------+-------------------------------------------------------------------------+------+
| phase                     | deliverable                                                               | time   |
+=========================+=========================================================================+======+
| **0. Clone & scrape**     | Repo + scoreboard JSON for 2015-24; build `human_time.jsonl`.             | 0.5 d  |
+-------------------------+-------------------------------------------------------------------------+------+
| **1. Baseline run**       | GPT-3.5 & GPT-4o on all timed tasks (strict filter).                      | 1 d    |
+-------------------------+-------------------------------------------------------------------------+------+
| **2. Horizon curves**     | Notebook generating METR-style plots; include CyBench overlay.            | +0.5 d |
+-------------------------+-------------------------------------------------------------------------+------+
| **3. Sparse-timing gaps** | Decide whether to keep or drop challenges with `timing_quality="sparse"`. | +0.5 d |
+-------------------------+-------------------------------------------------------------------------+------+
| **4. Paper section**      | “NYU-CTF replication” methods & results write-up.                         | +1 d   |
+-------------------------+-------------------------------------------------------------------------+------+

Total ≈ 3 work-days once infra is ready.

8 Open questions / next actions
-------------------------------

* **API auth** – some older CSAW scoreboards still require login; easiest fix is to spin up a headless browser once per contest and export cookies for the scrape script.
* **Dev vs test split** – keep the official *development* folder for *prompt-engineering only*; never score on it to avoid over-fitting.
* **Port collisions** – when running many challenges in parallel, randomise exposed host ports with `docker-compose --project-name`.

Ready for sign-off
------------------

If this NYU-CTF plan aligns with your expectations, the next strategy brief will cover **picoCTF easy tier** (short horizon) followed by the **filtered NL2Bash** micro-study.

.. _arXiv NYU: https://arxiv.org/abs/2406.05590?utm_source=chatgpt.com
.. _GitHub NYU: https://github.com/NYU-LLM-CTF/NYU_CTF_Bench
.. _PyPI NYU: https://pypi.org/project/nyuctf/?utm_source=chatgpt.com
.. _JusCodin's Blog CTFdPy: https://jus-codin.github.io/CTFdPy/endpoints/challenges/?utm_source=chatgpt.com
