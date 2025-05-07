.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

Benchmark Datasets Overview
===========================

This document provides an overview of the four datasets that constitute the cyber-horizon benchmark. It explains what each dataset is, how human time-to-completion (HTC) is determined, the segment of the time–difficulty spectrum it covers, and key considerations for building a unified evaluation harness.

.. toctree::
   :maxdepth: 1
   :caption: Individual Dataset Strategies:

   cybench
   nyuctf
   picoctf
   nl2bash

1 Why these four together?
--------------------------

+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+
| Seconds                                                                                               | Minutes                                                                                                                       | Tens of minutes → Couple hours                                                                                                       | Hour → Day                                                                           |
+=======================================================================================================+===============================================================================================================================+======================================================================================================================================+======================================================================================+
| **Filtered NL2Bash** – atomic one-line CLI tasks with empirical or boot-strapped human times (5 – 30 s) | **picoCTF easy tier** – 75/100/125-pt "General Skills / Web / Crypto / Forensics" challenges, median human solve ≈ 90 s – 8 min | **NYU-CTF Bench** (zero-prior first-blood filter) – mid-difficulty CSAW finals problems; strict upper-bound expert HTCs ≈ 12 min – 2 h | **CyBench** – 40 pro CTF exploits; author-supplied first-solve times run 30 min → 24 h |
+-------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+

Stacking them yields one contiguous time axis, entirely within a single **offensive-security** theme—mirroring the METR horizon curves but without re-using any METR datasets.

2 Dataset capsules
------------------

+---------------------------------+----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Dataset                         | Scope & Task Types                                                                                                       | #Tasks kept                                                | Human-time signal                                                                                                                                                                                   | Licence / Access                                                  | Key Nuances                                                                                                                                                |
+=================================+============================================================================================================================+============================================================+=====================================================================================================================================================================================================+===================================================================+============================================================================================================================================================+
| **CyBench (2024)**              | 40 hard CTF exploits from Hack The Box, SekaiCTF, GlacierCTF, HKCERT (web, pwn, rev, crypto). Dockerised; optional subtasks. | 40 (all)                                                     | `fst_sec` – first-solve time scraped by authors and shipped in repo (2 min – 24 h, median ≈ 62 min) (`CyBench Website`_, `Stanford CRFM CyBench`_)                                                   | Apache-2.0 · `git clone https://github.com/andyzorigin/cybench.git` | Clean upper-bounds already validated; subtasks allow guided runs; nested Docker needs privileged host.                                                         |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **NYU-CTF Bench (2024)**        | Full CSAW finals archive 2014-23; 6 categories; each challenge a Docker bundle.                                              | ≈ 45–60 after "**first_solve ≈ team's first solve**" filter | `htc_sec = t_first_solve` for teams with **zero earlier solves** (strict upper bound). Data pulled via public CTFd `/solves` API. (`arXiv NYU CTF`_, `NYU-LLM-CTF GitHub.io`_)                     | GPL-2.0 · `github.com/NYU-LLM-CTF/NYU_CTF_Bench`                    | Must scrape scoreboards once per contest; filter culls many tasks but leaves a solid 10 min – 2 h band with reliable HTCs.                                     |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **picoCTF easy tier (2019-25)** | ≤ 125-pt "General Skills / Web / Forensics / Crypto" challenges; mostly static files or tiny services.                       | ≈ 150                                                        | `median_sec` across all solves (hundreds–thousands per task) from CTFd API; smooths out speed-runners. (`GitHub picoCTF Platform`_)                                                                | BSD-like; requires free picoCTF account & cookie for API calls      | Tasks genuinely short; API still online after each contest; regenerate flag + shuffle filenames to avoid memorisation.                                         |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Filtered NL2Bash (2018)**     | One-line coreutils/find/grep/awk commands with deterministic output.                                                         | ≈ 3 400 after filters                                        | **Micro-study median** (10 volunteers × 100 prompts) for 30 % sample → linear model assigns 7 s + 0.6 s·tokens to rest; entries tagged `empirical` vs `estimate`. (`GitHub TellinaTool NL2Bash`_, `Victoria Lin NL2Bash PDF`_) | GPL-3 · `github.com/TellinaTool/nl2bash`                            | Provide fixture generator (dummy files) + SHA-1 output grading so any synonym command counts; seconds-level timing differentiates GPT-2 / GPT-J from random. |
+---------------------------------+----------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+

3 What each dataset **adds** to the picture
-------------------------------------------

+-------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| Dataset     | Gap it fills                                                              | Strengths for benchmarking                                                                                                        | Watch-outs                                                                                                               |
+=============+=========================================================================+=====================================================================================================================================+==========================================================================================================================+
| **CyBench** | Long-horizon exploits (> 1 h) that require deep binary/web skill.         | Ready-made human upper bounds; subtasks for diagnostic runs; identical interface to other CTF corpora.                                | Nested Docker on Mac is painful; a few tasks > 20 h may need 4× budget cap.                                                |
+-------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| **NYU-CTF** | Mid-range realistic exploits (tens of minutes) in same domain as CyBench. | Zero-prior first-solve filter provides near-pure work-time; big pool of tasks across years.                                           | Scraper must handle login captchas for some old years; sparse timing on a few very hard tasks—tag `timing_quality=sparse`. |
+-------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| **picoCTF** | Short, authentic security puzzles that humans crack in minutes.           | Thousands of solve timestamps ⇒ robust HTCs; tasks emphasise basic tool usage (curl, grep, steghide) perfect for tool-augmented LMs.    | Need session cookie for API; some 2019-20 challenges ship vulnerable old Node stacks—build images offline to stay safe.      |
+-------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+
| **NL2Bash** | Atomic CLI actions in the **5 – 30 s** bracket (typing+think time).       | Deterministic grading; easy for humans yet differentiates weak vs. strong models; not CTF but still security-adjacent shell literacy. | Micro-study required; many commands overlap StackOverflow so random file-name shuffling is vital to avoid recall wins.       |
+-------------+-------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------+

4 Shared evaluation scaffold
----------------------------

* **One agent harness** (`agent_runner.py`) wraps all four datasets:

  * Launches the dataset-supplied Docker compose.
  * Drops model into `/workspace` BusyBox shell.
  * Limits wall-time to {1×, 2×, 4×} HTC and counts tool invocations.
* **Success detectors**

  * CTF sets: run their bundled checker scripts (expect `FLAG{}` pattern).
  * NL2Bash: run candidate & reference commands, compare SHA-1 of output.
* **Timing provenance field** in every metadata row (`fst`, `median`, `empirical`, `estimate`, `sparse`) lets analysts decide how to weight each point when drawing METR-style horizon curves.

5 Why this mix solves the earlier problems
------------------------------------------

* **Clear binary success** – every task ends with an automated checker or byte-match; no heuristic parsing of logs required (the issue that killed the KYPO attempt).
* **Human time ≥ upper bound** – CyBench ships validated FST; NYU filter eliminates order bias; picoCTF median is conservative; NL2Bash micro-study measures directly.
* **Full time bandwidth** – seconds → day scale, all in cybersecurity shell context, so model comparisons are apples-to-apples across horizons.
* **Open access** – none of the four datasets require commercial licences; only picoCTF scraping needs a free account cookie.

6 Next-step checklist
---------------------

+-----+------------------------------------------------------------------------+
| ✔︎   | task                                                                     |
+=====+========================================================================+
| ☐   | Finalise NYU scoreboard scraper; produce `human_time.jsonl`.             |
+-----+------------------------------------------------------------------------+
| ☐   | Run NL2Bash volunteer study; fit linear tail model.                      |
+-----+------------------------------------------------------------------------+
| ☐   | Consolidate all metadata into `datasets/index.csv` with provenance tags. |
+-----+------------------------------------------------------------------------+
| ☐   | Smoke-test harness on one task per dataset with GPT-4o.                  |
+-----+------------------------------------------------------------------------+
| ☐   | Draft horizon-curve notebook template.                                   |
+-----+------------------------------------------------------------------------+

Once these are done you can replicate the METR "time-to-completion" plots—now in a **single, security-focused benchmark** that spans 5 s to 24 h and cleanly separates weak, mid, and frontier models.

.. _CyBench Website: https://cybench.github.io/?utm_source=chatgpt.com
.. _Stanford CRFM CyBench: https://crfm.stanford.edu/2024/08/19/cybench.html?utm_source=chatgpt.com
.. _arXiv NYU CTF: https://arxiv.org/abs/2406.05590?utm_source=chatgpt.com
.. _NYU-LLM-CTF GitHub.io: https://nyu-llm-ctf.github.io/?utm_source=chatgpt.com
.. _GitHub picoCTF Platform: https://github.com/picoCTF/picoCTF?utm_source=chatgpt.com
.. _GitHub TellinaTool NL2Bash: https://github.com/TellinaTool/nl2bash?utm_source=chatgpt.com
.. _Victoria Lin NL2Bash PDF: https://victorialin.org/pubs/nl2bash.pdf?utm_source=chatgpt.com
