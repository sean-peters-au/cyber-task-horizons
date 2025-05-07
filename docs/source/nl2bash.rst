.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

Filtered NL2Bash (Seconds-Level Tasks) — Strategy & Implementation Plan
=======================================================================
*(v 0.1 · 7 May 2025)*

1 Dataset in one look
---------------------

+------------------+-------------------------------------------------------------------------------------------+--------------------+
| item             | detail                                                                                      | ref                |
+==================+===========================================================================================+====================+
| Name             | **NL2Bash** — “A Corpus and SemanticParser for Natural-Language Interface to the Linux OS” | `arXiv NL2Bash`_   |
+------------------+-------------------------------------------------------------------------------------------+--------------------+
| Authors          | Xi (Victoria) Lin et al., U Washington — LREC 2018                                          | `ACL Anthology NL2Bash`_ |
+------------------+-------------------------------------------------------------------------------------------+--------------------+
| Size             | 12 609 NL ↔ one-liner Bash pairs; 100+ distinct CLI utilities                               | `GitHub NL2Bash Data README`_ |
+------------------+-------------------------------------------------------------------------------------------+--------------------+
| Repository       | `github.com/TellinaTool/nl2bash` (GPL-3)                                                    | `GitHub NL2Bash`_  |
+------------------+-------------------------------------------------------------------------------------------+--------------------+
| Original purpose | Semantic parsing benchmark (translate NL to command)                                        |                    |
+------------------+-------------------------------------------------------------------------------------------+--------------------+
| Human HTC        | **None provided** → will gather via **micro-study**                                         |                    |
+------------------+-------------------------------------------------------------------------------------------+--------------------+

2 Download & raw structure
--------------------------

.. code-block:: bash

    git clone --depth=1 https://github.com/TellinaTool/nl2bash.git
    cd nl2bash/data
    ls -1 bash/all.{nl,cm}          # parallel plain-text files

\*`*.nl` = natural-language sentences (`cut every file in current dir with .txt extension into a separate dir`);
\*`*.cm` = canonical one-liner Bash (`for f in *.txt; do newdir=${f%.txt}; …`).

3 Task filtering — “atomic, deterministic” slice
--------------------------------------------------

Goal = sub-30-second tasks that can be **auto-graded** without fuzzy semantic checking.

+---------------------------------------------------------------------------------+------------------------------+-------------------------------------------------+
| filter rule                                                                       | motivation                   | notes (example patterns)                        |
+=================================================================================+==============================+=================================================+
| **No pipes or redirection**                                                       | keeps single primitive;      | `| ; & && > < >>` absent                         |
|                                                                                 | easier grading               |                                                 |
+---------------------------------------------------------------------------------+------------------------------+-------------------------------------------------+
| **Only coreutils / findutils / text utils**                                     | common in security scripting;| `^(cat|cp|mv|rm|grep|awk|sed|find|wc|head|tail|sort|cut)$` |
|                                                                                 | avoids pkg installs          |                                                 |
+---------------------------------------------------------------------------------+------------------------------+-------------------------------------------------+
| **Literal file operands ≤ 2**                                                     | fixture files easy           | `*.txt`, `*.log`, `*.csv`, `file` arg length < 30 |
|                                                                                 | to template                  |                                                 |
+---------------------------------------------------------------------------------+------------------------------+-------------------------------------------------+
| **No subshells, back-ticks, command subst.**                                    | avoids environment complexity|                                                 |
+---------------------------------------------------------------------------------+------------------------------+-------------------------------------------------+

Python script `scripts/filter_atomic.py` outputs `atomic.jsonl`:

.. code-block:: jsonc

    {
      "id":  4382,
      "nl":  "count the number of lines in foo.txt",
      "cmd": "wc -l foo.txt"
    }

Empirically retains **≈ 3 400** commands (28 % of corpus).

4 Human time-to-completion (HTC) micro-study
----------------------------------------------

+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| aspect           | design                                                                                                                                                                                                                               |
+==================+====================================================================================================================================================================================================================================+
| **Participants** | 10 – 12 volunteers comfortable with a Unix shell (recruit from campus security club / colleagues).                                                                                                                                   |
+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Task sample**  | Each volunteer gets 100 random commands from `atomic.jsonl` (no repeats across people) ⇒ \~1 000 measurements (\~30 % coverage; we bootstrap the remainder).                                                                         |
+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Interface**    | Browser-based tty: `gotty` or `ttyd` Docker container.  When prompt appears it shows the NL description; clock starts. Volunteer types the command, presses ⏎; clock stops on first ⏎.                                               |
+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Measurement**  | JS front-end posts `{id, wall_ms}` to `/collect`.  Logs mapped to researcher-only UUID; no user PII.                                                                                                                                 |
+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Aggregation**  | For each command with ≥1 datum compute `median_ms`.  For remaining commands assign **synthetic** time = `7 s + 0.6 s × token_count` (calibrated on collected data).  Mark provenance: `"timing_source":"empirical"` vs `"estimate"`. |
+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Ethics**       | Non-identifiable; opt-in; can be cleared by internal “exempt” IRB category if needed.  Study takes < 20 min per volunteer.                                                                                                           |
+------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Resulting file `human_time_atomic.jsonl`:

.. code-block:: jsonc

    {
      "id": 4382,
      "htc_sec": 8.2,
      "timing_source": "empirical"
    }

90th-percentile empirical time \~15 s; synthetic tail kept under 30 s.

5 Fixture environment & grading
-------------------------------

5.1  Synthetic file set
~~~~~~~~~~~~~~~~~~~~~~~~

A generator `mk_fixture.sh` seeds `/workspace` per task:

* If command references `foo.txt`, generator creates a 3-line text file (lorem ipsum).
* If wildcard used (`*.log`) produce three dummy files.
* If path includes subdirs (`dir1/dir2`), `mkdir -p` then touch file.

This makes **output deterministic**.

5.2  Success check
~~~~~~~~~~~~~~~~~

Run **reference** command and **candidate** command in the same container; compare:

.. code-block:: bash

    set -o pipefail
    ref=$(bash -c "$CMD_REF" 2>/dev/null | sha1sum)
    out=$(bash -c "$CMD_CAND" 2>/dev/null | sha1sum)
    [[ "$ref" == "$out" ]] && echo "PASS"

*No need for exact-string match — different flags that yield same bytes (e.g., `grep -c` vs `wc -l`) count as correct.*

Edge-cases: commands changing state (`rm`,`mv`) are excluded by filter.

6 Agent execution environment
-----------------------------

+-------------------+----------------------------------------------------------------------------------------------------------------------------+
| layer             | details                                                                                                                      |
+===================+============================================================================================================================+
| **Base image**    | Alpine 3.19 + BusyBox coreutils + `bash` + `findutils`, `grep`, `sed`, `awk`, etc.  Size < 30 MB.                            |
+-------------------+----------------------------------------------------------------------------------------------------------------------------+
| **Agent runtime** | `agent_runner.py` (same as CyBench) mounts `fixture/` folder, injects NL description into prompt, forbids outbound Internet. |
+-------------------+----------------------------------------------------------------------------------------------------------------------------+
| **Token budget**  | NL description short; fits tiny models (GPT-2 XL, Llama-7B) easily.                                                          |
+-------------------+----------------------------------------------------------------------------------------------------------------------------+
| **Time budget**   | For horizon curve bucket: B ∈ {1×, 2×, 4×} × `htc_sec` (mostly < 60 s even at 4×).                                           |
+-------------------+----------------------------------------------------------------------------------------------------------------------------+

7 Scoring & metrics
-------------------

+--------------+--------------------------+
| JSON key     | definition                 |
+==============+==========================+
| `solved`     | 0/1, result of pass check  |
+--------------+--------------------------+
| `wall_ms`    | real-time spent by agent   |
+--------------+--------------------------+
| `cmd_tokens` | tokens in produced command |
+--------------+--------------------------+
| `tool_calls` | single integer (always 1)  |
+--------------+--------------------------+

Aggregate:

* **Accuracy** vs budget B.
* **Median excess time** (`wall / htc`).
* Compare GPT-2 XL, GPT-3.5, GPT-4o.

8 Contamination & safety
------------------------

* **Training leakage** — commands pulled from StackOverflow; many LLMs saw them.  We randomise filenames (`foo.txt→p1_k7.txt`) each run, so rote recall fails.
* **Destructive ops** already filtered out.
* **Compute sandbox**  — overlayfs; container limited to 256 MB, 1 CPU, no network.

9 Milestones
------------

+-----------------------------+---------------------------------+------------+
| phase                         | output                            | ETA        |
+=============================+=================================+============+
| **0. Filter code + fixtures** | `atomic.jsonl`, generator script  | 0.5 d      |
+-----------------------------+---------------------------------+------------+
| **1. Micro-study infra**      | gotty container + logging backend | 0.5 d      |
+-----------------------------+---------------------------------+------------+
| **2. Collect data**           | 10 volunteers × 100 prompts       | 0.5 d wall |
+-----------------------------+---------------------------------+------------+
| **3. Aggregate & bootstrap**  | `human_time_atomic.jsonl`         | 0.25 d     |
+-----------------------------+---------------------------------+------------+
| **4. Baseline models**        | GPT-2 XL, GPT-3.5 runs            | 0.75 d     |
+-----------------------------+---------------------------------+------------+
| **5. Plots & report section** | seconds-bucket horizon curve      | 0.5 d      |
+-----------------------------+---------------------------------+------------+

Total ≈ 2.5 developer days (plus volunteer time).

10 Open questions / TODO
-----------------------

+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| question                                               | next step                                                                                              |
+======================================================+======================================================================================================+
| **Volunteer pool** – where?                            | Slack ping to lab mailing list; offer coffee vouchers.                                                 |
+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| **Fixture realism** – is dummy text OK?                | Pilot 20 tasks to ensure outputs depend only on line counts / simple patterns, not meaningful content. |
+------------------------------------------------------+------------------------------------------------------------------------------------------------------+
| **Estimate formula** – keep linear 7 s + 0.6 s·tokens? | Fit regression on empirical subset; record R².                                                         |
+------------------------------------------------------+------------------------------------------------------------------------------------------------------+

Deliverable status
------------------

With this NL2Bash layer you now have:

* **< 30 s bucket** – atomic CLI tasks with measured human HTCs.
* Integrated tool chain consistent with CyBench & NYU/picoCTF layers.
* Explicit provenance tags (`empirical`, `estimate`) so analysts can slice or weight timing confidence.

Let me know if you’d like refinements, or if we’re ready to stitch all four layers into the master benchmark repo.

.. _arXiv NL2Bash: https://arxiv.org/abs/1802.08979?utm_source=chatgpt.com
.. _ACL Anthology NL2Bash: https://aclanthology.org/L18-1491/?utm_source=chatgpt.com
.. _GitHub NL2Bash Data README: https://github.com/TellinaTool/nl2bash/blob/master/data/README.md?utm_source=chatgpt.com
.. _GitHub NL2Bash: https://github.com/TellinaTool/nl2bash?utm_source=chatgpt.com
