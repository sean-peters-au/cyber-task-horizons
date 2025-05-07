.. cybersecurity-dataset-analysis documentation master file, created by
   sphinx-quickstart on Wed May  8 10:31:43 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: raw-html(raw)
    :format: html

.. default-role:: literal

Cyber-Horizon Benchmark
======================================

1 Context
---------

METR (2025), "Measuring AI Ability to Complete Long Tasks" timed humans on real software-engineering jobs and plotted horizon curves—% tasks solved as the time budget grows. GPT-4o matched or beat humans on work that takes humans < 1 h, showing steep scaling on harder jobs.

2 Our Goal
----------

Replicate METR's horizon-curve method purely in offensive-security tasks using open datasets—no new ranges, no private logs. This reveals when AI exploitation speed overtakes human experts and how far current models are from end-to-end, day-scale intrusions.

3 Dataset Stack (5 s → 24 h)
----------------------------

+------------+----------------------------------------------------------+----------------------------------------------------------------+--------------+
| Horizon    | Dataset slice                                            | Human HTC source                                               | Typical HTC  |
+============+==========================================================+================================================================+==============+
| Seconds    | Filtered NL2Bash – one-line coreutils commands         | Micro-study (volunteer typing)                                 | 5–30 s       |
+------------+----------------------------------------------------------+----------------------------------------------------------------+--------------+
| Minutes    | picoCTF easy (≤ 125 pts, 2019-25)                        | Median solve time from public scoreboards                      | 1–10 min     |
+------------+----------------------------------------------------------+----------------------------------------------------------------+--------------+
| 10 min–2 h | NYU-CTF finals (first-solve by a team with no prior solves) | Scraped CTFd logs                                              | 12–120 min   |
+------------+----------------------------------------------------------+----------------------------------------------------------------+--------------+
| 2–24 h     | CyBench – 40 pro-level exploits                          | Author-supplied first-solve times                              | 0.5–24 h     |
+------------+----------------------------------------------------------+----------------------------------------------------------------+--------------+

All tasks run in Docker, end with a deterministic flag checker, and regenerate secrets each run.

4 Why Cybersecurity Horizons Matter
-----------------------------------

Speed is leverage. A model that reduces exploit development from hours to minutes lets a low-skill actor launch complex attacks at machine scale, overwhelming defenders' capacity to monitor, triage, and patch. Tracking when AI agents cross human timing thresholds on real CTF tasks gives an early signal of capability-driven risk escalation—well before those skills appear in the wild or in black-box services.

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Benchmark Details:

   datasets

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 