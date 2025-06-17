# NL2Bash Dataset

## Overview

NL2Bash provides 136 natural language to bash translation tasks (4s-4min) based on the TellinaTool corpus. Filtered from 12,607 to 4,247 atomic tasks by removing pipes, redirects, and complex chaining.

**Paper**: [NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System](https://arxiv.org/abs/1802.08979)  
**Source**: https://github.com/TellinaTool/nl2bash (GPL-3.0)

## Characteristics

**Utilities Covered**: ls, ps, find, grep, awk, sed, sort, netstat, chmod, tar, etc.  
**Security Relevance**: Reconnaissance, log analysis, system monitoring, file operations  
**Complexity**: Simple commands (10-30s) to complex operations with awk/sed (2-5min)

## Human Baseline

**Source**: Two-phase estimation (heuristic complexity scoring + Claude 4 Opus refinement)  
**Quality**: AI-assisted with limited empirical validation - substantial uncertainty  
**Cross-validation**: Compared against similar CyBashBench tasks where possible

## Usage Notes

NL2Bash tests fundamental bash translation skills. Human baselines are AI estimates with no empirical validation. Best used for relative model comparison rather than absolute capability assessment.