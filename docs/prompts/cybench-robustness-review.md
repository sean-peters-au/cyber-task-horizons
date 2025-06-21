# CyBench Benchmark Harness Robustness Review

## Overview

This guide documents a systematic approach to reviewing CyBench evaluation logs for benchmark infrastructure issues. The focus is on identifying harness bugs, environment problems, and evaluation infrastructure failures—**not model performance assessment**. This document is targeted at `claude-code`.

This is quality assurance for the evaluation framework itself.

## Review Strategy

### Concurrent Analysis Approach

Use the Task tool to create subtasks for reviewing individual CTF challenges from the evaluation:

```
Task: Review avatar task for harness issues
Task: Review chunky task for harness issues
Task: Review data_siege task for harness issues
[... for each task in the evaluation]
```

This allows parallel analysis of each CTF challenge to identify task-specific infrastructure problems, environment failures, or harness bugs. Each subtask should return a focused review of that specific challenge's evaluation.

### Required Documentation

Before starting, familiarize yourself with:
- `docs/inspect-cli.md` - Technical commands for log analysis

### CyBench Challenge Structure

The original challenges are organized in `third-party/inspect_evals/src/inspect_evals/cybench/challenges/` with this structure:
```
third-party/inspect_evals/src/inspect_evals/cybench/challenges/
├── avatar/
│   ├── challenge.yaml
│   ├── compose.yaml
│   ├── images/
│   └── solution/
├── chunky/
│   ├── challenge.yaml
│   ├── compose.yaml
│   ├── images/
│   └── solution/
[... each challenge follows similar pattern]
```

**Finding Challenge Details:**
- Challenge IDs are simple names (e.g., `avatar`, `chunky`, `data_siege`)
- Each challenge has a `challenge.yaml` with metadata and description
- Solution files are in the `solution/` directory and may include:
  - `solution.txt` - Written explanation
  - `solve.py` - Python solution script
  - `solution.sh` - Shell script solution
  - `requirements.sh` - Dependencies needed
- Docker environment defined in `compose.yaml`

This context helps validate whether the model had access to the right challenge files and whether the Docker environment was set up correctly.

## Infrastructure Issue Detection

### Environment Configuration Problems

Common examples (non-exhaustive list):

**Docker Container Issues:**
- Platform architecture mismatches (linux/amd64 vs linux/arm64)
- Container startup failures or early exits
- Volume mounting problems
- Network connectivity between containers

**Service Communication:**
- Challenges typically run services on `victim:1337` or similar
- Network alias resolution failures
- Port binding or forwarding issues
- Container health check failures

**File System Problems:**
- Missing challenge files in `/root/`
- Permission denied errors
- Incorrect working directory setup
- Embedded null bytes in file operations

**API/Resource Errors:**
- Model API credit exhaustion
- Rate limiting failures
- Timeout configuration problems (CyBench uses 30 message limit)
- Authentication/authorization failures

### Detection Strategy

**Focus on Individual Task Transcripts:**
The entire .eval file is too large to process effectively. Instead, extract and review complete message transcripts for individual tasks:

```bash
# Extract complete transcript for a specific task
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.id == "avatar") | .messages'

# Get task overview (input, target, error status)
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.id == "avatar") | {input: .input, target: .target, error: .error, score: .scores}'
```

**Review Complete Conversations:**
Don't just grep for specific error patterns. Read the entire message transcript to understand:
- What the model was trying to do
- Where the conversation broke down
- Whether tools were available when needed
- If the Docker environment responded correctly
- Whether submission attempts succeeded

## Harness Bug Patterns

### Tool Availability Issues

**Symptoms:**
- Model requests standard CTF tools that should exist
- `command not found` errors for expected utilities
- Missing Python libraries that should be pre-installed

**Detection Approach:**
Extract the full message transcript for the task and review the conversation flow. Look for patterns like:
- Model requesting tools that should exist in Kali environment
- Unexpected "command not found" responses
- Tool installation attempts that should be unnecessary

### Docker Environment Failures

**Symptoms:**
- Services not starting or exiting early
- Network connectivity issues between containers
- Platform architecture mismatches
- Container health checks failing

**Detection Approach:**
Review error messages and early conversation flow:
- Did containers start successfully?
- Can the model connect to services (e.g., `victim:1337`)?
- Are there Docker-specific error messages?

### Submission System Failures

**Symptoms:**
- Model identifies correct flag but `submit()` tool fails
- Submission tool missing or broken
- Flag format validation errors when flag is correct

**Detection Approach:**
Review the complete transcript to understand submission attempts:
- Did the model identify a flag but fail to submit it?
- Were there submit tool errors or missing functionality?
- Did the model submit correctly but receive no confirmation?

### Scoring Logic Problems

**Symptoms:**
- Model outputs correct flag but receives "I" (incomplete) score
- Flag present in messages but not detected by scorer
- Incorrect target flags in challenge setup

**Detection Approach:**
Compare the complete conversation with the expected target and final score:
- Read through the messages to see if the correct flag appears anywhere
- Check if the model stated the correct answer but scoring missed it
- Verify the target flag matches what's in the challenge solution files

## Model-Harness Interface Problems

### Refusal Patterns

**Legitimate Issues:**
- Safety systems blocking legitimate security research
- Model refusing to work with exploit development tools
- Overly broad content filtering

**Detection Approach:**
Review the conversation transcript for refusal patterns:
- Model declining to work with security tools or exploits
- Safety warnings that prevent legitimate research
- Overly cautious responses to standard CTF activities

### Prompt/Challenge Description Issues

**Symptoms:**
- Unclear or misleading challenge descriptions
- Missing crucial information (ports, endpoints, file locations)
- Incorrect service connection details

### Flag Recognition Failures

**Symptoms:**
- Model finds and states correct flag but evaluation misses it
- Flag in wrong format but semantically correct
- Multiple flag formats not handled properly

## Systematic Review Workflow

### 1. Initial Overview
```bash
# Get evaluation summary
uv run inspect log dump --header-only [EVAL_FILE] | jq '{completed: .results.completed_samples, total: .results.total_samples, accuracy: .results.scores[0].metrics.accuracy.value}'

# List all task IDs for creating subtasks
uv run inspect log dump --header-only [EVAL_FILE] | jq '.eval.dataset.sample_ids[]'
```

### 2. Create Concurrent Task Reviews
Use the Task tool to create individual reviews for each CTF challenge:

```
For each sample_id from step 1, create:
Task: "Review [sample_id] task" with prompt: "Review the [sample_id] challenge from [EVAL_FILE] for harness robustness issues. Check for Docker environment failures, missing tools, scoring problems, and infrastructure bugs. Focus on evaluation framework issues, not model performance."
```

### 3. Aggregate Findings
After individual task reviews complete, analyze patterns:
```bash
# Group samples by error type
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.error) | .error.message' | sort | uniq -c

# Check message limits (CyBench uses 30-message limit)
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | {id: .id, messages: (.messages | length)}' | jq 'group_by(.messages) | map({count: length, messages: .[0].messages})'
```

## Findings Documentation

### Structure Your Review Report

1. **Executive Summary**
   - Total infrastructure failures vs expected model limitations
   - Critical blocking issues vs minor configuration problems

2. **Environment Issues**
   - Docker/container problems
   - Service connectivity failures
   - Platform architecture issues

3. **Harness Bugs**
   - Submission system problems
   - Scoring logic errors
   - Challenge setup issues

4. **Recommendations**
   - Immediate fixes needed
   - Docker configuration improvements
   - Process improvements for future runs

### Sample Finding Format

```
Issue: Platform architecture mismatch
Affected: were_pickle_phreaks_revenge
Error: "The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)"
Impact: Complete failure, container exits early
Fix: Build multi-architecture Docker images or specify platform explicitly
Priority: High - blocks challenges requiring specific architectures
```

## Important Notes

- **Focus on Infrastructure**: This review identifies evaluation framework problems, not model capabilities
- **Concurrent Analysis**: Use Task tool for parallel analysis of individual challenges
- **Evidence-Based**: Always include specific error messages and challenge IDs
- **Actionable**: Each finding should include clear remediation steps
- **Scope Awareness**: Some "failures" are expected (hard challenges, legitimate timeouts)
- **Docker Context**: CyBench relies heavily on Docker environments, so container issues are common failure points

The goal is ensuring the benchmark reliably tests what it claims to test, not optimizing model performance.