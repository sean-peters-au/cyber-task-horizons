# NYUCTF Benchmark Harness Robustness Review

## Overview

This guide documents a systematic approach to reviewing NYUCTF evaluation logs for benchmark infrastructure issues. The focus is on identifying harness bugs, environment problems, and evaluation infrastructure failures—**not model performance assessment**. This document is targeted at `claude-code`.

This is quality assurance for the evaluation framework itself.

## Review Strategy

### Concurrent Analysis Approach

Use the Task tool to create subtasks for reviewing individual CTF challenges from the evaluation:

```
Task: Review 2017f-pwn-humm_sch_t task for harness issues
Task: Review 2023q-web-rainbow_notes task for harness issues
Task: Review 2016q-cry-sleeping_guard task for harness issues
[... for each task in the evaluation]
```

This allows parallel analysis of each CTF challenge to identify task-specific infrastructure problems, environment failures, or harness bugs. Each subtask should return a focused review of that specific challenge's evaluation.

### Required Documentation

Before starting, familiarize yourself with:
- `docs/inspect-cli.md` - Technical commands for log analysis

### NYU CTF Challenge Structure

The original challenges are organized in `third-party/NYU_CTF_Bench/test` with this structure:
```
third-party/NYU_CTF_Bench/test/
├── 2017/
│   ├── CSAW-Finals/
│   └── CSAW-Quals/
├── 2018/
│   ├── CSAW-Finals/
│   └── CSAW-Quals/
[... through 2023]
```

**Finding Challenge Details:**
- Challenge IDs follow pattern: `{year}{q/f}-{category}-{name}` (e.g., `2017f-pwn-humm_sch_t`)
- Use `find` or `grep` to locate specific challenges: `find third-party/NYU_CTF_Bench/test -name "*humm*" -type d`
- Look for solution files, writeups, or flag files within challenge directories
- Not all challenges may have the same file structure - adapt your search approach

This context helps validate whether the model had access to the right challenge files and whether the environment was set up correctly.

## Infrastructure Issue Detection

### Environment Configuration Problems

Common examples (non-exhaustive list):

**Docker Container Issues:**
- Missing seccomp profiles (e.g., `chrome.json` not found)
- Container startup failures
- Volume mounting problems
- Network isolation issues

**Tool/Binary Availability:**
- Expected tools missing from Kali environment
- Broken tool installations or PATH issues
- Missing Python packages or dependencies

**File System Problems:**
- Permission denied errors
- Missing challenge files
- Incorrect working directory setup

**Network Connectivity:**
- Service endpoints unreachable
- Port binding failures
- DNS resolution problems

**API/Resource Errors:**
- Model API credit exhaustion
- Rate limiting failures
- Timeout configuration problems
- Authentication/authorization failures

**IMPORTANT - Message Limits Are NOT Infrastructure Issues:**
- NYUCTF uses a 40-message limit as a cost control mechanism
- Models hitting this limit are typically wandering aimlessly, not converging on solutions
- This is the expected failure mode for models that cannot solve the challenge
- **Only report message limit issues if there's strong evidence the model was converging on a correct solution before being cut off**

### Detection Strategy

**Focus on Individual Task Transcripts:**
The entire .eval file is too large to process effectively. Instead, extract and review complete message transcripts for individual tasks:

```bash
# Extract complete transcript for a specific task
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.id == "2017f-pwn-humm_sch_t") | .messages'

# Get task overview (input, target, error status)
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.id == "2017f-pwn-humm_sch_t") | {input: .input, target: .target, error: .error, score: .scores}'
```

**Review Complete Conversations:**
Don't just grep for specific error patterns. Read the entire message transcript to understand:
- What the model was trying to do
- Where the conversation broke down
- Whether tools were available when needed
- If the environment responded correctly
- Whether submission attempts succeeded

**Evaluating Message Limit Cases:**
For samples that hit the 40-message limit:
- Look for evidence of productive progress toward the solution
- Check if the model was systematically exploring viable approaches
- Distinguish between "almost solved" vs "wandering aimlessly"
- Only flag as infrastructure issue if clear convergence was interrupted

## Harness Bug Patterns

### Tool Availability Issues

**Symptoms:**
- Model requests standard CTF tools that should exist
- `command not found` errors for expected utilities
- Missing Python libraries that should be pre-installed

**Detection Approach:**
Extract the full message transcript for the task and review the conversation flow. Look for patterns like:
- Model requesting tools that should exist
- Unexpected "command not found" responses
- Tool installation attempts that should be unnecessary

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
- Verify the target flag matches what the model should have found

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
- Incorrect network service information

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
Task: "Review [sample_id] task" with prompt: "Review the [sample_id] challenge from [EVAL_FILE] for harness robustness issues. Check for environment failures, missing tools, scoring problems, and infrastructure bugs. Focus on evaluation framework issues, not model performance."
```

### 3. Aggregate Findings
After individual task reviews complete, analyze patterns:
```bash
# Group samples by error type
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | select(.error) | .error.message' | sort | uniq -c

# Check message limits (potential infrastructure constraint)
uv run inspect log dump [EVAL_FILE] | jq '.samples[] | {id: .id, messages: (.messages | length)}' | jq 'group_by(.messages) | map({count: length, messages: .[0].messages})'
```

## Findings Documentation

### Structure Your Review Report

1. **Executive Summary**
   - Total infrastructure failures vs expected model limitations
   - Critical blocking issues vs minor configuration problems

2. **Environment Issues**
   - Docker/container problems
   - Missing tools or dependencies
   - Network connectivity failures

3. **Harness Bugs**
   - Submission system problems
   - Scoring logic errors
   - Challenge setup issues

4. **Recommendations**
   - Immediate fixes needed
   - Configuration improvements
   - Process improvements for future runs

### Sample Finding Format

```
Issue: Missing seccomp profile for Chrome-based challenges
Affected: 2023q-web-rainbow_notes  
Error: "opening seccomp profile (chrome.json) failed: no such file or directory"
Impact: Complete failure, 0% completion rate for affected challenges
Fix: Add chrome.json seccomp profile to docker environment
Priority: High - blocks entire challenge category
```

## Important Notes

- **Focus on Infrastructure**: This review identifies evaluation framework problems, not model capabilities
- **Concurrent Analysis**: Use Task tool for parallel analysis of individual challenges
- **Evidence-Based**: Always include specific error messages and sample IDs
- **Actionable**: Each finding should include clear remediation steps
- **Scope Awareness**: Some "failures" are expected (hard challenges, legitimate timeouts)
- **Message Limits Are Normal**: 84% hitting the 40-message limit is the expected failure mode, not an infrastructure problem

**Critical Distinction**: Distinguish between:
- **Infrastructure failures**: Environment setup issues, missing tools, container problems
- **Resource constraints**: Message limits, timeouts (these are working as intended)
- **Model limitations**: Inability to solve challenges within constraints (expected)

The goal is ensuring the benchmark reliably tests what it claims to test, not optimizing model performance.