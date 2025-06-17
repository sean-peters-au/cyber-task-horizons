# Datasets Overview

## Dataset Portfolio

| Dataset | Tasks | Time Range | Source | Description | Scoring | Human Baseline |
|---------|-------|------------|---------|-------------|---------|----------------|
| **CyBashBench** | 131 | 0.6s-15s | Author-created | Cybersecurity command reflexes across 5 task types | LLM functional equivalence | AI-assisted (author-calibrated) |
| **NL2Bash** | 136 | 4s-4min | [TellinaTool corpus](https://arxiv.org/abs/1802.08979) | Natural language to bash translation, filtered atomic tasks | LLM functional equivalence | AI-assisted (anchored) |
| **InterCode-CTF** | 99 | 10s-10min | [InterCode framework](https://arxiv.org/abs/2306.14898) | Interactive CTF solving with bash/python | Flag discovery | AI-assisted (author-reviewed) |
| **NYUCTF** | 50 | 2min-6h | [NYU CTF Dataset](https://arxiv.org/abs/2406.05590) | CSAW-CTF challenges (2011-2023), dockerized | Flag discovery | AI-assisted (CyBench-anchored) |
| **CyBench** | 40 | 2min-25h | [Professional CTFs](https://arxiv.org/abs/2408.08926) | Recent high-level competitions, full docker environments | Flag discovery | Competition data (AI-validated) |

## Key Characteristics

### Human Baseline Reliability
1. **CyBashBench**: AI-assisted with extensive author calibration and review
2. **CyBench**: Competition data with AI validation and minimal adjustments  
3. **InterCode-CTF**: AI-assisted with author review for logical consistency
4. **NL2Bash**: AI-assisted with anchoring but limited author calibration
5. **NYUCTF**: AI-assisted with limited author expertise on advanced tasks

### Coverage Gaps
- **Sub-second reflexes**: Very limited coverage at the fastest end
- **Multi-day campaigns**: No representation of complex, persistent operations  
- **Defensive tasks**: Entirely focused on offensive cybersecurity
- **Non-technical skills**: Social engineering, OSINT, and strategic planning absent

## Systematic Biases

**AI Estimation Issues**:
- May underestimate expert speed (conservative bias)
- Lacks understanding of expert workflow optimization
- Could reflect training data patterns rather than human performance

**Competition Data Issues**:
- Strategic behavior (easier tasks first)
- Team coordination overhead
- Non-task time inclusion (breaks, discussions)