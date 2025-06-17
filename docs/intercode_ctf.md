# InterCode-CTF Dataset

## Overview

InterCode-CTF provides 99 interactive CTF challenges (10s-10min) based on the InterCode framework. Tasks are PicoCTF-style challenges targeted at high school students using bash and Python in Docker environments.

**Paper**: [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback](https://arxiv.org/abs/2306.14898)  
**Source**: InterCode framework (MIT License)  
**Repository**: https://github.com/princeton-nlp/intercode

## Categories
- **Web Security** (25%): Basic HTTP analysis, directory traversal
- **Reverse Engineering** (20%): Binary analysis, string extraction  
- **Cryptography** (20%): Classical ciphers, encoding/decoding
- **Programming** (15%): Python scripting, text processing
- **Forensics** (10%): File metadata, log analysis
- **Network/Misc** (10%): Basic network tools, reconnaissance

## Human Baseline

**Source**: Cybench paper reports "average of 3.5 minutes" to solve tasks  
**Quality**: Single aggregate estimate with no methodology details, participant background, or task-specific breakdown  
**Reliability**: Lowest among all datasets - treat as rough directional indicator only

## Usage Notes

InterCode-CTF tests interactive problem-solving on educational-level CTF challenges. The human baseline has unknown reliability and should not be used for capability comparisons.

Best used for:
- Relative AI model comparison
- Testing interactive problem-solving ability
- Educational-level cybersecurity assessment