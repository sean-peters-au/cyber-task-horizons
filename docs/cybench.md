# CyBench Dataset

## Overview

CyBench provides 40 professional CTF challenges (2min-25h) from recent high-level competitions including Hack The Box Cyber Apocalypse 2024, SekaiCTF 2024, GlacierCTF 2023, and HKCERT CTF 2023.

**Paper**: [CyBench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models](https://arxiv.org/abs/2408.08926)

## Categories
- **Web Exploitation** (30%): SQL injection, deserialization, authentication bypass
- **Binary Exploitation** (25%): Heap exploitation, ROP/JOP chains, kernel exploits  
- **Reverse Engineering** (20%): Malware analysis, algorithm reconstruction
- **Cryptography** (15%): Implementation attacks, protocol breaking
- **Miscellaneous** (10%): Blockchain, cloud, hardware security

## Human Baseline

**Source**: Competition first-solve times with AI validation  
**Issue**: Competition timing includes multi-tasking, strategic ordering, team coordination, and breaks - not isolated task time  
**Quality**: Competition data mostly preserved after AI review for basic reasonableness

## Usage Notes

CyBench timing data has substantial contamination from competition context. Best used for:
- Relative AI model comparison
- Success rate analysis rather than timing comparison
- Demonstrating AI can solve professional-level challenges