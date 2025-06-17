# NYUCTF Dataset

## Overview

NYUCTF provides 50 university-level CTF challenges (2min-6h) from CSAW-CTF competitions (2011-2023). Based on the NYU CTF Dataset (NeurIPS 2024).

**Paper**: [NYU CTF Bench: A Scalable Open-Source Benchmark Dataset for Evaluating LLMs in Offensive Security](https://arxiv.org/abs/2406.05590)  
**Source**: https://github.com/NYU-LLM-CTF/NYU_CTF_Bench (GPL-2.0)

## Categories
- **Web Exploitation**: SQL injection, XSS, CSRF, authentication bypass
- **Binary Pwn**: Buffer overflows, ROP chains, heap exploitation  
- **Reverse Engineering**: Binary analysis, decompilation
- **Forensics**: Memory dumps, network captures, file carving
- **Cryptography**: Classical and modern cipher breaking
- **Miscellaneous**: Steganography, programming challenges

## Human Baseline

**Source**: AI-assisted estimates anchored on CyBench competition times  
**Process**: Claude 4 Opus cross-referenced NYUCTF challenges with comparable CyBench tasks  
**Quality**: Limited author expertise on advanced tasks - substantial uncertainty

## Usage Notes

NYUCTF challenges use AI-assisted timing estimates with limited validation. While based on real university competition problems, the human baselines have substantial uncertainty due to the estimation methodology.