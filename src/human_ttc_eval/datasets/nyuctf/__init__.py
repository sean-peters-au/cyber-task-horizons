"""
NYUCTF dataset implementation for human-ttc-eval.

This module provides the complete pipeline for processing and evaluating
CSAW CTF challenges from the NYU_CTF_Bench repository.
"""

# Import all components to register them with the system
from . import nyuctf_retrieve
from . import nyuctf_prepare  
from . import nyuctf_describe
from . import nyuctf_bench

__all__ = [
    'nyuctf_retrieve',
    'nyuctf_prepare', 
    'nyuctf_describe',
    'nyuctf_bench'
]