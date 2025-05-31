"""
METR Run dataclass and utilities.

This module defines the canonical Run schema used throughout the pipeline,
matching METR's all_runs.jsonl format exactly.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import json


@dataclass
class Run:
    """
    A single run (attempt) at a task, following METR's exact schema.
    
    A run represents one attempt to solve a task, either by a human baseline
    or an AI model. Multiple runs can exist for the same task.
    
    Field Calculation Notes:
    - human_minutes: Task-level property. For human runs, this is the actual time taken.
                    For AI runs, copy from the human baseline for that task.
    - human_cost: Calculated as human_minutes * cost_per_minute. We use $1.25/minute
                  based on $150k/year for an offensive security expert 
                  ($150k / 2080 hours / 60 minutes ≈ $1.20/minute, rounded to $1.25)
    - equal_task_weight: 1.0 / total_tasks_in_dataset. Gives equal weight to each task.
    - invsqrt_task_weight: 1.0 / sqrt(total_tasks_in_dataset). Down-weights large task families.
    - score_binarized: 1 if task succeeded, 0 if failed
    - score_cont: Continuous score if available, otherwise same as score_binarized
    - human_score: For human runs, same as score_cont. For AI runs, can be None.
    
    Fields we intentionally set to None/defaults:
    - time_limit: None (we don't enforce time limits currently)
    - started_at/completed_at: For human baselines, started_at is typically 0.0 (seconds from run epoch)
                               and completed_at reflects human_minutes in seconds.
                               For live AI or human runs, these would be actual epoch timestamps or run-relative seconds.
    - task_version: Default "1.0" (we don't track task versions)
    - generation_cost: 0.0 for humans, actual API costs for AI models
    """
    
    # Required fields (no defaults) - must come first
    task_id: str  # e.g., "nl2bash_simple/task_123"
    task_family: str  # e.g., "nl2bash_simple"
    run_id: str  # e.g., "human_nl2bash_simple_task_123_heuristic"
    alias: str  # Display name, e.g., "Human (NL2Bash)"
    model: str  # "human" for baselines, or "openai/gpt-4" etc
    score_binarized: int  # 0 or 1
    human_minutes: float  # Baseline time for the task (same across all runs)
    
    # Optional fields with defaults - must come after required fields
    score_cont: Optional[float] = None  # Continuous score if available
    fatal_error_from: Optional[str] = None  # Error message if run failed catastrophically
    human_score: Optional[float] = None  # Human performance score
    human_source: str = "estimate"  # How human time was determined
    task_source: str = ""  # Dataset name
    generation_cost: float = 0.0  # API costs for AI, 0 for humans
    human_cost: Optional[float] = None  # Dollar cost based on human_minutes
    time_limit: Optional[float] = None  # We don't use time limits
    started_at: Optional[float] = None  # E.g., seconds from start of this run, or epoch timestamp
    completed_at: Optional[float] = None  # E.g., seconds from start of this run, or epoch timestamp
    task_version: Optional[str] = "1.0"  # We use static version
    equal_task_weight: Optional[float] = None  # 1/n for n tasks
    invsqrt_task_weight: Optional[float] = None  # 1/sqrt(n) for n tasks
    
    # Constants
    HUMAN_COST_PER_MINUTE: float = 1.25  # Based on $150k/year expert
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Calculate human_cost if not provided
        if self.human_cost is None and self.human_minutes is not None:
            self.human_cost = self.human_minutes * self.HUMAN_COST_PER_MINUTE
            
        # Set score_cont to match score_binarized if not provided
        if self.score_cont is None:
            self.score_cont = float(self.score_binarized)
            
        # Set human_score to match score_cont for human runs
        if self.model == "human" and self.human_score is None:
            self.human_score = self.score_cont
    
    def validate(self) -> None:
        """
        Validate that all required fields are present and valid.
        
        Raises:
            ValueError: If validation fails
        """
        # Required string fields
        required_str_fields = ['task_id', 'task_family', 'run_id', 'alias', 'model', 
                              'human_source', 'task_source']
        for field in required_str_fields:
            value = getattr(self, field)
            if not value or not isinstance(value, str):
                raise ValueError(f"Field '{field}' must be a non-empty string, got: {value}")
        
        # Required numeric fields
        if self.score_binarized not in [0, 1]:
            raise ValueError(f"score_binarized must be 0 or 1, got: {self.score_binarized}")
            
        if self.human_minutes is None or self.human_minutes <= 0:
            raise ValueError(f"human_minutes must be positive, got: {self.human_minutes}")
            
        if self.generation_cost is None or self.generation_cost < 0:
            raise ValueError(f"generation_cost must be non-negative, got: {self.generation_cost}")
        
        # Validate score_cont is between 0 and 1 if provided
        if self.score_cont is not None:
            if not 0 <= self.score_cont <= 1:
                raise ValueError(f"score_cont must be between 0 and 1, got: {self.score_cont}")
    
    def to_jsonl_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSONL output, excluding None values.
        
        Returns:
            Dict with all non-None fields
        """
        data = asdict(self)
        # Remove None values to match METR format
        return {k: v for k, v in data.items() if v is not None and k != 'HUMAN_COST_PER_MINUTE'}
    
    def to_jsonl_line(self) -> str:
        """
        Convert to a JSONL line.
        
        Returns:
            JSON string with newline
        """
        return json.dumps(self.to_jsonl_dict()) + '\n'
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Run':
        """
        Create a Run instance from a dictionary.
        
        Args:
            data: Dictionary with run data
            
        Returns:
            Run instance
        """
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)
    
    @staticmethod
    def calculate_weights(runs: List['Run']) -> None:
        """
        Calculate and apply task weights (equal_task_weight and invsqrt_task_weight)
        to a list of Run objects in-place.

        Args:
            runs: A list of Run objects. Weights will be added/updated on these objects.
        """
        if not runs:
            return

        unique_task_ids = {run.task_id for run in runs}
        total_unique_tasks = len(unique_task_ids)

        if total_unique_tasks <= 0:
            # Should not happen if runs is not empty, but as a safeguard.
            for run in runs:
                run.equal_task_weight = 0.0
                run.invsqrt_task_weight = 0.0
            return

        # Calculate equal_task_weight
        equal_weight = 1.0 / total_unique_tasks
        for run in runs:
            run.equal_task_weight = equal_weight

        # Calculate invsqrt_task_weight (based on task_family size)
        tasks_per_family: Dict[str, int] = {}
        for run in runs:
            pass # Initial pass to gather unique tasks per family
        
        # To correctly calculate invsqrt_task_weight, we first need counts of *unique tasks* per family.
        family_task_counts: Dict[str, int] = {}
        tasks_in_family: Dict[str, set] = {}

        for run in runs:
            if run.task_family not in tasks_in_family:
                tasks_in_family[run.task_family] = set()
            tasks_in_family[run.task_family].add(run.task_id)
        
        for family, task_set in tasks_in_family.items():
            family_task_counts[family] = len(task_set)

        # Now assign invsqrt_task_weight
        for run in runs:
            if run.task_family and run.task_family in family_task_counts:
                num_tasks_in_family = family_task_counts[run.task_family]
                if num_tasks_in_family > 0:
                    run.invsqrt_task_weight = 1.0 / (num_tasks_in_family ** 0.5)
                else:
                    run.invsqrt_task_weight = 0.0 # Should not happen if family exists
            else:
                run.invsqrt_task_weight = 0.0 # Task family not found or empty
    
    @staticmethod
    def filter_unique_tasks(runs: List['Run']) -> List[str]:
        """
        Get unique task IDs from a list of runs.
        
        Args:
            runs: List of Run instances
            
        Returns:
            List of unique task_ids
        """
        return list({run.task_id for run in runs})
    
    @staticmethod
    def load_jsonl(filepath: str) -> List['Run']:
        """
        Load runs from a JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            List of Run instances
        """
        runs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    runs.append(Run.from_dict(data))
        return runs
    
    @staticmethod
    def save_jsonl(runs: List['Run'], filepath: str) -> None:
        """
        Save runs to a JSONL file.
        
        Args:
            runs: List of Run instances
            filepath: Path to save JSONL file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for run in runs:
                f.write(run.to_jsonl_line()) 