"""
NYUCTF dataset preparer.

Transforms raw NYUCTF data into standardized Run objects
representing human baseline performance, adhering to METR schema.
"""

import json
import logging
import random
from typing import List, Dict, Any

from human_ttc_eval.core.prepare import Prepare
from human_ttc_eval.core.registry import register_preparer
from human_ttc_eval.core.run import Run
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_preparer("nyuctf")
class NyuctfPrepare(Prepare):
    """Prepares raw NYUCTF data into standardized Run objects."""
    
    RAW_FILENAME = "nyuctf_raw_data.jsonl"
    HUMAN_RUNS_FILENAME = "nyuctf_human_runs.jsonl"
    
    def __init__(self):
        """Initialize the NYUCTF preparer."""
        super().__init__(dataset_name="nyuctf")
        self.default_raw_input_filename = self.RAW_FILENAME
        
        # Cache for raw data to avoid re-reading file in get_dataset_task_metadata
        self._raw_data_cache: Dict[str, Dict[str, Any]] = {}
        self.human_time_estimates: Dict[str, float] = {}
    
    def _load_human_time_estimates(self) -> Dict[str, float]:
        """
        Loads human time estimates from the manually curated JSONL file.
        This file is expected to be in data/keep/nyuctf/.
        """
        if self.human_time_estimates:
            return self.human_time_estimates

        # The raw_data_dir is usually 'data/raw'. We navigate to 'data/keep'.
        keep_dir = self.raw_data_dir.parent.parent / "keep" / self.dataset_name
        estimates_path = keep_dir / self.HUMAN_RUNS_FILENAME
        
        estimates: Dict[str, float] = {}
        if estimates_path.exists():
            with open(estimates_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    task_id = data.get("task_id")
                    time_sec = data.get("estimated_time_seconds")
                    if task_id is not None and time_sec is not None:
                        estimates[task_id] = float(time_sec)
            logger.info(f"Loaded {len(estimates)} human time estimates from {estimates_path}")
        else:
            logger.warning(f"Human estimates file not found at {estimates_path}. Proceeding with defaults.")

        self.human_time_estimates = estimates
        return self.human_time_estimates
    
    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        """
        Extract NYUCTF specific metadata for task definitions.
        
        This metadata will be stored in the tasks.jsonl file and used
        by the benchmark harness to run challenges.
        
        Args:
            representative_run: A Run object for the task
            
        Returns:
            Dictionary with NYUCTF specific metadata including challenge
            configuration, Docker setup, and validation information
        """
        task_id = representative_run.task_id
        
        # Use cached data if available
        if task_id in self._raw_data_cache:
            raw_data = self._raw_data_cache[task_id]
        else:
            # Fallback to re-reading file if cache miss
            raw_data = self._find_task_in_raw_data(task_id)
            if not raw_data:
                logger.warning(f"Could not find raw data for task {task_id}")
                return {}
        
        # Build comprehensive task metadata
        task_metadata = {
            # Core challenge information
            "name": raw_data.get("name", task_id),
            "flag": raw_data.get("flag", ""),
            "category": raw_data.get("category", "unknown"),
            "year": raw_data.get("year", ""),
            "event": raw_data.get("event", ""),
            "challenge": raw_data.get("challenge", ""),
            "description": raw_data.get("description", ""),
            "author": raw_data.get("author", ""),
            "points": raw_data.get("points", 0),
            
            # Challenge files and setup
            "files": raw_data.get("files", []),
            "challenge_directory": raw_data.get("challenge_directory", ""),
            "path": raw_data.get("path", ""),
            
            # Docker and deployment information
            "has_docker_compose": raw_data.get("has_docker_compose", False),
            "docker_compose_path": raw_data.get("docker_compose_path"),
            "challenge_type": raw_data.get("challenge_type", "static"),
            "internal_port": raw_data.get("internal_port"),
            "box": raw_data.get("box"),
            
            # Reference and metadata
            "reference": raw_data.get("reference", ""),
            "split": raw_data.get("split", "test"),
            
            # Full challenge info for benchmark runner
            "challenge_info": raw_data.get("challenge_info", {}),
            
            # Timing information
            "timing_source": raw_data.get("timing_source", "default_estimate"),
        }
        
        return task_metadata
    
    def _find_task_in_raw_data(self, task_id: str) -> Dict[str, Any]:
        """
        Find a specific task in the raw data file.
        
        Args:
            task_id: The task ID to search for
            
        Returns:
            Raw data dictionary for the task, or None if not found
        """
        raw_input_file = self.raw_data_dir / self.default_raw_input_filename
        if not raw_input_file.exists():
            return {}
        
        try:
            with open(raw_input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data = json.loads(line.strip())
                    if raw_data.get("task_id") == task_id:
                        return raw_data
        except Exception as e:
            logger.error(f"Error searching for task {task_id} in raw data: {e}")
        
        return {}
    
    def prepare(self) -> List[Run]:
        """
        Load raw NYUCTF data and transform into Run objects.
        
        Returns:
            List of Run objects representing human baseline performance
        """
        logger.info("Starting NYUCTF dataset preparation")
        
        human_time_estimates = self._load_human_time_estimates()
        # Use the task IDs from the human runs file as the ground truth set
        ground_truth_task_ids = set(human_time_estimates.keys())
        logger.info(f"Using fixed set of {len(ground_truth_task_ids)} tasks from the human runs file.")
        
        raw_input_file = self.raw_data_dir / self.default_raw_input_filename
        if not raw_input_file.exists():
            logger.error(f"Raw NYUCTF data file not found: {raw_input_file}")
            return []
        
        runs: List[Run] = []
        
        try:
            with open(raw_input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        raw_data = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON line {line_num} in {raw_input_file}: {e}")
                        continue
                    
                    # Extract fields
                    task_id = raw_data.get("task_id")

                    # Only include tasks that are in our curated ground truth set
                    if task_id not in ground_truth_task_ids:
                        continue
                    
                    name = raw_data.get("name", "")
                    category = raw_data.get("category", "unknown")
                    year = raw_data.get("year", "")
                    event = raw_data.get("event", "")
                    
                    # Get timing data from our estimates, falling back to the original default
                    if task_id in human_time_estimates:
                        solve_time_seconds = human_time_estimates[task_id]
                        timing_source = "human_expert_median"
                    else:
                        solve_time_seconds = raw_data.get("human_minutes", 10.0) * 60
                        timing_source = "default_estimate"
                    
                    if not task_id:
                        logger.warning(f"Skipping record in {raw_input_file} (line {line_num}) due to missing 'task_id'.")
                        continue
                    
                    if solve_time_seconds is None or solve_time_seconds <= 0:
                        logger.warning(f"Invalid timing for task '{task_id}', using default 10 minutes")
                        solve_time_seconds = 600.0
                    
                    # Cache raw data for later use in get_dataset_task_metadata
                    self._raw_data_cache[task_id] = raw_data
                    
                    # Create task family based on category and event
                    task_family = f"nyuctf_{category}"
                    
                    # Create unique run ID
                    event_short = "f" if "Finals" in event else "q"  # Finals or Quals
                    run_id = f"human_{year}{event_short}_{category}_{raw_data.get('challenge', task_id)}_{timing_source}"
                    
                    # Create Run object
                    run_obj = Run(
                        task_id=task_id,  # e.g., "2021f-rev-maze"
                        task_family=task_family,  # e.g., "nyuctf_rev"
                        run_id=run_id,
                        alias="Human Baseline (NYUCTF)",
                        model="human",
                        score_binarized=1,  # All human baselines assumed successful
                        score_cont=1.0,
                        human_minutes=solve_time_seconds / 60.0,
                        human_source=f"nyuctf_{timing_source}",
                        task_source="nyuctf_dataset",
                        started_at=0.0,
                        completed_at=float(solve_time_seconds),
                        generation_cost=0.0,
                        fatal_error_from=None
                    )
                    
                    runs.append(run_obj)
            
            logger.info(f"Successfully prepared {len(runs)} runs from {raw_input_file}")
            
            # Log category and year distribution
            category_counts = {}
            year_counts = {}
            for run in runs:
                category = run.task_family.replace("nyuctf_", "")
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Extract year from task_id (e.g., "2021f-rev-maze" -> "2021")
                year = run.task_id.split('-')[0][:4] if '-' in run.task_id else "unknown"
                year_counts[year] = year_counts.get(year, 0) + 1
            
            logger.info(f"Category distribution: {category_counts}")
            logger.info(f"Year distribution: {year_counts}")
            
        except IOError as e:
            logger.error(f"Error reading raw NYUCTF data file {raw_input_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during NYUCTF preparation: {e}", exc_info=True)
            return []
        
        # Sampling is no longer needed as we use a fixed set from human_runs.jsonl
        return runs