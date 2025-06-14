"""
CyBench dataset preparer.

Transforms raw CyBench data from inspect_evals into standardized Run objects
representing human baseline performance, adhering to METR schema.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from human_ttc_eval.core.prepare import Prepare
from human_ttc_eval.core.registry import register_preparer
from human_ttc_eval.core.run import Run

logger = logging.getLogger(__name__)


@register_preparer("cybench")
class CybenchPrepare(Prepare):
    """Prepares raw CyBench data into standardized Run objects."""
    
    RAW_FILENAME = "cybench_raw_data.jsonl"
    HUMAN_RUNS_FILENAME = "cybench_human_runs.jsonl"
    DEFAULT_VARIANT = "hard"  # Default to hard variant for benchmarking
    
    def __init__(self):
        """Initialize the CyBench preparer."""
        super().__init__(dataset_name="cybench")
        self.default_raw_input_filename = self.RAW_FILENAME
        
        # Cache for raw data to avoid re-reading file in get_dataset_task_metadata
        self._raw_data_cache: Dict[str, Dict[str, Any]] = {}
        self.human_time_estimates: Optional[Dict[str, float]] = None
    
    def _load_human_time_estimates(self) -> Dict[str, float]:
        """
        Loads human time estimates from the manually curated JSONL file.
        This file is expected to be in data/keep/cybench/.
        """
        if self.human_time_estimates is not None:
            return self.human_time_estimates

        # The raw_data_dir is usually 'data/raw'. We navigate to 'data/keep'.
        keep_dir = self.raw_data_dir.parent.parent / "keep" / self.dataset_name
        estimates_path = keep_dir / self.HUMAN_RUNS_FILENAME
        
        if not estimates_path.exists():
            raise FileNotFoundError(
                f"Human estimates file not found at expected path: {estimates_path}"
            )
        
        estimates: Dict[str, float] = {}
        with open(estimates_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                task_id = data.get("task_id")
                time_sec = data.get("estimated_time_seconds")
                if task_id is not None and time_sec is not None:
                    estimates[task_id] = float(time_sec)
        
        logger.info(f"Loaded {len(estimates)} human time estimates from {estimates_path}")
        self.human_time_estimates = estimates
        return self.human_time_estimates
    
    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        """
        Extract CyBench specific metadata for task definitions.
        
        This metadata will be stored in the tasks.jsonl file and used
        by the benchmark harness to run challenges.
        
        Args:
            representative_run: A Run object for the task
            
        Returns:
            Dictionary with CyBench specific metadata including challenge
            configuration, prompts, and validation information
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
        
        # Extract variant data (default to "hard" variant)
        variants = raw_data.get("variants", {})
        variant_data = variants.get(self.DEFAULT_VARIANT, {})
        
        # If hard variant doesn't exist, try to find another variant
        if not variant_data and variants:
            # Priority order: hard, easy, first available
            for variant_name in ["hard", "easy"]:
                if variant_name in variants:
                    variant_data = variants[variant_name]
                    logger.info(f"Using {variant_name} variant for task {task_id} (hard not available)")
                    break
            else:
                # Use first available variant
                first_variant = next(iter(variants.keys()))
                variant_data = variants[first_variant]
                logger.info(f"Using {first_variant} variant for task {task_id} (no standard variant available)")
        
        # Build comprehensive task metadata
        task_metadata = {
            # Core challenge information
            "name": raw_data.get("name", task_id),
            "flag": raw_data.get("flag", ""),
            "category": raw_data.get("category", "unknown"),
            "challenge_dir": raw_data.get("challenge_dir", ""),
            
            # Variant-specific data
            "prompt": variant_data.get("prompt", ""),
            "variant_files": variant_data.get("files", {}),
            "variant_metadata": variant_data.get("metadata", {}),
            
            # Challenge-level files
            "challenge_files": raw_data.get("files", {}),
            
            # Competition metadata
            "competition": raw_data.get("metadata", {}).get("competition", ""),
            "first_solve_time_minutes": raw_data.get("first_solve_time_minutes", 0),
            
            # Available variants (for potential future use)
            "available_variants": list(variants.keys()),
            "selected_variant": self.DEFAULT_VARIANT if self.DEFAULT_VARIANT in variants else (list(variants.keys())[0] if variants else "none")
        }
        
        return task_metadata
    
    def _find_task_in_raw_data(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a specific task in the raw data file.
        
        Args:
            task_id: The task ID to search for
            
        Returns:
            Raw data dictionary for the task, or None if not found
        """
        raw_input_file = self.raw_data_dir / self.default_raw_input_filename
        if not raw_input_file.exists():
            return None
        
        try:
            with open(raw_input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data = json.loads(line.strip())
                    if raw_data.get("task_id") == task_id:
                        return raw_data
        except Exception as e:
            logger.error(f"Error searching for task {task_id} in raw data: {e}")
        
        return None
    
    def prepare(self) -> List[Run]:
        """
        Load raw CyBench data and transform into Run objects.
        
        Returns:
            List of Run objects representing human baseline performance
        """
        logger.info("Starting CyBench dataset preparation")
        
        human_time_estimates = self._load_human_time_estimates()

        raw_input_file = self.raw_data_dir / self.default_raw_input_filename
        if not raw_input_file.exists():
            logger.error(f"Raw CyBench data file not found: {raw_input_file}")
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
                    name = raw_data.get("name", "")
                    category = raw_data.get("category", "unknown")
                    
                    if not task_id:
                        logger.warning(f"Skipping record in {raw_input_file} (line {line_num}) due to missing 'task_id'.")
                        continue

                    if task_id not in human_time_estimates:
                        raise ValueError(
                            f"Task ID '{task_id}' from raw data not found in "
                            f"human estimates file ({self.HUMAN_RUNS_FILENAME}). "
                            "All tasks must have a time estimate."
                        )
                    
                    solve_time_seconds = human_time_estimates[task_id]
                    timing_source = "human_expert_median"
                    
                    # Cache raw data for later use in get_dataset_task_metadata
                    self._raw_data_cache[task_id] = raw_data
                    
                    # Convert time to minutes
                    try:
                        human_minutes = float(solve_time_seconds) / 60.0
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid time format for task '{task_id}': {solve_time_seconds}")
                        continue
                    
                    # Create task family based on category
                    task_family = f"cybench_{category}"
                    
                    # Create Run object
                    run_obj = Run(
                        task_id=task_id,  # Simple name like "avatar"
                        task_family=task_family,  # e.g., "cybench_web"
                        run_id=f"human_{task_id}_{timing_source}",
                        alias="Human Baseline (CyBench)",
                        model="human",
                        score_binarized=1,  # All human baselines assumed successful
                        score_cont=1.0,
                        human_minutes=human_minutes,
                        human_source=f"cybench_{timing_source}",
                        task_source="cybench_dataset",
                        started_at=0.0,
                        completed_at=float(solve_time_seconds),
                        generation_cost=0.0,
                        fatal_error_from=None
                    )
                    
                    runs.append(run_obj)
            
            logger.info(f"Successfully prepared {len(runs)} runs from {raw_input_file}")
            
            # Log category distribution
            category_counts = {}
            for run in runs:
                category = run.task_family.replace("cybench_", "")
                category_counts[category] = category_counts.get(category, 0) + 1
            logger.info(f"Category distribution: {category_counts}")
            
        except IOError as e:
            logger.error(f"Error reading raw CyBench data file {raw_input_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during CyBench preparation: {e}", exc_info=True)
            return []
        
        return runs
