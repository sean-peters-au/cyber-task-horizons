"""
CyberGym dataset preparer.

Transforms raw CyberGym data into standardized Run objects
representing human baseline performance, adhering to METR schema.

Note: CyberGym does not provide human timing data, so we use a fixed estimate.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from human_ttc_eval.core.prepare import Prepare
from human_ttc_eval.core.registry import register_preparer
from human_ttc_eval.core.run import Run
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_preparer("cybergym")
class CybergymPrepare(Prepare):
    """Prepares raw CyberGym data into standardized Run objects."""
    
    RAW_FILENAME = "cybergym_raw_data.jsonl"
    HUMAN_RUNS_FILENAME = "cybergym_human_runs.jsonl"
    
    def __init__(self):
        """Initialize the CyberGym preparer."""
        super().__init__(dataset_name="cybergym")
        self.default_raw_input_filename = self.RAW_FILENAME
        
        # Cache for raw data to avoid re-reading file in get_dataset_task_metadata
        self._raw_data_cache: Dict[str, Dict[str, Any]] = {}
    
    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        """
        Extract CyberGym specific metadata for task definitions.
        
        This metadata will be stored in the tasks.jsonl file and used
        by the benchmark harness to run challenges.
        
        Args:
            representative_run: A Run object for the task
            
        Returns:
            Dictionary with CyberGym specific metadata
        """
        task_id = representative_run.task_id
        
        # Use cached data if available
        raw_data: Optional[Dict[str, Any]] = None
        if task_id in self._raw_data_cache:
            raw_data = self._raw_data_cache[task_id]
        else:
            # Fallback to re-reading file if cache miss
            raw_data = self._find_task_in_raw_data(task_id)
        
        if not raw_data:
            logger.warning(f"Could not find raw data for task {task_id}")
            return {}
        
        # Build comprehensive task metadata for CyberGym
        task_metadata = {
            # Core task information
            "task_type": raw_data.get("task_type", "unknown"),
            "project_name": raw_data.get("project_name", "unknown"),
            "project_language": raw_data.get("project_language", "unknown"),
            "vulnerability_description": raw_data.get("vulnerability_description", ""),
            
            # Difficulty levels with associated files
            "task_difficulty": raw_data.get("task_difficulty", {}),
            "default_difficulty": config.CYBERGYM_DEFAULT_DIFFICULTY,
            
            # Additional metadata
            "project_homepage": raw_data.get("metadata", {}).get("project_homepage", ""),
            "project_main_repo": raw_data.get("metadata", {}).get("project_main_repo", ""),
            
            # Source information
            "source": "cybergym",
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
        Load raw CyberGym data and transform into Run objects.
        
        Note: CyberGym does not provide human timing data, so we use
        a fixed estimate from config.CYBERGYM_HUMAN_MINUTES_ESTIMATE.
        
        Returns:
            List of Run objects representing human baseline performance
        """
        logger.info("Starting CyberGym dataset preparation")
        
        raw_input_file = self.raw_data_dir / self.default_raw_input_filename
        if not raw_input_file.exists():
            logger.error(f"Raw CyberGym data file not found: {raw_input_file}")
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
                    task_type = raw_data.get("task_type", "unknown")
                    
                    if not task_id:
                        logger.warning(f"Skipping record in {raw_input_file} (line {line_num}) due to missing 'task_id'.")
                        continue
                    
                    # Cache raw data for later use in get_dataset_task_metadata
                    self._raw_data_cache[task_id] = raw_data
                    
                    # Use fixed estimate for human time (CyberGym has no human baseline data)
                    human_minutes = config.CYBERGYM_HUMAN_MINUTES_ESTIMATE
                    
                    # Create task family based on task type
                    task_family = f"cybergym_{task_type}"
                    
                    # Create Run object
                    run_obj = Run(
                        task_id=task_id,
                        task_family=task_family,
                        run_id=f"human_{task_id}_estimated",
                        alias="Human Baseline (CyberGym, estimated)",
                        model="human",
                        score_binarized=1,  # All human baselines assumed successful
                        score_cont=1.0,
                        human_minutes=human_minutes,
                        human_source="cybergym_fixed_estimate",
                        task_source="cybergym_dataset",
                        started_at=0.0,
                        completed_at=float(human_minutes * 60),
                        generation_cost=0.0,
                        fatal_error_from=None
                    )
                    
                    runs.append(run_obj)
            
            logger.info(f"Successfully prepared {len(runs)} runs from {raw_input_file}")
            
            # Log task type distribution
            type_counts: Dict[str, int] = {}
            for run in runs:
                task_type = run.task_family.replace("cybergym_", "")
                type_counts[task_type] = type_counts.get(task_type, 0) + 1
            logger.info(f"Task type distribution: {type_counts}")
            
        except IOError as e:
            logger.error(f"Error reading raw CyberGym data file {raw_input_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during CyberGym preparation: {e}", exc_info=True)
            raise
        
        return runs

