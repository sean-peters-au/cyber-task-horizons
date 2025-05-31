"""
Prepares CyBench dataset by transforming raw data into standardized Run objects
representing human baseline performance, adhering to METR schema.
"""
import logging
import json
from typing import List, Dict, Any

from human_ttc_eval.core.prepare import Prepare
from human_ttc_eval.core.registry import register_preparer
from human_ttc_eval.core.run import Run

logger = logging.getLogger(__name__)

@register_preparer("cybench")
class CybenchPrepare(Prepare):
    """Prepares raw CyBench data into standardized Run objects."""

    RAW_FILENAME = "cybench_raw_data.jsonl"

    def __init__(self):
        """
        Initializes the CybenchPrepare class.
        """
        super().__init__(dataset_name="cybench")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.default_raw_input_filename = self.RAW_FILENAME 

    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        """
        CyBench does not have additional dataset-specific task metadata to extract.
        Returns an empty dictionary.
        """
        return {}

    def prepare(self) -> List[Run]:
        """
        Loads raw CyBench data, transforms it, and creates Run objects.
        Raw data is expected at self.raw_data_dir / f"{self.dataset_name}_raw.jsonl".
        """
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

                    task_id = raw_data.get("task_id")
                    solve_time_seconds = raw_data.get("fastest_solve_time_seconds")
                    name = raw_data.get("name", "Unknown Task")
                    category = raw_data.get("category", "unknown")
                    event = raw_data.get("event", "unknown_event")
                    ctf_name = raw_data.get("ctf_name", "unknown_ctf")
                    readme_data = raw_data.get("readme_data", {})
                    difficulty = readme_data.get("difficulty", "unknown")

                    if not task_id:
                        logger.warning(f"Skipping record in {raw_input_file} (line {line_num}) due to missing 'task_id'.")
                        continue
                    
                    if solve_time_seconds is None:
                         # METR schema requires completed_at, and for human runs, it's duration.
                         # If solve time is missing, we cannot create a valid human Run object.
                        logger.warning(f"Skipping task '{task_id}' due to missing 'fastest_solve_time_seconds'.")
                        continue
                    
                    try:
                        human_minutes = float(solve_time_seconds) / 60.0
                    except (TypeError, ValueError):
                        logger.warning(f"Skipping task '{task_id}' due to invalid format for 'fastest_solve_time_seconds': {solve_time_seconds}")
                        continue

                    # Construct task_family based on the second component of task_id
                    task_id_parts = task_id.split('/')
                    if len(task_id_parts) > 1:
                        task_family = task_id_parts[1]
                    else:
                        task_family = self.dataset_name # Fallback to dataset name if task_id has no '/'

                    # Human baseline runs should have model="human"
                    run_obj = Run(
                        task_id=task_id,
                        task_family=task_family,
                        run_id=f"human_{task_id.replace('/ ', '_')}",
                        alias="Human Baseline (CyBench)",
                        model="human",
                        score_binarized=1,
                        score_cont=1.0,
                        human_minutes=human_minutes,
                        human_source=self.dataset_name,
                        task_source=self.dataset_name,
                        started_at=0.0,
                        completed_at=float(solve_time_seconds)
                    )
                    runs.append(run_obj)

            logger.info(f"Successfully prepared {len(runs)} runs from {raw_input_file}")

        except IOError as e:
            logger.error(f"Error reading raw CyBench data file {raw_input_file}: {e}")
            return [] # Return empty list on read error
        except Exception as e:
            logger.error(f"An unexpected error occurred during CyBench preparation: {e}", exc_info=True)
            return []
            
        return runs
