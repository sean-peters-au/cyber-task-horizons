"""
CyBashBench dataset preparer.

Transforms raw CyBashBench data into standardized Run objects representing
human baseline performance, adhering to METR schema. Since the source data
is already structured, this is mostly a pass-through operation.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from human_ttc_eval.core.prepare import Prepare
from human_ttc_eval.core.registry import register_preparer
from human_ttc_eval.core.run import Run

logger = logging.getLogger(__name__)


@register_preparer("cybashbench")
class CyBashBenchPrepare(Prepare):
    """
    Prepares raw CyBashBench data.

    Since the source file is already in a clean JSONL format with all necessary
    fields, this class primarily reads the file, converts each line into a Run
    object, and then lets the base class handle saving tasks and runs.
    """

    def __init__(self):
        """Initialize the CyBashBench preparer."""
        super().__init__(dataset_name="cybashbench")

    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        """
        Extracts the original dataset-specific metadata.
        
        This is stored in the `dataset_task_metadata` field of the raw task file.
        We just need to find the original raw task data and return that dictionary.
        """
        # In this implementation, all metadata is already loaded into the Run's `_raw_data`
        # attribute during prepare(), so we can just retrieve it from there.
        if hasattr(representative_run, '_raw_data') and 'dataset_task_metadata' in representative_run._raw_data:
            return representative_run._raw_data['dataset_task_metadata']
        
        logger.warning(f"Could not find raw metadata for task {representative_run.task_id}")
        return {}

    def prepare(self) -> List[Run]:
        """
        Load raw CyBashBench JSONL and transform into Run objects.
        
        Returns:
            List of Run objects representing human baseline performance.
        """
        logger.info("Starting CyBashBench dataset preparation")
        
        source_file = self.raw_data_dir / "cybashbench_tasks.jsonl"
        
        if not source_file.exists():
            logger.error(f"Raw data file not found: {source_file}")
            return []

        runs: List[Run] = []
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                raw_task_data = json.loads(line)
                
                # Convert the raw task line into a Run object
                run = self._task_to_run(raw_task_data)
                if run:
                    runs.append(run)

        logger.info(f"Created {len(runs)} Run objects from raw data")
        return runs

    def _task_to_run(self, raw_task: Dict[str, Any]) -> Optional[Run]:
        """Converts a raw task dictionary from the JSONL file to a Run object."""
        try:
            task_id = raw_task['task_id']
            task_family = raw_task['task_family']
            human_minutes = float(raw_task['human_minutes'])
            metadata = raw_task.get('dataset_task_metadata', {})
            timing_source = metadata.get('timing_source', 'heuristic')

            run = Run(
                task_id=task_id,
                task_family=task_family,
                run_id=f"human_{task_id.replace('/', '_')}",
                alias="Human Baseline (CyBashBench)",
                model="human",
                score_binarized=1,  # All baselines are successful
                score_cont=1.0,
                human_minutes=human_minutes,
                human_source=f"cybashbench_{timing_source}_estimates",
                task_source="cybashbench_dataset",
                started_at=0.0,
                completed_at=human_minutes * 60,
                generation_cost=0.0,
                fatal_error_from=None
            )
            
            # Store the original raw data on the object so `get_dataset_task_metadata` can access it
            run._raw_data = raw_task
            
            return run

        except (KeyError, ValueError) as e:
            logger.error(f"Error converting raw task to Run object: {e}. Data: {raw_task}")
            return None 