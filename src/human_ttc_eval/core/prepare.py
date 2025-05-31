"""
Base class for dataset preparers.

Preparers transform raw data from data/raw/<dataset>/ into standardized
Run objects that strictly conform to the METR all_runs.jsonl schema.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import logging

from .run import Run
from .task import Task
from .. import config

logger = logging.getLogger(__name__)


class Prepare(ABC):
    """
    Abstract base class for dataset preparation (formerly parsing).

    Preparers transform raw data (from data/raw/<dataset_name>/) into a list of 
    Run objects adhering to the METR schema. These are then saved to 
    data/processed/<dataset_name>/<dataset_name>_human_runs.jsonl.
    Additionally, a <dataset_name>_tasks.jsonl file is created with task-level definitions.
    """
    
    def __init__(self, dataset_name: str):
        """
        Initialize the preparer.

        Args:
            dataset_name: The unique identifier for the dataset.
                          This name is used to determine input (raw) and output (processed) 
                          directories.
        """
        self.dataset_name = dataset_name
        self.raw_data_dir = config.DATA_DIR / "raw" / self.dataset_name
        self.processed_data_dir = config.DATA_DIR / "processed" / self.dataset_name
        self.default_human_runs_filename = f"{self.dataset_name}_human_runs.jsonl"
        self.default_tasks_filename = f"{self.dataset_name}_tasks.jsonl"
        self.default_raw_input_filename = f"{self.dataset_name}_raw.jsonl"

        if not self.raw_data_dir.exists():
            logger.warning(f"Raw data directory {self.raw_data_dir} does not exist. Preparation might fail if raw data is required.")

        try:
            self.processed_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized {self.__class__.__name__} for dataset '{self.dataset_name}'. "
                        f"Raw data expected in: {self.raw_data_dir}, Processed data will be saved to: {self.processed_data_dir}")
        except OSError as e:
            logger.error(f"Error creating processed data directory {self.processed_data_dir}: {e}")
            raise
    
    @abstractmethod
    def prepare(self) -> List[Run]:
        """
        Load raw data, transform it into Run objects, and return the list.

        Subclasses must implement this method to read from `self.raw_data_dir`
        (e.g., `self.raw_data_dir / self.default_raw_input_filename` or other files)
        and produce a list of Run objects. These Run objects should represent human attempts.
        
        Returns:
            A list of Run objects representing the prepared human data.
        """
        pass

    @abstractmethod
    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        """
        Extracts dataset-specific metadata for a given task.

        This method is called when creating the <dataset_name>_tasks.jsonl file.
        It should use the provided 'representative_run' (which is one of the Run objects
        for a specific task_id) to extract any additional metadata relevant at the task level
        for this particular dataset.

        Args:
            representative_run: A Run object that corresponds to the task for which
                                metadata is being extracted.

        Returns:
            A dictionary containing dataset-specific task metadata.
            Return an empty dictionary if no dataset-specific metadata is applicable.
        """
        pass

    def save_runs(self, runs: List[Run], output_filename: str = None) -> Path:
        """
        Save a list of human Run objects to a JSONL file after validation and weight calculation.
        The weights are calculated and applied to the Run objects in-place.

        Args:
            runs: A list of Run objects to save (these should be human runs).
            output_filename: Optional name for the output file. 
                             Defaults to f"{self.dataset_name}_human_runs.jsonl".

        Returns:
            Path to the saved JSONL file.
        """
        if not runs:
            logger.warning(f"No runs provided to save for dataset '{self.dataset_name}'.")

        output_file_name_to_use = output_filename or self.default_human_runs_filename
        output_path = self.processed_data_dir / output_file_name_to_use

        valid_runs: List[Run] = []
        for i, run_obj in enumerate(runs):
            if run_obj.model != "human":
                logger.warning(f"Run {i+1} (ID: {run_obj.run_id}) for dataset '{self.dataset_name}' is not a human run (model='{run_obj.model}'). It will be excluded from the human runs file.")
                continue
            try:
                run_obj.validate() 
                valid_runs.append(run_obj)
            except ValueError as e:
                logger.error(f"Human Run {i+1} (ID: {run_obj.run_id}) for dataset '{self.dataset_name}' failed validation and will be skipped: {e}")
        
        if not valid_runs:
            logger.warning(f"No valid human runs to save for '{self.dataset_name}' to {output_path}.")
        
        # Calculate and apply weights in-place to the valid Run objects
        Run.calculate_weights(valid_runs) 

        try:
            Run.save_jsonl(valid_runs, str(output_path))
            logger.info(f"Successfully saved {len(valid_runs)} processed human runs to {output_path}")
            return output_path
        except IOError as e:
            logger.error(f"Error writing processed human runs to {output_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving human runs to {output_path}: {e}")
            raise

    def save_tasks(self, runs_for_tasks: List[Run], output_filename: str = None) -> Path:
        """
        Creates and saves a list of Task objects derived from a list of Run objects.
        One Task object is created for each unique task_id found in the runs.
        Weights are taken from the provided Run objects (ensure save_runs was called first).

        Args:
            runs_for_tasks: A list of Run objects (typically human runs that have had weights calculated).
            output_filename: Optional name for the tasks output file.
                             Defaults to f"{self.dataset_name}_tasks.jsonl".
        Returns:
            Path to the saved tasks JSONL file.
        """
        if not runs_for_tasks:
            logger.warning(f"No runs provided to derive tasks for dataset '{self.dataset_name}'.")
        
        output_file_name_to_use = output_filename or self.default_tasks_filename
        output_path = self.processed_data_dir / output_file_name_to_use

        unique_tasks_dict: Dict[str, Run] = {}
        for run_obj in runs_for_tasks:
            if run_obj.task_id not in unique_tasks_dict:
                unique_tasks_dict[run_obj.task_id] = run_obj

        task_objects: List[Task] = []
        for _, representative_run in unique_tasks_dict.items():
            task_metadata = self.get_dataset_task_metadata(representative_run)
            task = Task(
                task_id=representative_run.task_id,
                task_family=representative_run.task_family,
                human_minutes=representative_run.human_minutes,
                equal_task_weight=representative_run.equal_task_weight,
                invsqrt_task_weight=representative_run.invsqrt_task_weight,
                dataset_task_metadata=task_metadata
            )
            task_objects.append(task)
            
        if not task_objects:
            logger.warning(f"No unique tasks derived from runs for '{self.dataset_name}'. Tasks file at {output_path} may be empty or not created.")

        try:
            Task.save_jsonl(task_objects, str(output_path))
            logger.info(f"Successfully saved {len(task_objects)} unique tasks to {output_path}")
            return output_path
        except IOError as e:
            logger.error(f"Error writing tasks to {output_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving tasks to {output_path}: {e}")
            raise

    def run(self) -> List[Run]:
        """
        Orchestrates the preparation and saving process for both human runs and task definitions.
        
        Returns:
            The list of prepared (and validated) human Run objects.
        """
        logger.info(f"Starting preparation for dataset: {self.dataset_name}...")
        prepared_runs = self.prepare() # Should yield human runs
        
        if prepared_runs is None: # prepare() might return None if it handles saving or errors
            logger.warning(f"Prepare method for '{self.dataset_name}' returned None. No runs to process further.")
            prepared_runs = []
        
        if not prepared_runs:
             logger.warning(f"Preparation yielded no runs for '{self.dataset_name}'. Output files may be empty.")
        else:
            logger.info(f"Preparation yielded {len(prepared_runs)} runs for '{self.dataset_name}'.")

        # Save human runs (this also calculates and assigns weights to prepared_runs in-place)
        self.save_runs(prepared_runs)
        
        # Save task definitions using the (potentially weight-populated) runs
        self.save_tasks(prepared_runs)
        
        logger.info(f"Preparation and saving complete for {self.dataset_name}.")
        return prepared_runs