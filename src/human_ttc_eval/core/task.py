"""
Dataclass for representing a Task, including task-specific metadata.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """
    Represents a single task with its core properties and dataset-specific metadata.
    """
    task_id: str
    task_family: str
    human_minutes: float  # Task-level human completion time

    # Weights are typically calculated based on a collection of tasks
    equal_task_weight: Optional[float] = None
    invsqrt_task_weight: Optional[float] = None
    
    # For dataset-specific fields not part of the core schema
    dataset_task_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """Converts the Task object to a dictionary suitable for JSONL, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def save_jsonl(cls, tasks: List['Task'], file_path: Path | str) -> None:
        """
        Saves a list of Task objects to a JSONL file.
        Each task is written as a new line in JSON format.
        """
        file_path_obj = Path(file_path)
        try:
            with open(file_path_obj, 'w') as f:
                for task in tasks:
                    f.write(json.dumps(task.to_jsonl_dict()) + '\n')
            logger.info(f"Successfully saved {len(tasks)} tasks to {file_path_obj}")
        except IOError as e:
            logger.error(f"Error writing tasks to {file_path_obj}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving tasks to {file_path_obj}: {e}")
            raise

    @classmethod
    def load_jsonl(cls, file_path: Path | str) -> List['Task']:
        """
        Loads a list of Task objects from a JSONL file.
        Assumes each line is a JSON representation of a Task.
        """
        file_path_obj = Path(file_path)
        tasks: List[Task] = []
        try:
            with open(file_path_obj, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        logger.debug(f"Skipping empty line {line_num} in {file_path_obj}")
                        continue
                    try:
                        data = json.loads(line)
                        tasks.append(cls(**data))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON on line {line_num} in {file_path_obj}: {e}. Line: '{line}'")
                    except TypeError as e: # Handles missing fields or type mismatches during instantiation
                        logger.error(f"Error creating Task object from data on line {line_num} in {file_path_obj}: {e}. Data: '{data}'")
            logger.info(f"Successfully loaded {len(tasks)} tasks from {file_path_obj}")
            return tasks
        except FileNotFoundError:
            logger.error(f"Task file not found: {file_path_obj}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading tasks from {file_path_obj}: {e}")
            raise
