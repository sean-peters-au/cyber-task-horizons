"""
CyberGym dataset retriever.

Retrieves task metadata from the CyberGym HuggingFace dataset.
The HuggingFace dataset contains task definitions including vulnerability descriptions
and difficulty levels.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)

# Subset task IDs for the 10-task subset
SUBSET_TASK_IDS = [
    "arvo:47101",
    "arvo:3938",
    "arvo:24993",
    "arvo:1065",
    "arvo:10400",
    "arvo:368",
    "oss-fuzz:42535201",
    "oss-fuzz:42535468",
    "oss-fuzz:370689421",
    "oss-fuzz:385167047",
]


@register_retriever("cybergym")
class CybergymRetrieve(Retrieve):
    """
    Retrieves CyberGym task metadata from the HuggingFace dataset.
    
    Reads from the cloned HuggingFace dataset and extracts task definitions
    for the subset of 10 tasks.
    """
    
    DEFAULT_OUTPUT_FILENAME = "cybergym_raw_data.jsonl"
    
    def __init__(self, dataset_name: str = "cybergym", output_filename: Optional[str] = None):
        """
        Initialize the CyberGym retriever.
        
        Args:
            dataset_name: Name of the dataset (default: "cybergym")
            output_filename: Optional custom name for the output JSONL file.
        """
        super().__init__(dataset_name)
        self.output_filename = output_filename or self.DEFAULT_OUTPUT_FILENAME
        self.data_dir = config.CYBERGYM_DATA_PATH
    
    def _verify_data_available(self) -> bool:
        """Verify that the CyberGym HuggingFace data is available."""
        if not self.data_dir.exists():
            logger.error(f"CyberGym data directory not found: {self.data_dir}")
            logger.error("Run 'make cybergym-setup' to download the dataset")
            return False
        
        tasks_file = self.data_dir / "tasks.json"
        if not tasks_file.exists():
            logger.error(f"CyberGym tasks.json not found: {tasks_file}")
            return False
        
        return True
    
    def _load_tasks_from_huggingface(self) -> List[Dict[str, Any]]:
        """
        Load task definitions from the HuggingFace dataset.
        
        Returns:
            List of task dictionaries
        """
        tasks_file = self.data_dir / "tasks.json"
        
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                all_tasks = json.load(f)
            
            # Filter to subset tasks only
            subset_tasks = [
                task for task in all_tasks 
                if task.get("task_id") in SUBSET_TASK_IDS
            ]
            
            logger.info(f"Loaded {len(subset_tasks)} subset tasks from {len(all_tasks)} total tasks")
            return subset_tasks
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing tasks.json: {e}")
            return []
        except IOError as e:
            logger.error(f"Error reading tasks.json: {e}")
            return []
    
    def _build_raw_data(self) -> List[Dict[str, Any]]:
        """
        Build raw data records from the HuggingFace dataset.
        
        Returns:
            List of dictionaries containing task metadata
        """
        tasks = self._load_tasks_from_huggingface()
        
        raw_data = []
        for task in tasks:
            task_id = task.get("task_id", "")
            task_type = task_id.split(":")[0] if ":" in task_id else "unknown"
            
            record = {
                "task_id": task_id,
                "task_type": task_type,
                "project_name": task.get("project_name", "unknown"),
                "project_language": task.get("project_language", "unknown"),
                "vulnerability_description": task.get("vulnerability_description", ""),
                "task_difficulty": task.get("task_difficulty", {}),
                "source": "cybergym",
                "metadata": {
                    "project_homepage": task.get("project_homepage", ""),
                    "project_main_repo": task.get("project_main_repo", ""),
                }
            }
            raw_data.append(record)
        
        return raw_data
    
    def retrieve(self) -> Optional[Path]:
        """
        Retrieve CyberGym metadata from the HuggingFace dataset.
        
        Returns:
            Path to the output file if successful, None otherwise
        """
        logger.info(f"Starting retrieval of CyberGym dataset to {self.output_dir}")
        
        # Check if data is available
        if not self._verify_data_available():
            logger.error("Cannot retrieve CyberGym - dataset not available")
            logger.error("Run 'make cybergym-setup' to download the dataset")
            return None
        
        # Build raw data from HuggingFace dataset
        raw_data = self._build_raw_data()
        
        if not raw_data:
            logger.error("No CyberGym tasks found in subset")
            return None
        
        logger.info(f"Found {len(raw_data)} CyberGym tasks in subset")
        
        # Write to output file
        output_file_path = self.output_dir / self.output_filename
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for item in raw_data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Successfully wrote {len(raw_data)} CyberGym records to {output_file_path}")
            return output_file_path
        except IOError as e:
            logger.error(f"Error writing CyberGym raw data to {output_file_path}: {e}")
            return None
    
    def cleanup(self) -> None:
        """No specific cleanup needed for CyberGym retrieval."""
        pass

