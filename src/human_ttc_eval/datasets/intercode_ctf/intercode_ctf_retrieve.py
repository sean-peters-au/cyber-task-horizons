"""
InterCode-CTF dataset retriever.

Retrieves raw InterCode CTF data from the third-party repository
and stores it in data/raw/intercode-ctf/.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_retriever("intercode-ctf")
class InterCodeCTFRetrieve(Retrieve):
    """Retrieves raw InterCode CTF data from the repository."""
    
    DEFAULT_OUTPUT_FILENAME = "intercode_ctf_raw_data.jsonl"
    
    def __init__(self, dataset_name: str = "intercode-ctf", output_filename: Optional[str] = None):
        """
        Initialize the InterCode-CTF retriever.
        
        Args:
            dataset_name: Name of the dataset (default: "intercode-ctf")
            output_filename: Optional custom name for the output JSONL file.
                           Defaults to "intercode_ctf_raw_data.jsonl".
        """
        super().__init__(dataset_name)
        self.intercode_repo_path = config.INTERCODE_REPO_PATH
        self.source_data_file = self.intercode_repo_path / "data" / "ctf" / "ic_ctf.json"
        self.output_filename = output_filename or self.DEFAULT_OUTPUT_FILENAME
        
    def retrieve(self) -> Optional[Path]:
        """
        Retrieve the raw InterCode CTF data.
        
        This method:
        1. Reads the ic_ctf.json file from the InterCode repository
        2. Processes it into JSONL format with initial metadata
        3. Saves to data/raw/intercode-ctf/
        
        Returns:
            Path to the output file if successful, None otherwise
        """
        logger.info(f"Starting retrieval of InterCode-CTF dataset to {self.output_dir}")
        
        # Check if repository exists
        if not self.intercode_repo_path.exists():
            logger.error(f"InterCode repository not found at {self.intercode_repo_path}")
            return None
        
        # Verify source file exists
        if not self.source_data_file.is_file():
            logger.error(f"InterCode CTF data file not found: {self.source_data_file}")
            return None
        
        output_records = []
        try:
            # Read the source JSON file
            with open(self.source_data_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)  # It's a list of dicts
            
            logger.info(f"Found {len(tasks_data)} tasks in {self.source_data_file}")
            
            # Process each task
            for task in tasks_data:
                processed_task = {
                    "task_id_intercode": task.get("task_id"),
                    "description": task.get("query"),
                    "solution_flag": task.get("gold"),
                    "source_url": task.get("source"),
                    "tags": task.get("tags", []),
                    "setup_commands": task.get("setup"),
                    # Add our initial HTC estimate
                    "estimated_time_seconds": 210.0,  # 3.5 minutes
                    "timing_source": "author_reported_average_via_cybench_paper"
                }
                output_records.append(processed_task)
            
            logger.info(f"Successfully processed {len(output_records)} tasks from {self.source_data_file}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.source_data_file}: {e}")
            return None
        except IOError as e:
            logger.error(f"Error reading {self.source_data_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during data processing: {e}", exc_info=True)
            return None
        
        if not output_records:
            logger.warning("No records were processed. Output file will be empty.")
            return None
        
        # Write to output file
        output_file_path = self.output_dir / self.output_filename
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                for record in output_records:
                    f_out.write(json.dumps(record) + '\n')
            logger.info(f"Successfully wrote {len(output_records)} InterCode-CTF records to {output_file_path}")
            return output_file_path
        except IOError as e:
            logger.error(f"Error writing data to {output_file_path}: {e}")
            return None
    
    def cleanup(self) -> None:
        """
        Clean up any temporary resources.
        
        Note: InterCode-CTF challenge content is part of the cloned repository 
        and its Docker environment. No specific cleanup needed.
        """
        pass 