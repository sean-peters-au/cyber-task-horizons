"""
CyBashBench dataset retriever.

Retrieves the handcrafted CyBashBench task file from its source location
in the repository and copies it to data/raw/cybashbench/.
"""

import logging
import shutil
from pathlib import Path
from typing import List

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_retriever("cybashbench")
class CyBashBenchRetrieve(Retrieve):
    """Retrieves the raw CyBashBench dataset file."""

    def __init__(self, dataset_name: str = "cybashbench"):
        """
        Initialize the CyBashBench retriever.

        Args:
            dataset_name: Name of the dataset (default: "cybashbench")
        """
        super().__init__(dataset_name)
        # Assumes the tasks file is in a 'cybashbench' directory at the project root.
        # This path might need to be added to config.py if it doesn't use PROJECT_ROOT
        self.source_data_file = config.PROJECT_ROOT / "data" / "keep" / "cybashbench" / "cybashbench_tasks.jsonl"

    def retrieve(self) -> List[Path]:
        """
        Retrieve the raw CyBashBench data file.

        This method copies the raw .jsonl file to data/raw/cybashbench/.

        Returns:
            List containing the path to the copied file.
        """
        logger.info(f"Starting retrieval of CyBashBench dataset to {self.output_dir}")

        # Verify source file exists
        if not self.source_data_file.exists():
            raise FileNotFoundError(f"CyBashBench source file not found: {self.source_data_file}. Please ensure it is located at 'data/keep/cybashbench/'.")

        # Define destination file
        dest_file = self.output_dir / self.source_data_file.name

        # Copy file to raw data directory
        copied_files = []
        try:
            shutil.copy2(self.source_data_file, dest_file)
            copied_files.append(dest_file)
            logger.info(f"Copied {self.source_data_file} to {dest_file}")

            # Log file statistics
            with open(dest_file, 'r', encoding='utf-8') as f:
                task_count = sum(1 for _ in f)

            logger.info(f"Retrieved CyBashBench dataset: {task_count} tasks")

            return copied_files

        except Exception as e:
            logger.error(f"Error copying CyBashBench file: {e}")
            raise

    def cleanup(self) -> None:
        """No cleanup needed for CyBashBench retrieval."""
        pass 