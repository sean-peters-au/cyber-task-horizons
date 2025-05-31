"""
Base class for dataset retrievers.

Retrievers are responsible for fetching raw data from external sources
and storing it in data/raw/<dataset>/ for processing by Prepare classes.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

from .. import config

logger = logging.getLogger(__name__)


class Retrieve(ABC):
    """
    Abstract base class for dataset retrievers.
    
    Retrievers fetch raw data from external sources (repos, APIs, websites)
    and store it unmodified in the data/raw/<dataset>/ directory.
    
    Key principle: No data transformation occurs here. Raw data is saved
    exactly as retrieved for processing by the Prepare stage.
    """
    
    def __init__(self, dataset_name: str):
        """
        Initialize the retriever.

        Args:
            dataset_name: The unique identifier for the dataset.
                          This name will be used to determine the output directory
                          (e.g., data/raw/<dataset_name>/).
        """
        self.dataset_name = dataset_name
        self.output_dir = config.DATA_DIR / "raw" / self.dataset_name
        
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized {self.__class__.__name__} for dataset \'{self.dataset_name}\'. Output directory: {self.output_dir}")
        except OSError as e:
            logger.error(f"Error creating output directory {self.output_dir}: {e}")
            raise
    
    @abstractmethod
    def retrieve(self) -> Any:
        """
        Retrieve the raw data for the dataset.

        This method should implement the logic to download, extract, or otherwise
        obtain the raw dataset files and save them into `self.output_dir`.

        Returns:
            A confirmation (e.g., list of downloaded file paths, success boolean, or None).
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Optional: Clean up any temporary files or resources after retrieval.
        
        This method can be overridden by subclasses if specific cleanup
        actions are needed (e.g., deleting temporary archives).
        """
        pass

    def run(self) -> Any:
        """
        Orchestrates the retrieval process.
        
        This typically involves calling retrieve() and then cleanup().
        """
        logger.info(f"Starting retrieval for dataset: {self.dataset_name}...")
        try:
            result = self.retrieve()
            logger.info(f"Retrieval successful for {self.dataset_name}.")
            self.cleanup()
            logger.info(f"Cleanup successful for {self.dataset_name}.")
            return result
        except Exception as e:
            logger.error(f"Error during retrieval for {self.dataset_name}: {e}", exc_info=True)
            raise