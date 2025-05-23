from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional

class BaseRetriever(ABC):
    """
    Abstract base class for dataset retrievers.
    Retrievers are responsible for fetching raw data, challenge lists, solve times, etc.,
    from external sources (e.g., APIs, websites) for a specific dataset.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def retrieve_metadata(self, **kwargs) -> Optional[Path]:
        """
        Fetches metadata for challenges (e.g., IDs, names, points, categories, human times).
        Should save the metadata to a file (e.g., JSONL) within self.output_dir 
        and return the path to the saved file, or None if failed.
        Specific arguments will vary by retriever.
        """
        pass

    @abstractmethod
    def download_challenge_content(self, metadata_file: Path, **kwargs) -> None:
        """
        Downloads the actual challenge content (files, descriptions, dockerfiles, etc.)
        based on previously fetched metadata.
        Saves content into self.output_dir, structured appropriately.
        Specific arguments will vary by retriever.
        """
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        Returns a slug-like name for the dataset this retriever handles.
        """
        pass 