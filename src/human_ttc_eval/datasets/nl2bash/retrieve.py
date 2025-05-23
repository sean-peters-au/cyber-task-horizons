"""Retrieve NL2Bash dataset."""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from human_ttc_eval.core.base_retriever import BaseRetriever
from human_ttc_eval.core.registry import register_retriever

logger = logging.getLogger(__name__)


@register_retriever("nl2bash")
class NL2BashRetriever(BaseRetriever):
    """Retriever for NL2Bash dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.repo_path = self.project_root / "third-party" / "nl2bash"
        self.data_path = self.repo_path / "data" / "bash"
    
    def retrieve_metadata(self) -> Dict[str, Any]:
        """Retrieve metadata about the NL2Bash dataset."""
        self._ensure_dataset()
        
        nl_file = self.data_path / "all.nl"
        cm_file = self.data_path / "all.cm"
        
        # Count lines in each file
        with open(nl_file) as f:
            nl_count = sum(1 for _ in f)
        with open(cm_file) as f:
            cm_count = sum(1 for _ in f)
        
        metadata = {
            "dataset_name": "nl2bash",
            "total_examples": nl_count,
            "nl_descriptions": nl_count,
            "commands": cm_count,
            "files_match": nl_count == cm_count,
            "data_path": str(self.data_path),
            "nl_file": str(nl_file),
            "command_file": str(cm_file)
        }
        
        logger.info(f"NL2Bash dataset contains {nl_count} examples")
        if nl_count == cm_count:
            logger.info("✓ Parallel files have matching line counts")
        else:
            logger.warning("✗ Parallel files have different line counts")
        
        return metadata
    
    def download_challenge_content(self, challenge_id: str, output_dir: Path) -> Optional[Path]:
        """Download content for a specific challenge.
        
        For NL2Bash, this is not applicable as it's a paired dataset.
        Individual examples are extracted during parsing.
        """
        logger.warning("NL2Bash does not support individual challenge downloads")
        return None
    
    def _ensure_dataset(self) -> None:
        """Ensure NL2Bash dataset is available, downloading if necessary."""
        nl_file = self.data_path / "all.nl"
        cm_file = self.data_path / "all.cm"
        
        if nl_file.exists() and cm_file.exists():
            logger.info(f"NL2Bash dataset already available at {self.data_path}")
            return
        
        # Download the repository
        logger.info("Downloading NL2Bash dataset...")
        third_party_dir = self.repo_path.parent
        third_party_dir.mkdir(exist_ok=True)
        
        if self.repo_path.exists():
            logger.info(f"Repository exists at {self.repo_path}, pulling latest...")
            subprocess.run(
                ["git", "pull"],
                cwd=self.repo_path,
                check=True
            )
        else:
            subprocess.run(
                ["git", "clone", "--depth=1", "https://github.com/TellinaTool/nl2bash.git"],
                cwd=third_party_dir,
                check=True
            )
        
        # Verify files exist
        if not (nl_file.exists() and cm_file.exists()):
            raise FileNotFoundError(f"Expected data files not found in {self.data_path}")
        
        logger.info(f"NL2Bash dataset ready at {self.data_path}")


if __name__ == "__main__":
    retriever = NL2BashRetriever()
    metadata = retriever.retrieve_metadata()
    print(f"Dataset metadata: {metadata}") 