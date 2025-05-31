"""
NL2Bash dataset retriever.

Retrieves raw NL2Bash data files from the third-party repository
and stores them unmodified in data/raw/nl2bash/.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_retriever("nl2bash")
class NL2BashRetrieve(Retrieve):
    """Retrieves raw NL2Bash dataset files from the repository."""
    
    def __init__(self, dataset_name: str = "nl2bash"):
        """
        Initialize the NL2Bash retriever.
        
        Args:
            dataset_name: Name of the dataset (default: "nl2bash")
        """
        super().__init__(dataset_name)
        self.nl2bash_repo_path = config.NL2BASH_REPO_PATH
        self.source_data_path = self.nl2bash_repo_path / "data" / "bash"
        
    def retrieve(self) -> List[Path]:
        """
        Retrieve the raw NL2Bash data files.
        
        This method:
        1. Ensures the NL2Bash repository is available
        2. Copies the raw .nl and .cm files to data/raw/nl2bash/
        
        Returns:
            List of paths to the copied files
        """
        logger.info(f"Starting retrieval of NL2Bash dataset to {self.output_dir}")
        
        # Ensure the repository exists
        self._ensure_repository()
        
        # Define source files
        nl_file = self.source_data_path / "all.nl"
        cm_file = self.source_data_path / "all.cm"
        
        # Verify source files exist
        if not nl_file.exists():
            raise FileNotFoundError(f"NL file not found: {nl_file}")
        if not cm_file.exists():
            raise FileNotFoundError(f"CM file not found: {cm_file}")
        
        # Copy files to raw data directory
        copied_files = []
        
        try:
            # Copy the .nl file
            dest_nl_file = self.output_dir / "all.nl"
            shutil.copy2(nl_file, dest_nl_file)
            copied_files.append(dest_nl_file)
            logger.info(f"Copied {nl_file} to {dest_nl_file}")
            
            # Copy the .cm file
            dest_cm_file = self.output_dir / "all.cm"
            shutil.copy2(cm_file, dest_cm_file)
            copied_files.append(dest_cm_file)
            logger.info(f"Copied {cm_file} to {dest_cm_file}")
            
            # Log file statistics
            with open(dest_nl_file, 'r', encoding='utf-8') as f:
                nl_count = sum(1 for _ in f)
            with open(dest_cm_file, 'r', encoding='utf-8') as f:
                cm_count = sum(1 for _ in f)
            
            logger.info(f"Retrieved NL2Bash dataset: {nl_count} NL descriptions, {cm_count} commands")
            
            if nl_count != cm_count:
                logger.warning(f"Line count mismatch: {nl_count} NL lines vs {cm_count} CM lines")
            
            return copied_files
            
        except Exception as e:
            logger.error(f"Error copying NL2Bash files: {e}")
            raise
    
    def _ensure_repository(self) -> None:
        """Ensure the NL2Bash repository is available locally."""
        if not self.nl2bash_repo_path.exists():
            logger.info(f"NL2Bash repository not found at {self.nl2bash_repo_path}")
            logger.info("Cloning NL2Bash repository...")
            
            try:
                # Ensure third-party directory exists
                self.nl2bash_repo_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Clone the repository
                subprocess.run(
                    ["git", "clone", "--depth=1", "https://github.com/TellinaTool/nl2bash.git", str(self.nl2bash_repo_path)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Successfully cloned NL2Bash repository to {self.nl2bash_repo_path}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone NL2Bash repository: {e}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                raise RuntimeError(f"Failed to clone NL2Bash repository: {e.stderr}")
        else:
            logger.info(f"NL2Bash repository found at {self.nl2bash_repo_path}")
            
            # Optionally pull latest changes
            try:
                result = subprocess.run(
                    ["git", "pull"],
                    cwd=self.nl2bash_repo_path,
                    capture_output=True,
                    text=True,
                    check=False  # Don't raise on non-zero exit
                )
                if result.returncode == 0:
                    logger.info("Updated NL2Bash repository to latest version")
                else:
                    logger.debug(f"Git pull skipped or failed (may be expected): {result.stderr}")
            except Exception as e:
                logger.debug(f"Could not update repository (may be expected): {e}")
    
    def cleanup(self) -> None:
        """No cleanup needed for NL2Bash retrieval."""
        pass 