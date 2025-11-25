"""
CVE-Bench dataset retriever using the cve-bench repository.

Dynamically reads available CVEs from the cloned cve-bench repo's challenges directory.
This retriever scans the actual challenges and creates raw CVE metadata for the prepare step.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_retriever("cvebench")
class CvebenchRetrieve(Retrieve):
    """
    Retrieves CVE-Bench metadata by scanning the cloned cve-bench repository.
    
    Dynamically reads CVE IDs from the challenges directory and extracts
    metadata from each challenge's metadata.yaml file.
    """
    
    DEFAULT_OUTPUT_FILENAME = "cvebench_raw_data.jsonl"
    
    def __init__(self, dataset_name: str = "cvebench", output_filename: Optional[str] = None):
        """
        Initialize the CVE-Bench retriever.
        
        Args:
            dataset_name: Name of the dataset (default: "cvebench")
            output_filename: Optional custom name for the output JSONL file.
        """
        super().__init__(dataset_name)
        self.output_filename = output_filename or self.DEFAULT_OUTPUT_FILENAME
        self.challenges_dir = config.CVEBENCH_CHALLENGES_DIR
    
    def _verify_repo_available(self) -> bool:
        """Verify that the cve-bench repo is cloned."""
        if not self.challenges_dir.exists():
            logger.error(f"CVE-Bench challenges directory not found: {self.challenges_dir}")
            logger.error("Run 'make third-party' to clone the cve-bench repo")
            return False
        return True
    
    def _get_cve_list(self) -> List[str]:
        """
        Get the list of CVE IDs from the challenges directory.
        
        Returns:
            List of CVE IDs (directory names starting with 'CVE-')
        """
        if not self.challenges_dir.exists():
            return []
        
        cve_ids = []
        for item in self.challenges_dir.iterdir():
            if item.is_dir() and item.name.startswith("CVE-"):
                cve_ids.append(item.name)
        
        return sorted(cve_ids)
    
    def _load_challenge_metadata(self, cve_id: str) -> Dict[str, Any]:
        """
        Load metadata for a specific CVE challenge.
        
        Args:
            cve_id: The CVE identifier
            
        Returns:
            Dictionary with challenge metadata
        """
        challenge_dir = self.challenges_dir / cve_id
        metadata_file = challenge_dir / "metadata.yaml"
        
        metadata = {
            "category": "unknown",
            "application": "unknown",
            "description": "",
        }
        
        if metadata_file.exists():
            try:
                import yaml
                with open(metadata_file, 'r') as f:
                    yaml_data = yaml.safe_load(f) or {}
                    metadata["category"] = yaml_data.get("attack_type", "unknown")
                    metadata["application"] = yaml_data.get("application", "unknown")
                    metadata["description"] = yaml_data.get("description", "")
            except Exception as e:
                logger.warning(f"Could not parse metadata.yaml for {cve_id}: {e}")
        
        return metadata
    
    def _build_raw_data(self) -> List[Dict[str, Any]]:
        """
        Build raw data records by scanning the challenges directory.
        
        Returns:
            List of dictionaries containing CVE metadata
        """
        raw_data = []
        cve_ids = self._get_cve_list()
        
        for cve_id in cve_ids:
            metadata = self._load_challenge_metadata(cve_id)
            
            record = {
                "task_id": cve_id,
                "name": cve_id,
                "category": metadata["category"],
                "application": metadata["application"],
                "variants": ["one_day", "zero_day"],
                "source": "cve-bench",
                "metadata": {
                    "cve_id": cve_id,
                    "category": metadata["category"],
                    "target_application": metadata["application"],
                    "description": metadata.get("description", ""),
                }
            }
            raw_data.append(record)
        
        return raw_data
    
    def retrieve(self) -> Optional[Path]:
        """
        Retrieve CVE-Bench metadata by scanning the challenges directory.
        
        Returns:
            Path to the output file if successful, None otherwise
        """
        logger.info(f"Starting retrieval of CVE-Bench dataset to {self.output_dir}")
        
        # Check if repo is cloned
        if not self._verify_repo_available():
            logger.error("Cannot retrieve CVE-Bench - repository not available")
            logger.error("Run 'make third-party' to clone the cve-bench repo")
            return None
        
        # Build raw data by scanning challenges directory
        raw_data = self._build_raw_data()
        
        if not raw_data:
            logger.error("No CVE challenges found in challenges directory")
            return None
        
        logger.info(f"Found {len(raw_data)} CVE challenges in {self.challenges_dir}")
        
        # Write to output file
        output_file_path = self.output_dir / self.output_filename
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for item in raw_data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Successfully wrote {len(raw_data)} CVE records to {output_file_path}")
            return output_file_path
        except IOError as e:
            logger.error(f"Error writing CVE-Bench raw data to {output_file_path}: {e}")
            return None
    
    def cleanup(self) -> None:
        """No specific cleanup needed for CVE-Bench retrieval."""
        pass

