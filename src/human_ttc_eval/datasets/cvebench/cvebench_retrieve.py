"""
CVE-Bench dataset retriever using the cve-bench repository.

Dynamically reads available CVEs from the cloned cve-bench repo's challenges directory.
This retriever scans the actual challenges and creates raw CVE metadata for the prepare step.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_retriever("cvebench")
class CvebenchRetrieve(Retrieve):
    """
    Retrieves CVE-Bench metadata by scanning the cloned cve-bench repository.
    
    Dynamically reads CVE IDs from the challenges directory and extracts
    metadata from each challenge's eval.yml file.
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
        Load metadata for a specific CVE challenge from eval.yml.
        
        Parses the eval.yml file to extract:
        - Vulnerability description from one_day.prompt
        - Application URL and attacker role from metadata section
        - Target platform inferred from compose.yml
        
        Args:
            cve_id: The CVE identifier
            
        Returns:
            Dictionary with challenge metadata
        """
        challenge_dir = self.challenges_dir / cve_id
        eval_file = challenge_dir / "eval.yml"
        compose_file = challenge_dir / "compose.yml"
        
        metadata: Dict[str, Any] = {
            "vulnerability_description": "",
            "application_url": "",
            "attacker_role": "user",
            "target_platform": "unknown",
        }
        
        # Parse eval.yml for vulnerability info and metadata
        if eval_file.exists():
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    eval_data = yaml.safe_load(f) or {}
                
                # Extract vulnerability description from one_day prompt
                one_day_prompt = eval_data.get("variants", {}).get("one_day", {}).get("prompt", "")
                if one_day_prompt:
                    metadata["vulnerability_description"] = self._extract_vulnerability_description(one_day_prompt)
                
                # Extract metadata fields
                eval_metadata = eval_data.get("metadata", {})
                metadata["application_url"] = eval_metadata.get("application_url", "")
                metadata["attacker_role"] = eval_metadata.get("attacker_role", "user")
                
            except Exception as e:
                logger.warning(f"Could not parse eval.yml for {cve_id}: {e}")
        
        # Infer target platform from compose.yml
        if compose_file.exists():
            metadata["target_platform"] = self._infer_target_platform(compose_file)
        
        return metadata
    
    def _extract_vulnerability_description(self, prompt: str) -> str:
        """
        Extract the vulnerability description from the one_day prompt.
        
        The prompt follows a pattern like:
        "The website ... is known to have the following vulnerability:
        <description>
        
        As a/an user, exploit this vulnerability..."
        
        Args:
            prompt: The one_day variant prompt text
            
        Returns:
            Extracted vulnerability description
        """
        # Look for the pattern between "vulnerability:" and "As a/an"
        match = re.search(
            r"is known to have the following vulnerability:\s*\n\s*(.+?)\s*\n\s*As a/an",
            prompt,
            re.DOTALL
        )
        if match:
            return match.group(1).strip()
        
        # Fallback: try to extract just the line after "vulnerability:"
        match = re.search(r"following vulnerability:\s*\n\s*(.+?)(?:\n|$)", prompt)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _infer_target_platform(self, compose_file: Path) -> str:
        """
        Infer the target platform from compose.yml service definitions.
        
        Looks for common service patterns like:
        - wp, wordpress -> wordpress
        - mysql, mariadb -> database
        - zabbix -> zabbix
        - python, flask, django -> python
        
        Args:
            compose_file: Path to the compose.yml file
            
        Returns:
            Inferred platform name or "unknown"
        """
        try:
            with open(compose_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            # Check for common platform patterns
            if "compose-wp.yml" in content or "wordpress" in content:
                return "wordpress"
            if "zabbix" in content:
                return "zabbix"
            if "flask" in content or "django" in content:
                return "python"
            if "node" in content or "npm" in content:
                return "nodejs"
            if "php" in content:
                return "php"
            if "java" in content or "spring" in content:
                return "java"
            if "rust" in content or "cargo" in content:
                return "rust"
            if "go" in content or "golang" in content:
                return "go"
            
            return "unknown"
            
        except Exception as e:
            logger.warning(f"Could not parse compose.yml: {e}")
            return "unknown"
    
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
                "target_platform": metadata["target_platform"],
                "variants": ["one_day", "zero_day"],
                "source": "cve-bench",
                "metadata": {
                    "cve_id": cve_id,
                    "vulnerability_description": metadata["vulnerability_description"],
                    "application_url": metadata["application_url"],
                    "attacker_role": metadata["attacker_role"],
                    "target_platform": metadata["target_platform"],
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

