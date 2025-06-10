"""
CyBench dataset retriever using inspect_evals challenges.

Retrieves CyBench challenge data from the inspect_evals repository
and stores it in data/raw/cybench/ for processing.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_retriever("cybench")
class CybenchRetrieve(Retrieve):
    """Retrieves CyBench challenges from the inspect_evals repository."""
    
    DEFAULT_OUTPUT_FILENAME = "cybench_raw_data.jsonl"
    CHALLENGE_INFO_FILENAME = "challenge.yaml"
    
    def __init__(self, dataset_name: str = "cybench", output_filename: Optional[str] = None):
        """
        Initialize the CyBench retriever.
        
        Args:
            dataset_name: Name of the dataset (default: "cybench")
            output_filename: Optional custom name for the output JSONL file.
                           Defaults to "cybench_raw_data.jsonl".
        """
        super().__init__(dataset_name)
        
        # Path to inspect_evals cybench challenges
        self.inspect_evals_path = Path("third-party/inspect_evals")
        self.cybench_challenges_path = self.inspect_evals_path / "src" / "inspect_evals" / "cybench" / "challenges"
        self.output_filename = output_filename or self.DEFAULT_OUTPUT_FILENAME
        
        # Load timing data from the original CyBench if available
        self.timing_data = self._load_original_timing_data()
        
    def _load_original_timing_data(self) -> Dict[str, Any]:
        """Load timing data from original CyBench repository if available."""
        timing_data = {}
        
        # Try to load from the original CyBench analytics if it exists
        original_cybench_path = config.CYBENCH_REPO_PATH
        fastest_solve_times_path = original_cybench_path / "analytics" / "CTF fastest solve times.json"
        
        if fastest_solve_times_path.exists():
            try:
                with open(fastest_solve_times_path, 'r', encoding='utf-8') as f:
                    fst_list = json.load(f)
                    # Create lookup by challenge name
                    for item in fst_list:
                        # Extract challenge name from path (last component)
                        challenge_dir = item.get("challenge_dir", "")
                        if challenge_dir:
                            challenge_name = Path(challenge_dir).name
                            timing_data[challenge_name] = {
                                "fastest_solve_time_str": item.get("challenge_fastest_solve_time"),
                                "fastest_solve_time_seconds": self._parse_time_to_seconds(
                                    item.get("challenge_fastest_solve_time")
                                )
                            }
                logger.info(f"Loaded timing data for {len(timing_data)} challenges from original CyBench")
            except Exception as e:
                logger.warning(f"Could not load timing data from original CyBench: {e}")
        
        return timing_data
    
    def _parse_time_to_seconds(self, time_str: Optional[str]) -> Optional[float]:
        """Converts time string (H:MM:SS or MM:SS) to seconds."""
        if not time_str:
            return None
        try:
            parts = list(map(int, time_str.split(':')))
            if len(parts) == 3:
                return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
            elif len(parts) == 2:
                return float(parts[0] * 60 + parts[1])
            logger.warning(f"Unexpected time format: {time_str}")
            return None
        except ValueError:
            logger.warning(f"Could not parse time string: {time_str}")
            return None
    
    def _find_challenge_dirs(self) -> List[Path]:
        """Find all challenge directories containing challenge.yaml files."""
        challenge_dirs = []
        
        if not self.cybench_challenges_path.exists():
            logger.error(f"CyBench challenges directory not found: {self.cybench_challenges_path}")
            return challenge_dirs
        
        # Recursively find all challenge.yaml files
        for yaml_file in self.cybench_challenges_path.rglob(self.CHALLENGE_INFO_FILENAME):
            challenge_dirs.append(yaml_file.parent)
        
        return sorted(challenge_dirs)  # Sort for consistent ordering
    
    def _load_challenge_info(self, challenge_dir: Path) -> Optional[Dict[str, Any]]:
        """Load and parse challenge.yaml file."""
        yaml_path = challenge_dir / self.CHALLENGE_INFO_FILENAME
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading challenge info from {yaml_path}: {e}")
            return None
    
    def _extract_challenge_data(self, challenge_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract comprehensive data for a single challenge."""
        challenge_info = self._load_challenge_info(challenge_dir)
        if not challenge_info:
            return None
        
        challenge_name = challenge_info.get("name", challenge_dir.name)
        
        # Get timing data if available
        timing_info = self.timing_data.get(challenge_name, {})
        
        # Extract metadata from challenge info
        metadata = challenge_info.get("metadata", {})
        
        # Build the raw data record
        raw_data = {
            "task_id": challenge_name,
            "name": challenge_name,
            "flag": challenge_info.get("flag"),
            "variants": {},
            "metadata": metadata,
            "challenge_dir": str(challenge_dir.relative_to(self.inspect_evals_path)),
            "files": challenge_info.get("files", {}),
        }
        
        if "first_solve_time" in metadata:
            # Convert from minutes to seconds
            first_solve_minutes = metadata.get("first_solve_time", 0)
            raw_data["first_solve_time_minutes"] = first_solve_minutes
            raw_data["first_solve_time_seconds"] = float(first_solve_minutes * 60)
            raw_data["timing_source"] = "inspect_evals_metadata"
        else:
            # No timing data available
            logger.warning(f"No timing data found for challenge: {challenge_name}")
            raw_data["timing_source"] = "none"
        
        # Process variants
        for variant_name, variant_data in challenge_info.get("variants", {}).items():
            raw_data["variants"][variant_name] = {
                "prompt": variant_data.get("prompt", ""),
                "files": variant_data.get("files", {}),
                "metadata": variant_data.get("metadata", {})
            }
        
        # Try to determine category from metadata or default to "unknown"
        raw_data["category"] = metadata.get("category", "unknown")
        
        return raw_data
    
    def retrieve(self) -> Optional[Path]:
        """
        Retrieve CyBench challenges from inspect_evals repository.
        
        Returns:
            Path to the output file if successful, None otherwise
        """
        logger.info(f"Starting retrieval of CyBench dataset from inspect_evals to {self.output_dir}")
        
        # Check if inspect_evals exists
        if not self.inspect_evals_path.exists():
            logger.error(f"inspect_evals repository not found at {self.inspect_evals_path}")
            logger.error("Please run 'make third-party' to clone required repositories")
            return None
        
        # Find all challenge directories
        challenge_dirs = self._find_challenge_dirs()
        if not challenge_dirs:
            logger.error("No challenges found in inspect_evals CyBench directory")
            return None
        
        logger.info(f"Found {len(challenge_dirs)} challenge directories")
        
        # Process each challenge
        all_challenge_data = []
        for challenge_dir in challenge_dirs:
            challenge_data = self._extract_challenge_data(challenge_dir)
            if challenge_data:
                all_challenge_data.append(challenge_data)
            else:
                logger.warning(f"Failed to extract data from {challenge_dir}")
        
        if not all_challenge_data:
            logger.error("No challenge data could be extracted")
            return None
        
        logger.info(f"Successfully processed {len(all_challenge_data)} challenges")
        
        # Write to output file
        output_file_path = self.output_dir / self.output_filename
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for item in all_challenge_data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Successfully wrote {len(all_challenge_data)} CyBench challenges to {output_file_path}")
            return output_file_path
        except IOError as e:
            logger.error(f"Error writing CyBench raw data to {output_file_path}: {e}")
            return None
    
    def cleanup(self) -> None:
        """No specific cleanup needed for CyBench retrieval."""
        pass