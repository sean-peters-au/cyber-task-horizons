"""
NYUCTF dataset retriever.

Retrieves NYUCTF challenge data from the NYU_CTF_Bench repository
and stores it in data/raw/nyuctf/ for processing.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)


@register_retriever("nyuctf")
class NyuctfRetrieve(Retrieve):
    """Retrieves NYUCTF challenges from the NYU_CTF_Bench repository."""
    
    DEFAULT_OUTPUT_FILENAME = "nyuctf_raw_data.jsonl"
    
    def __init__(self, dataset_name: str = "nyuctf", output_filename: Optional[str] = None):
        """
        Initialize the NYUCTF retriever.
        
        Args:
            dataset_name: Name of the dataset (default: "nyuctf")
            output_filename: Optional custom name for the output JSONL file.
                           Defaults to "nyuctf_raw_data.jsonl".
        """
        super().__init__(dataset_name)
        
        # Path to NYU_CTF_Bench repository
        self.nyu_ctf_bench_path = Path("third-party/NYU_CTF_Bench")
        self.output_filename = output_filename or self.DEFAULT_OUTPUT_FILENAME
        
        # Load dataset metadata
        self.test_dataset_path = self.nyu_ctf_bench_path / "test_dataset.json"
        self.dev_dataset_path = self.nyu_ctf_bench_path / "development_dataset.json"
        
    def _load_dataset_metadata(self, dataset_path: Path) -> Dict[str, Any]:
        """Load challenge metadata from dataset JSON file."""
        if not dataset_path.exists():
            logger.warning(f"Dataset file not found: {dataset_path}")
            return {}
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading dataset metadata from {dataset_path}: {e}")
            return {}
    
    def _load_challenge_info(self, challenge_dir: Path) -> Optional[Dict[str, Any]]:
        """Load challenge.json file from a challenge directory."""
        challenge_json_path = challenge_dir / "challenge.json"
        
        if not challenge_json_path.exists():
            logger.warning(f"No challenge.json found in {challenge_dir}")
            return None
        
        try:
            with open(challenge_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading challenge info from {challenge_json_path}: {e}")
            return None
    
    def _extract_challenge_data(self, task_id: str, challenge_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract comprehensive data for a single challenge."""
        challenge_path = self.nyu_ctf_bench_path / challenge_meta["path"]
        
        if not challenge_path.exists():
            logger.warning(f"Challenge directory not found: {challenge_path}")
            return None
        
        # Load challenge.json
        challenge_info = self._load_challenge_info(challenge_path)
        if not challenge_info:
            return None
        
        # Check for Docker compose support
        docker_compose_path = challenge_path / "docker-compose.yml"
        has_docker_compose = docker_compose_path.exists()
        
        # Extract files list
        files = challenge_info.get("files", [])
        
        # Build the raw data record
        raw_data = {
            "task_id": task_id,
            "name": challenge_info.get("name", challenge_meta["challenge"]),
            "category": challenge_meta["category"],
            "year": challenge_meta["year"],
            "event": challenge_meta["event"],
            "challenge": challenge_meta["challenge"],
            "path": challenge_meta["path"],
            "flag": challenge_info.get("flag", ""),
            "description": challenge_info.get("description", ""),
            "author": challenge_info.get("author", ""),
            "points": challenge_info.get("points", challenge_info.get("initial", 0)),
            "files": files,
            "has_docker_compose": has_docker_compose,
            "docker_compose_path": str(docker_compose_path.relative_to(self.nyu_ctf_bench_path)) if has_docker_compose else None,
            "challenge_type": challenge_info.get("type", "static"),
            "internal_port": challenge_info.get("internal_port"),
            "box": challenge_info.get("box"),
            "reference": challenge_info.get("reference", ""),
            
            # Timing information - use default for now
            "human_minutes": 10.0,  # Default 10 minutes as requested
            "timing_source": "default_estimate",
            
            # Full challenge info for benchmark use
            "challenge_info": challenge_info,
            "challenge_directory": str(challenge_path.relative_to(self.nyu_ctf_bench_path))
        }
        
        return raw_data
    
    def retrieve(self) -> Optional[Path]:
        """
        Retrieve NYUCTF challenges from NYU_CTF_Bench repository.
        
        Returns:
            Path to the output file if successful, None otherwise
        """
        logger.info(f"Starting retrieval of NYUCTF dataset from NYU_CTF_Bench to {self.output_dir}")
        
        # Check if NYU_CTF_Bench exists
        if not self.nyu_ctf_bench_path.exists():
            logger.error(f"NYU_CTF_Bench repository not found at {self.nyu_ctf_bench_path}")
            logger.error("Please run 'make third-party' to clone required repositories")
            return None
        
        # Load test and development datasets
        test_dataset = self._load_dataset_metadata(self.test_dataset_path)
        dev_dataset = self._load_dataset_metadata(self.dev_dataset_path)
        
        if not test_dataset and not dev_dataset:
            logger.error("No dataset metadata could be loaded")
            return None
        
        # Combine datasets (prioritize test dataset)
        all_challenges = {}
        all_challenges.update(dev_dataset)
        all_challenges.update(test_dataset)  # test overwrites dev if same task_id
        
        logger.info(f"Found {len(all_challenges)} total challenges ({len(test_dataset)} test, {len(dev_dataset)} dev)")
        
        # Process each challenge
        all_challenge_data = []
        skipped_challenges = []
        
        for task_id, challenge_meta in all_challenges.items():
            challenge_data = self._extract_challenge_data(task_id, challenge_meta)
            if challenge_data:
                # Mark whether this is test or dev split
                challenge_data["split"] = "test" if task_id in test_dataset else "development"
                all_challenge_data.append(challenge_data)
            else:
                skipped_challenges.append(task_id)
        
        if skipped_challenges:
            logger.warning(f"Skipped {len(skipped_challenges)} challenges due to missing data: {skipped_challenges[:10]}{'...' if len(skipped_challenges) > 10 else ''}")
        
        if not all_challenge_data:
            logger.error("No challenge data could be extracted")
            return None
        
        logger.info(f"Successfully processed {len(all_challenge_data)} challenges")
        
        # Log category distribution
        category_counts = {}
        for challenge in all_challenge_data:
            category = challenge["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        logger.info(f"Category distribution: {category_counts}")
        
        # Write to output file
        output_file_path = self.output_dir / self.output_filename
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for item in all_challenge_data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Successfully wrote {len(all_challenge_data)} NYUCTF challenges to {output_file_path}")
            return output_file_path
        except IOError as e:
            logger.error(f"Error writing NYUCTF raw data to {output_file_path}: {e}")
            return None
    
    def cleanup(self) -> None:
        """No specific cleanup needed for NYUCTF retrieval."""
        pass