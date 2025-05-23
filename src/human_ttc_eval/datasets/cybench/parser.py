import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ...core.base_parser import BaseParser
from ...core.registry import register_parser
from ...core.utils import slugify

logger = logging.getLogger(__name__)

@register_parser("cybench")
class CyBenchParser(BaseParser):
    """
    Parses CyBench metadata (from retriever) into the METR all_runs.jsonl format.
    
    Converts cybench_metadata.jsonl into METR-style records where each task
    represents a human "run" with the fastest solve time as the human baseline.
    """

    @property
    def dataset_name(self) -> str:
        return "cybench"

    def __init__(self, input_dir: Path, output_file: Path, metadata_file: str = "cybench_metadata.jsonl"):
        """
        Args:
            input_dir: Directory containing the cybench metadata file
            output_file: Output path for the all_runs.jsonl file
            metadata_file: Name of the metadata file (default: cybench_metadata.jsonl)
        """
        super().__init__(input_dir, output_file)
        self.metadata_file_path = self.input_dir / metadata_file
        logger.info(f"CyBenchParser initialized. Metadata: {self.metadata_file_path}, Output: {self.output_file}")

    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse cybench metadata into METR all_runs.jsonl format.
        
        Each metadata record becomes one "human run" record representing
        the fastest solve time for that challenge.
        """
        if not self.metadata_file_path.exists():
            logger.error(f"CyBench metadata file not found: {self.metadata_file_path}")
            return []

        all_runs: List[Dict[str, Any]] = []
        
        try:
            with open(self.metadata_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        metadata_record = json.loads(line)
                        run_record = self._convert_metadata_to_run(metadata_record)
                        if run_record:
                            all_runs.append(run_record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSONDecodeError on line {line_num} in {self.metadata_file_path}: {e}. Skipping line.")
                    except Exception as e:
                        logger.error(f"Error processing line {line_num} in {self.metadata_file_path}: {e}. Skipping line.")

        except IOError as e:
            logger.error(f"Error reading metadata file {self.metadata_file_path}: {e}")
            return []

        logger.info(f"Successfully converted {len(all_runs)} CyBench metadata records to run format")
        return all_runs

    def _convert_metadata_to_run(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert a single cybench metadata record to METR all_runs format.
        
        Args:
            metadata: Single record from cybench_metadata.jsonl
            
        Returns:
            Dict in METR all_runs format, or None if conversion fails
        """
        try:
            # Extract basic info
            task_path = metadata.get("task_path_in_repo")
            if not task_path:
                logger.warning("Missing task_path_in_repo in metadata record. Skipping.")
                return None

            # Create task identifiers
            task_family = self._create_task_family(metadata)
            task_id = f"{task_family}/main"
            
            # Get timing data
            fastest_solve_time_seconds = metadata.get("fastest_solve_time_seconds")
            if fastest_solve_time_seconds is None:
                logger.warning(f"Missing fastest_solve_time_seconds for {task_path}. Skipping.")
                return None
            
            human_minutes = fastest_solve_time_seconds / 60.0
            
            # Create run ID
            organization = metadata.get("organization", "unknown")
            event = metadata.get("event", "unknown")
            title_slug = slugify(metadata.get("title", "unknown"))
            run_id = f"human_cybench_{slugify(organization)}_{slugify(event)}_{title_slug}_fst"

            # Build the run record
            run_record = {
                "task_id": task_id,
                "task_family": task_family,
                "run_id": run_id,
                "alias": "Human (CyBench FST)",
                "model": "human",
                "score_binarized": 1,  # We only have successful solves
                "score_cont": 1.0,
                "fatal_error_from": None,
                "human_minutes": human_minutes,
                "human_score": 1.0,
                "human_source": "cybench_first_solve_times",
                "task_source": "cybench",
                "generation_cost": 0.0,
                "human_cost": None,  # Could calculate from human_minutes if needed
                "time_limit": None,   # CyBench doesn't specify time limits
                "started_at": None,   # Contest start time not available
                "completed_at": None, # Could be start + human_minutes if we had start time
                "task_version": "1.0",
                "equal_task_weight": None,    # Will be calculated later if needed
                "invsqrt_task_weight": None,  # Will be calculated later if needed
                
                # CyBench-specific metadata
                "category": metadata.get("category"),
                "difficulty": metadata.get("difficulty"),
                "organization": metadata.get("organization"),
                "event": metadata.get("event"),
                "points": metadata.get("points"),
                "description": metadata.get("description"),
                "authors": metadata.get("authors"),
                "tags": metadata.get("tags", []),
                "fastest_solve_time_str": metadata.get("fastest_solve_time_str"),
                "timing_source": metadata.get("timing_source"),
                
                # Raw metadata for reference
                "_raw_task_path": task_path,
                "_raw_metadata": metadata
            }
            
            return run_record
            
        except Exception as e:
            logger.error(f"Error converting metadata to run format: {e}", exc_info=True)
            return None

    def _create_task_family(self, metadata: Dict[str, Any]) -> str:
        """
        Create a task family identifier from metadata.
        
        Format: {organization}_{event}_{category}_{title_slug}
        Example: hackthebox_cyber-apocalypse-2024_crypto_dynastic
        """
        organization = metadata.get("organization", "unknown")
        event = metadata.get("event", "unknown") 
        category = metadata.get("category", "unknown")
        title = metadata.get("title", "unknown")
        
        # Create slugified components
        org_slug = slugify(organization)
        event_slug = slugify(event)
        category_slug = slugify(category)
        title_slug = slugify(title)
        
        return f"{org_slug}_{event_slug}_{category_slug}_{title_slug}" 