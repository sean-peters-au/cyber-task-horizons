"""
InterCode-CTF dataset preparer.

Transforms raw InterCode-CTF data into standardized Run objects representing
human baseline performance, adhering to METR schema.
"""

import json
import logging
import re
from typing import List, Dict, Any

from human_ttc_eval.core.prepare import Prepare
from human_ttc_eval.core.registry import register_preparer
from human_ttc_eval.core.run import Run

logger = logging.getLogger(__name__)


@register_preparer("intercode-ctf")
class InterCodeCTFPrepare(Prepare):
    """Prepares raw InterCode-CTF data into standardized Run objects."""
    
    RAW_FILENAME = "intercode_ctf_raw_data.jsonl"
    
    def __init__(self):
        """Initialize the InterCode-CTF preparer."""
        super().__init__(dataset_name="intercode-ctf")
        self.default_raw_input_filename = self.RAW_FILENAME
    
    def _convert_ssh_to_https(self, command: str) -> str:
        """
        Convert SSH git URLs to HTTPS URLs to avoid SSH host key verification issues.
        
        Args:
            command: Command string that may contain SSH git URLs
            
        Returns:
            Command string with SSH URLs converted to HTTPS
        """
        # Pattern to match SSH git URLs: git@hostname:user/repo.git
        ssh_pattern = r'git@([^:]+):([^/\s]+)/([^\s]+)'
        
        def replace_ssh_url(match):
            hostname = match.group(1)
            user = match.group(2)
            repo = match.group(3)
            # Convert to HTTPS format
            return f'https://{hostname}/{user}/{repo}'
        
        # Replace SSH URLs with HTTPS URLs
        converted_command = re.sub(ssh_pattern, replace_ssh_url, command)
        
        if converted_command != command:
            logger.info(f"Converted SSH URL to HTTPS: {command} -> {converted_command}")
        
        return converted_command
    
    def _process_setup_commands(self, setup_commands: Any) -> List[str]:
        """
        Process setup commands, converting SSH URLs to HTTPS and ensuring proper format.
        
        Args:
            setup_commands: Setup commands from raw data (string, list, or None)
            
        Returns:
            List of processed setup commands
        """
        if not setup_commands:
            return []
        
        # Handle both string and list formats
        if isinstance(setup_commands, str):
            commands = [setup_commands]
        elif isinstance(setup_commands, list):
            commands = setup_commands
        else:
            logger.warning(f"Invalid setup_commands format: {type(setup_commands)}")
            return []
        
        # Convert SSH URLs to HTTPS in each command
        processed_commands = []
        for command in commands:
            if isinstance(command, str):
                processed_command = self._convert_ssh_to_https(command)
                processed_commands.append(processed_command)
            else:
                logger.warning(f"Non-string command in setup_commands: {command}")
                processed_commands.append(str(command))
        
        return processed_commands
    
    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        """
        Extract InterCode-CTF specific metadata for task definitions.
        
        This metadata will be stored in the tasks.jsonl file and used
        by the benchmark harness.
        
        Args:
            representative_run: A Run object for the task
            
        Returns:
            Dictionary with InterCode-CTF specific metadata
        """
        # We need to re-read the raw data to get the full task metadata
        # This is inefficient but maintains separation of concerns
        task_metadata = {
            "description": "",
            "solution_flag": "",
            "source_url": "",
            "tags": [],
            "setup_commands": [],
            "task_id_intercode": "",
            "task_dir": ""
        }
        
        # Extract the original task ID from our standardized task_id
        # Format: intercode-ctf_{category}/task_{original_id}
        task_id_parts = representative_run.task_id.split('/')
        if len(task_id_parts) == 2 and task_id_parts[1].startswith('task_'):
            original_task_id_str = task_id_parts[1].replace('task_', '')
            try:
                original_task_id = int(original_task_id_str)
            except ValueError:
                logger.warning(f"Could not parse task ID from {representative_run.task_id}")
                return task_metadata
            
            # Re-read raw data to find this task
            raw_input_file = self.raw_data_dir / self.default_raw_input_filename
            if raw_input_file.exists():
                try:
                    with open(raw_input_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            raw_data = json.loads(line.strip())
                            if raw_data.get("task_id_intercode") == original_task_id:
                                # Process setup commands to convert SSH URLs to HTTPS
                                processed_setup_commands = self._process_setup_commands(
                                    raw_data.get("setup_commands")
                                )
                                
                                task_metadata = {
                                    "description": raw_data.get("description", ""),
                                    "solution_flag": raw_data.get("solution_flag", ""),
                                    "source_url": raw_data.get("source_url", ""),
                                    "tags": raw_data.get("tags", []),
                                    "setup_commands": processed_setup_commands,
                                    "task_id_intercode": raw_data.get("task_id_intercode", ""),
                                    "task_dir": str(raw_data.get("task_id_intercode", ""))
                                }
                                break
                except Exception as e:
                    logger.warning(f"Could not extract metadata for task {representative_run.task_id}: {e}")
        
        return task_metadata
    
    def prepare(self) -> List[Run]:
        """
        Load raw InterCode-CTF data and transform into Run objects.
        
        Returns:
            List of Run objects representing human baseline performance
        """
        logger.info("Starting InterCode-CTF dataset preparation")
        
        raw_input_file = self.raw_data_dir / self.default_raw_input_filename
        if not raw_input_file.exists():
            logger.error(f"Raw InterCode-CTF data file not found: {raw_input_file}")
            return []
        
        runs: List[Run] = []
        
        try:
            with open(raw_input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        raw_data = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON line {line_num} in {raw_input_file}: {e}")
                        continue
                    
                    # Extract fields
                    task_id_intercode = raw_data.get("task_id_intercode")
                    description = raw_data.get("description", "")
                    tags = raw_data.get("tags", [])
                    estimated_time_seconds = raw_data.get("estimated_time_seconds", 210.0)
                    timing_source = raw_data.get("timing_source", "author_reported_average")
                    
                    if not task_id_intercode:
                        logger.warning(f"Skipping record in {raw_input_file} (line {line_num}) due to missing 'task_id_intercode'.")
                        continue
                    
                    # Determine task category from tags
                    category = self._determine_category(tags)
                    
                    # Create standardized task_id and task_family
                    task_family = f"intercode-ctf_{category}"
                    task_id = f"{task_family}/task_{task_id_intercode}"
                    
                    # Convert time to minutes
                    try:
                        human_minutes = float(estimated_time_seconds) / 60.0
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid time format for task '{task_id_intercode}': {estimated_time_seconds}")
                        human_minutes = 3.5  # Default to 3.5 minutes
                    
                    # Create Run object
                    run_obj = Run(
                        task_id=task_id,
                        task_family=task_family,
                        run_id=f"human_{task_id.replace('/', '_')}_{timing_source}",
                        alias="Human Baseline (InterCode-CTF)",
                        model="human",
                        score_binarized=1,  # All human baselines assumed successful
                        score_cont=1.0,
                        human_minutes=human_minutes,
                        human_source=f"intercode_ctf_{timing_source}",
                        task_source="intercode_ctf_dataset",
                        started_at=0.0,
                        completed_at=float(estimated_time_seconds),
                        generation_cost=0.0,
                        fatal_error_from=None
                    )
                    
                    runs.append(run_obj)
            
            logger.info(f"Successfully prepared {len(runs)} runs from {raw_input_file}")
            
        except IOError as e:
            logger.error(f"Error reading raw InterCode-CTF data file {raw_input_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during InterCode-CTF preparation: {e}", exc_info=True)
            return []
        
        return runs
    
    def _determine_category(self, tags: List[str]) -> str:
        """
        Determine the category for a task based on its tags.
        
        Args:
            tags: List of tags from the raw data
            
        Returns:
            A category string for task grouping
        """
        if not tags:
            return "general"
        
        # Map common CTF categories
        category_keywords = {
            "web": ["web", "xss", "sql", "injection", "csrf"],
            "crypto": ["crypto", "cryptography", "cipher", "encryption", "hash"],
            "pwn": ["pwn", "binary", "exploit", "buffer", "overflow", "rop"],
            "reverse": ["reverse", "reversing", "re", "disassemble", "decompile"],
            "forensics": ["forensics", "forensic", "stego", "steganography", "memory"],
            "misc": ["misc", "miscellaneous", "trivia", "osint"]
        }
        
        # Check tags against category keywords
        for category, keywords in category_keywords.items():
            for tag in tags:
                tag_lower = tag.lower()
                if any(keyword in tag_lower for keyword in keywords):
                    return category
        
        # Default to first tag if no category match
        if tags:
            return tags[0].lower().replace(" ", "_")
        
        return "general" 