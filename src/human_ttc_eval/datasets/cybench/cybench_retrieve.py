import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from human_ttc_eval.core.registry import register_retriever
from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

logger = logging.getLogger(__name__)

@register_retriever("cybench")
class CybenchRetrieve(Retrieve):
    """Retrieves and processes raw data from the CyBench dataset repository."""
    
    DEFAULT_OUTPUT_FILENAME = "cybench_raw_data.jsonl"

    def __init__(self, dataset_name: str = "cybench", output_filename: Optional[str] = None):
        """
        Initializes the CybenchRetrieve.

        Args:
            dataset_name: Name of the dataset, used to determine raw data directory.
                          Defaults to "cybench".
            output_filename: Optional custom name for the output JSONL file.
                             Defaults to "cybench_raw_data.jsonl".
        """
        super().__init__(dataset_name) # Initializes self.dataset_name and self.output_dir
        
        self.cybench_repo_path = config.CYBENCH_REPO_PATH
        self.output_filename = output_filename or self.DEFAULT_OUTPUT_FILENAME

        if not self.cybench_repo_path.exists():
            logger.warning(f"CyBench repository path does not exist: {self.cybench_repo_path}")
            logger.warning("Please ensure the CyBench submodule is initialized or the path is correctly set in config.py.")
            # Depending on strictness, could raise an error here or allow to proceed (retrieve might fail)

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
            logger.warning(f"Unexpected time format (wrong number of parts): {time_str}")
            return None
        except ValueError:
            logger.warning(f"Could not parse time string to integers: {time_str}")
            return None

    def _parse_readme_data(self, readme_path: Path) -> Dict[str, Any]:
        """Parses relevant data from a challenge's README.md file."""
        data = {}
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if line.startswith("> Author:"):
                        data["author"] = line.split(":", 1)[1].strip()
                    elif line.startswith("> Difficulty:"):
                        data["difficulty"] = line.split(":", 1)[1].strip()
                    elif line.startswith("> Points:"):
                        try:
                            data["points"] = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            logger.warning(f"Could not parse points from line: {line} in {readme_path}")

                    if "author" in data and i + 1 < len(lines) and lines[i+1].strip() == "": # Look for content after metadata
                        desc_lines = []
                        for j in range(i + 2, len(lines)):
                            if lines[j].startswith("#") or lines[j].startswith(">") or not lines[j].strip(): # Heuristic stop conditions
                                break
                            desc_lines.append(lines[j].strip())
                        if desc_lines:
                            data["description"] = " ".join(desc_lines).strip()
                            break # Found description after metadata

                # Fallback for description if not found after metadata block
                if "description" not in data and lines:
                    # Try to grab the first few non-empty, non-metadata lines as description
                    desc_lines = []
                    for line in lines:
                        l_strip = line.strip()
                        # Skip empty lines, headings, blockquotes, and table rows
                        if not l_strip or l_strip.startswith("#") or l_strip.startswith(">") or "|" in l_strip:
                            if desc_lines: # Stop if we have some lines and encounter metadata/empty/table
                                break
                            continue # Skip these lines if we haven't started collecting description
                        desc_lines.append(l_strip)
                        if len(desc_lines) >= 3: # Take up to 3 lines for a short desc
                            break
                    if desc_lines:
                        data["description"] = " ".join(desc_lines).strip()

            except Exception as e:
                logger.error(f"Error parsing README {readme_path}: {e}")
        return data

    def _extract_challenge_data(self, challenge_dir_str: str) -> Optional[Dict[str, Any]]:
        """Extracts comprehensive data for a single challenge."""
        if not self.cybench_repo_path.exists(): # Guard against missing repo
            logger.error(f"CyBench repo path {self.cybench_repo_path} not found. Cannot extract challenge data.")
            return None

        challenge_path = self.cybench_repo_path / challenge_dir_str
        if not challenge_path.is_dir():
            logger.warning(f"Challenge directory not found: {challenge_path}")
            return None

        readme_data = self._parse_readme_data(challenge_path / "README.md")
        
        # Task ID is the relative path from the repo root
        task_id = challenge_dir_str 
        
        # Derive category and name from path
        parts = Path(challenge_dir_str).parts
        category = parts[-2] if len(parts) > 1 else "unknown"
        name = parts[-1]

        return {
            "task_id": task_id,
            "name": name,
            "category": category,
            "event": parts[1] if len(parts) > 2 else "unknown", # e.g. hackthebox, project-sekai-ctf
            "ctf_name": parts[2] if len(parts) > 3 else "unknown", # e.g. cyber-apocalypse-2024
            "readme_data": readme_data,
            "raw_dir_path": str(challenge_path) # For reference
        }

    def retrieve(self) -> Optional[Path]:
        """Retrieves data from the CyBench repository and compiles it into a JSONL file."""
        if not self.cybench_repo_path.exists():
            logger.error(f"CyBench repository not found at {self.cybench_repo_path}. Cannot retrieve data.")
            return None

        task_list_path = self.cybench_repo_path / "task_list.txt"
        if not task_list_path.exists():
            logger.error(f"task_list.txt not found in CyBench repository: {task_list_path}")
            return None

        fastest_solve_times_path = self.cybench_repo_path / "analytics" / "CTF fastest solve times.json"
        fst_data = {}
        if fastest_solve_times_path.exists():
            try:
                with open(fastest_solve_times_path, 'r', encoding='utf-8') as f:
                    fst_list = json.load(f)
                    fst_data = {item["challenge_dir"]: item for item in fst_list}
            except Exception as e:
                logger.warning(f"Could not load or parse fastest solve times from {fastest_solve_times_path}: {e}")
        else:
            logger.info(f"Fastest solve times file not found: {fastest_solve_times_path}. Proceeding without it.")

        all_challenge_data = []
        try:
            with open(task_list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    challenge_dir_str = line.strip()
                    if not challenge_dir_str or challenge_dir_str.startswith("#"):
                        continue
                    
                    challenge_data = self._extract_challenge_data(challenge_dir_str)
                    if challenge_data:
                        # Add fastest solve time if available
                        fst_entry = fst_data.get(challenge_dir_str)
                        if fst_entry:
                            challenge_data["fastest_solve_time_str"] = fst_entry.get("challenge_fastest_solve_time")
                            challenge_data["fastest_solve_time_seconds"] = self._parse_time_to_seconds(fst_entry.get("challenge_fastest_solve_time"))
                        all_challenge_data.append(challenge_data)
        except IOError as e:
            logger.error(f"Error reading task list {task_list_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during challenge data processing: {e}", exc_info=True)
            return None

        if not all_challenge_data:
            logger.warning("No challenge data extracted. Output file will be empty or not created.")
            # Depending on desired behavior, could return None or an empty file path

        # self.output_dir is now set by the base Retrieve class
        output_file_path = self.output_dir / self.output_filename
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for item in all_challenge_data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Successfully retrieved {len(all_challenge_data)} CyBench challenges to {output_file_path}")
            return output_file_path
        except IOError as e:
            logger.error(f"Error writing CyBench raw data to {output_file_path}: {e}")
            return None 

    def cleanup(self) -> None:
        """No specific cleanup implemented for CyBench."""
        pass