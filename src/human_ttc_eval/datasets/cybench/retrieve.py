import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import re

from ...core.base_retriever import BaseRetriever
from ...core.registry import register_retriever

logger = logging.getLogger(__name__)

@register_retriever("cybench")
class CyBenchRetriever(BaseRetriever):
    """Retrieves and processes CyBench metadata from a local repository clone."""

    @property
    def dataset_name(self) -> str:
        return "cybench"

    def __init__(self, output_dir: Path, cybench_repo_path: Path):
        """
        Args:
            output_dir: Directory to save the processed metadata JSONL file.
            cybench_repo_path: Path to the locally cloned cybench repository.
        """
        super().__init__(output_dir)
        self.cybench_repo_path = Path(cybench_repo_path)
        if not self.cybench_repo_path.is_dir():
            # Log a warning but allow initialization; retrieve_metadata will fail more explicitly.
            logger.warning(f"CyBench repository path does not exist or is not a directory: {self.cybench_repo_path}")

    def retrieve_metadata(self, output_filename: str = "cybench_metadata.jsonl") -> Optional[Path]:
        """
        Processes task_list.txt and README.md files from the CyBench repo.
        Cross-references with CTF fastest solve times.json where available.
        """
        logger.info(f"Starting CyBench metadata retrieval from repo: {self.cybench_repo_path}")
        
        # Check for required files
        task_list_path = self.cybench_repo_path / "task_list.txt"
        fst_json_path = self.cybench_repo_path / "analytics" / "CTF fastest solve times.json"
        
        if not task_list_path.is_file():
            logger.error(f"task_list.txt not found at {task_list_path}")
            return None

        # Load fastest solve times for cross-reference (optional)
        fastest_times_data = {}
        if fst_json_path.is_file():
            try:
                with open(fst_json_path, 'r', encoding='utf-8') as f_json:
                    fst_data = json.load(f_json)
                if isinstance(fst_data, list):
                    for entry in fst_data:
                        if isinstance(entry, dict):
                            challenge_dir = entry.get("challenge_dir")
                            if challenge_dir:
                                fastest_times_data[challenge_dir] = entry
                    logger.info(f"Loaded {len(fastest_times_data)} fastest solve time entries")
                else:
                    logger.warning(f"Unexpected format in {fst_json_path}")
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load fastest solve times from {fst_json_path}: {e}")

        output_records: List[Dict[str, Any]] = []

        try:
            # Load task list
            with open(task_list_path, 'r', encoding='utf-8') as f:
                task_paths = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Found {len(task_paths)} tasks in task_list.txt")

            for task_num, task_path in enumerate(task_paths):
                logger.debug(f"Processing task {task_num+1}/{len(task_paths)}: {task_path}")
                
                # Convert task path to absolute path within repo
                task_full_path = self.cybench_repo_path / task_path
                
                if not task_full_path.is_dir():
                    logger.warning(f"Task directory not found: {task_full_path}. Skipping.")
                    continue

                # Extract metadata from directory structure and README
                task_metadata = self._extract_task_metadata(task_path, task_full_path)
                
                # Cross-reference with fastest solve times
                timing_data = fastest_times_data.get(task_path, {})
                if timing_data:
                    fastest_solve_time_str = timing_data.get("challenge_fastest_solve_time")
                    fastest_solve_time_seconds = self._parse_time_to_seconds(fastest_solve_time_str) if fastest_solve_time_str else None
                else:
                    fastest_solve_time_str = None
                    fastest_solve_time_seconds = None

                # Combine all metadata
                combined_record = {
                    "task_path_in_repo": task_path,
                    "title": task_metadata.get("title", task_path.split('/')[-1]),
                    "category": task_metadata.get("category"),
                    "difficulty": task_metadata.get("difficulty"),
                    "points": task_metadata.get("points"),
                    "description": task_metadata.get("description"),
                    "authors": task_metadata.get("authors"),
                    "tags": task_metadata.get("tags", []),
                    "organization": task_metadata.get("organization"),
                    "event": task_metadata.get("event"),
                    
                    # Timing data (if available)
                    "fastest_solve_time_str": fastest_solve_time_str,
                    "fastest_solve_time_seconds": fastest_solve_time_seconds,
                    "timing_source": "cybench_fst_times.json" if timing_data else None,
                    
                    # Raw data for reference
                    "task_metadata": task_metadata,
                    "timing_data": timing_data if timing_data else None,
                }
                output_records.append(combined_record)

        except IOError as e:
            logger.error(f"Error reading task_list.txt from {task_list_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during metadata processing: {e}", exc_info=True)
            return None

        if not output_records:
            logger.warning("No metadata records were processed. Output file will be empty or not created.")
            return None

        output_file_path = self.output_dir / output_filename
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                for record in output_records:
                    f_out.write(json.dumps(record) + '\n')
            logger.info(f"Successfully wrote {len(output_records)} CyBench metadata records to {output_file_path}")
            return output_file_path
        except IOError as e:
            logger.error(f"Error writing metadata to {output_file_path}: {e}")
            return None

    def _extract_task_metadata(self, task_path: str, task_full_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from task directory structure and README.md file.
        """
        metadata = {}
        
        # Parse organization and event from path structure
        # Expected: benchmark/{org}/{event}/{category}/{challenge_name}
        path_parts = task_path.split('/')
        if len(path_parts) >= 5:
            metadata["organization"] = path_parts[1]  # e.g., "hackthebox"
            metadata["event"] = path_parts[2]         # e.g., "cyber-apocalypse-2024"
            metadata["category"] = path_parts[3]      # e.g., "crypto"
            challenge_name = path_parts[4]            # e.g., "[Easy] Blunt"
            metadata["title"] = challenge_name
            
            # Extract difficulty from challenge name if present
            difficulty_match = re.search(r'\[(Very Easy|Easy|Medium|Hard|Insane)\]', challenge_name)
            if difficulty_match:
                metadata["difficulty"] = difficulty_match.group(1)
            
        # Try to read README.md for additional metadata
        readme_path = task_full_path / "README.md"
        if readme_path.is_file():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                    
                # Extract metadata from README content
                readme_metadata = self._parse_readme_metadata(readme_content)
                metadata.update(readme_metadata)
                
            except (IOError, UnicodeDecodeError) as e:
                logger.debug(f"Could not read README.md for {task_path}: {e}")

        # Check for other common metadata files
        for filename in ["challenge.yml", "challenge.yaml", "metadata.json"]:
            metadata_file = task_full_path / filename
            if metadata_file.is_file():
                try:
                    if filename.endswith('.json'):
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            file_metadata = json.load(f)
                            if isinstance(file_metadata, dict):
                                metadata.update(file_metadata)
                    # Could add YAML parsing here if needed
                except Exception as e:
                    logger.debug(f"Could not parse {filename} for {task_path}: {e}")

        return metadata

    def _parse_readme_metadata(self, readme_content: str) -> Dict[str, Any]:
        """
        Extract metadata from README.md content using common patterns.
        """
        metadata = {}
        
        # Look for common patterns in CTF README files
        lines = readme_content.split('\n')
        
        # Extract title from first heading
        for line in lines:
            if line.startswith('# '):
                metadata["title"] = line[2:].strip()
                break
        
        # Look for author information (avoid markdown table lines)
        author_patterns = [
            r'author[s]?[:\-\s]+(.+)',
            r'by[:\-\s]+(.+)',
            r'created by[:\-\s]+(.+)'
        ]
        for pattern in author_patterns:
            for line in lines:
                # Skip markdown table lines and headers
                if '|' in line or line.strip().startswith('-'):
                    continue
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    author_text = match.group(1).strip()
                    # Additional filtering to avoid capturing markdown artifacts
                    if not any(char in author_text for char in ['|', '-', '*', '#']):
                        metadata["authors"] = [author.strip() for author in author_text.split(',')]
                        break
        
        # Look for points/score
        points_patterns = [
            r'points?[:\-\s]+(\d+)',
            r'score[:\-\s]+(\d+)',
            r'value[:\-\s]+(\d+)'
        ]
        for pattern in points_patterns:
            for line in lines:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    metadata["points"] = int(match.group(1))
                    break
        
        # Look for difficulty patterns
        difficulty_patterns = [
            r'difficulty[:\-\s]+(easy|medium|hard|insane|very easy)',
            r'level[:\-\s]+(easy|medium|hard|insane|very easy)',
        ]
        for pattern in difficulty_patterns:
            for line in lines:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    metadata["difficulty"] = match.group(1).title()
                    break
        
        # Extract description (first paragraph after title, excluding tables)
        description_lines = []
        collecting_description = False
        for line in lines:
            line = line.strip()
            if line.startswith('# ') and not collecting_description:
                collecting_description = True
                continue
            elif collecting_description:
                if line.startswith('#'):  # Hit another heading
                    break
                elif '|' in line or line.startswith('-'):  # Skip table lines
                    continue
                elif line:
                    description_lines.append(line)
                elif description_lines:  # Empty line after some content
                    break
        
        if description_lines:
            # Clean up description text
            description = ' '.join(description_lines)
            # Remove markdown formatting
            description = re.sub(r'\*\*([^*]+)\*\*', r'\1', description)  # Bold
            description = re.sub(r'\*([^*]+)\*', r'\1', description)      # Italic
            description = re.sub(r'`([^`]+)`', r'\1', description)        # Code
            metadata["description"] = description.strip()
        
        return metadata

    def _parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        """
        Parse time strings like "0:05:07" (H:MM:SS) or "5:07" (M:SS) into seconds.
        """
        try:
            parts = time_str.split(':')
            if len(parts) == 3:  # H:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # M:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            else:
                logger.warning(f"Unexpected time format: {time_str}")
                return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse time '{time_str}': {e}")
            return None

    def download_challenge_content(self, pull_docker_images: bool = False, **kwargs) -> None:
        """
        Ensures the CyBench repository is accessible. Optionally, can attempt to pull Docker images.
        The primary "download" is having the repo cloned, which is passed during __init__.
        """
        logger.info(f"Checking CyBench content at specified repo path: {self.cybench_repo_path}")
        if not self.cybench_repo_path.is_dir() or not (self.cybench_repo_path / "task_list.txt").is_file():
            logger.error(f"CyBench repository at {self.cybench_repo_path} seems invalid or incomplete. Cannot proceed.")
            return

        logger.info(f"CyBench content (repository clone) assumed to be at {self.cybench_repo_path}.")

        if pull_docker_images:
            logger.info("Attempting to pull Docker images for CyBench... (This may take a long time)")
            pull_script_path = self.cybench_repo_path / "tools" / "pull_all_images.sh"
            if pull_script_path.is_file():
                try:
                    # This is a blocking call. Consider running in subprocess with monitoring if needed.
                    # For now, just logging the intent.
                    # subprocess.run(["bash", str(pull_script_path)], cwd=self.cybench_repo_path, check=True)
                    logger.warning(f"Actual execution of pull_all_images.sh is not implemented in this retriever directly. Please run it manually if Docker images are needed: cd {self.cybench_repo_path} && ./tools/pull_all_images.sh")
                except Exception as e:
                    logger.error(f"Error trying to (log intent for) pull_all_images.sh: {e}")
            else:
                logger.warning(f"pull_all_images.sh script not found at {pull_script_path}. Cannot pull images automatically.")
        else:
            logger.info("Pulling Docker images was not requested.")
