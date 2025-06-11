"""
NL2Bash dataset preparer.

Transforms raw NL2Bash data into standardized Run objects representing
human baseline performance, adhering to METR schema.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from human_ttc_eval.core.prepare import Prepare
from human_ttc_eval.core.registry import register_preparer
from human_ttc_eval.core.run import Run
from human_ttc_eval import config
from human_ttc_eval.core.llm_utils import batch_process_parallel, estimate_batch_cost_realtime

logger = logging.getLogger(__name__)


@dataclass
class NL2BashTask:
    """Internal representation of an NL2Bash task during processing."""
    id: int
    nl_description: str
    bash_command: str
    word_count: int
    has_pipes: bool
    has_redirects: bool
    has_subcommands: bool
    complexity_score: float
    complexity_category: str
    utilities_used: List[str]
    estimated_time_seconds: float
    timing_source: str  # "human_provided" or "default"


@register_preparer("nl2bash")
class NL2BashPrepare(Prepare):
    """Prepares raw NL2Bash data into standardized Run objects."""
    
    def __init__(self):
        """Initialize the NL2Bash preparer."""
        super().__init__(dataset_name="nl2bash")
        self.atomic_only = False  # Could be made configurable
        self.sample_size = config.NL2BASH_SAMPLE_SIZE  # From config
        self.human_runs_path = config.PROJECT_ROOT / "data" / "keep" / "nl2bash" / "nl2bash_human_runs.jsonl"
        self.human_time_estimates = self._load_human_time_estimates()

    def _load_human_time_estimates(self) -> Dict[int, float]:
        """Loads human time estimates from a JSONL file."""
        estimates = {}
        if not self.human_runs_path.exists():
            raise FileNotFoundError(f"Human runs file not found: {self.human_runs_path}")
        
        with open(self.human_runs_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    task_id = data.get("task_id")
                    seconds = data.get("estimated_time_seconds")
                    if task_id is not None and seconds is not None:
                        estimates[int(task_id)] = float(seconds)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid line in {self.human_runs_path}: {line.strip()} - {e}")
        
        logger.info(f"Loaded {len(estimates)} human time estimates from {self.human_runs_path}")
        return estimates
        
    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        """
        Extract NL2Bash-specific metadata for task definitions.
        
        This metadata will be stored in the tasks.jsonl file and used
        by the benchmark harness.
        
        Args:
            representative_run: A Run object for the task
            
        Returns:
            Dictionary with NL2Bash-specific metadata
        """
        # Parse the task_id to get the original task number
        # Format: nl2bash_{complexity_category}/task_{id}
        task_id_parts = representative_run.task_id.split('/')
        if len(task_id_parts) == 2 and task_id_parts[1].startswith('task_'):
            original_id = int(task_id_parts[1].replace('task_', ''))
        else:
            original_id = -1
        
        # Find the original task data
        # This is a bit inefficient but works for now
        # In a real implementation, we might cache this mapping
        task_metadata = {
            "nl_description": "",
            "bash_command": "",
            "complexity_score": 0.0,
            "complexity_category": "",
            "word_count": 0,
            "has_pipes": False,
            "has_redirects": False,
            "has_subcommands": False,
            "utilities_used": [],
            "timing_source": "heuristic"
        }
        
        # We need to re-parse to get the metadata
        # This is inefficient but maintains separation of concerns
        try:
            nl_file = self.raw_data_dir / "all.nl"
            cm_file = self.raw_data_dir / "all.cm"
            
            if nl_file.exists() and cm_file.exists():
                with open(nl_file, 'r', encoding='utf-8') as nf, \
                     open(cm_file, 'r', encoding='utf-8') as cf:
                    for idx, (nl_line, cm_line) in enumerate(zip(nf, cf)):
                        if idx == original_id:
                            task = self._analyze_command(idx, nl_line.strip(), cm_line.strip())
                            task_metadata = {
                                "nl_description": task.nl_description,
                                "bash_command": task.bash_command,
                                "complexity_score": task.complexity_score,
                                "complexity_category": task.complexity_category,
                                "word_count": task.word_count,
                                "has_pipes": task.has_pipes,
                                "has_redirects": task.has_redirects,
                                "has_subcommands": task.has_subcommands,
                                "utilities_used": task.utilities_used,
                                "timing_source": task.timing_source
                            }
                            break
        except Exception as e:
            logger.warning(f"Could not extract metadata for task {representative_run.task_id}: {e}")
        
        return task_metadata
    
    def prepare(self) -> List[Run]:
        """
        Load raw NL2Bash data and transform into Run objects.
        
        Returns:
            List of Run objects representing human baseline performance
        """
        logger.info("Starting NL2Bash dataset preparation")
        
        # Load raw files
        nl_file = self.raw_data_dir / "all.nl"
        cm_file = self.raw_data_dir / "all.cm"
        
        if not nl_file.exists() or not cm_file.exists():
            logger.error(f"Raw data files not found in {self.raw_data_dir}")
            return []
        
        # Process tasks
        tasks = self._load_and_analyze_tasks(nl_file, cm_file)
        
        if not tasks:
            logger.warning("No tasks loaded from raw data")
            return []
        
        logger.info(f"Analyzed {len(tasks)} tasks")
        
        # Apply sampling if requested
        if self.sample_size and len(tasks) > self.sample_size:
            original_count = len(tasks)
            tasks = self._stratified_sample_by_complexity(tasks)
            logger.info(f"Stratified sampled {len(tasks)} tasks from original {original_count} total")
        
        # Convert to Run objects
        runs = []
        for task in tasks:
            run = self._task_to_run(task)
            if run:
                runs.append(run)
        
        logger.info(f"Created {len(runs)} Run objects")
        return runs
    
    def _load_and_analyze_tasks(self, nl_file: Path, cm_file: Path) -> List[NL2BashTask]:
        """Load and analyze all tasks from the raw files."""
        tasks = []
        
        with open(nl_file, 'r', encoding='utf-8') as nf, \
             open(cm_file, 'r', encoding='utf-8') as cf:
            
            for task_id, (nl_line, cm_line) in enumerate(zip(nf, cf)):
                nl_description = nl_line.strip()
                bash_command = cm_line.strip()
                
                # Skip empty lines
                if not nl_description or not bash_command:
                    continue
                
                # Apply atomic filter if requested
                if self.atomic_only and not self._is_atomic_command(bash_command):
                    continue
                
                # Analyze the command
                task = self._analyze_command(task_id, nl_description, bash_command)
                tasks.append(task)
        
        return tasks
    
    def _analyze_command(self, task_id: int, nl_description: str, bash_command: str) -> NL2BashTask:
        """Analyze a bash command and create a task object."""
        word_count = len(bash_command.split())
        has_pipes = '|' in bash_command
        has_redirects = any(op in bash_command for op in ['>', '<', '>>', '<<'])
        has_subcommands = any(op in bash_command for op in ['$(', '`', '('])
        
        complexity_score = self._calculate_complexity_score(bash_command)
        complexity_category = self._get_complexity_category(complexity_score)
        utilities_used = self._extract_utilities(bash_command)
        
        # Get pre-calculated time from loaded estimates, with a default fallback
        estimated_time_seconds = self.human_time_estimates.get(task_id, 30.0)
        timing_source = "human_provided" if task_id in self.human_time_estimates else "default"
        
        return NL2BashTask(
            id=task_id,
            nl_description=nl_description,
            bash_command=bash_command,
            word_count=word_count,
            has_pipes=has_pipes,
            has_redirects=has_redirects,
            has_subcommands=has_subcommands,
            complexity_score=complexity_score,
            complexity_category=complexity_category,
            utilities_used=utilities_used,
            estimated_time_seconds=estimated_time_seconds,
            timing_source=timing_source
        )
    
    def _extract_utilities(self, command: str) -> List[str]:
        """Extract the main utilities/commands used in a bash command."""
        segments = re.split(r'[|;&]', command)
        utilities = []
        
        for segment in segments:
            segment = segment.strip()
            if segment:
                # Handle command substitution
                segment = re.sub(r'\$\([^)]+\)', '', segment)
                segment = re.sub(r'`[^`]+`', '', segment)
                
                words = segment.split()
                if words:
                    cmd = words[0]
                    # Skip common prefixes
                    if cmd in ['sudo', 'env', 'nohup', 'time'] and len(words) > 1:
                        cmd = words[1]
                    utilities.append(cmd)
        
        return list(set(utilities))
    
    def _calculate_complexity_score(self, command: str) -> float:
        """Calculate a complexity score for a bash command."""
        score = 1.0  # Base score
        
        # Add points for various complexity indicators
        score += command.count('|') * 1.5  # Pipes
        score += command.count(';') * 1.0  # Command separation
        score += command.count('&&') * 1.2  # Logical AND
        score += command.count('||') * 1.2  # Logical OR
        score += command.count('$(') * 1.3  # Command substitution
        score += command.count('`') * 1.3   # Backtick substitution
        score += command.count('{') * 1.1   # Brace expansion
        score += command.count('[') * 1.1   # Bracket expressions
        score += command.count('>') * 0.8   # Redirects
        score += command.count('<') * 0.8   # Input redirects
        
        # Add points for complex utilities
        complex_utilities = ['awk', 'sed', 'grep', 'find', 'xargs', 'sort', 'uniq', 'perl', 'cut']
        for util in complex_utilities:
            if re.search(r'\b' + util + r'\b', command):
                score += 1.0
        
        # Normalize by word count to get relative complexity
        word_count = len(command.split())
        if word_count > 0:
            score = score / (word_count ** 0.5)
        
        return round(score, 2)
    
    def _is_atomic_command(self, command: str) -> bool:
        """Check if a command is 'atomic' (simple, no pipes/redirects)."""
        # No pipes, redirects, or command chaining
        if any(op in command for op in ['|', '>', '<', '>>', '<<', ';', '&&', '||', '$(', '`', '(', ')']):
            return False
        
        # Limit to common utilities
        common_utils = {
            'ls', 'cd', 'pwd', 'mkdir', 'rmdir', 'rm', 'cp', 'mv',
            'cat', 'head', 'tail', 'wc', 'grep', 'find', 'sort',
            'cut', 'awk', 'sed', 'chmod', 'chown', 'tar', 'gzip',
            'echo', 'printf', 'date', 'who', 'ps', 'top', 'kill',
            'df', 'du', 'mount', 'umount', 'history', 'alias',
            'which', 'whereis', 'locate', 'file', 'stat', 'touch',
            'ln', 'readlink', 'basename', 'dirname', 'realpath'
        }
        
        utilities = self._extract_utilities(command)
        return bool(utilities) and utilities[0] in common_utils
    
    def _get_complexity_category(self, complexity_score: float) -> str:
        """Categorize tasks by complexity score."""
        if complexity_score <= 1.5:
            return "very_simple"
        elif complexity_score <= 3.0:
            return "simple"
        elif complexity_score <= 5.0:
            return "medium"
        elif complexity_score <= 8.0:
            return "complex"
        else:
            return "very_complex"
    
    def _task_to_run(self, task: NL2BashTask) -> Optional[Run]:
        """Convert an NL2BashTask to a METR-compliant Run object."""
        try:
            # Create task_id and task_family based on complexity
            task_family = f"nl2bash_{task.complexity_category}"
            task_id = f"{task_family}/task_{task.id}"
            run_id = f"human_{task_id.replace('/', '_')}_{task.timing_source}"
            
            # Convert time to minutes
            human_minutes = task.estimated_time_seconds / 60.0
            
            # Create Run object with METR-compliant fields only
            run = Run(
                task_id=task_id,
                task_family=task_family,
                run_id=run_id,
                alias="Human Baseline (NL2Bash)",
                model="human",
                score_binarized=1,  # All human baselines assumed successful
                score_cont=1.0,
                human_minutes=human_minutes,
                human_source=f"nl2bash_{task.timing_source}_estimates",
                task_source="nl2bash_dataset",
                started_at=0.0,
                completed_at=task.estimated_time_seconds,
                generation_cost=0.0,
                human_cost=None,  # Could calculate based on human_minutes
                fatal_error_from=None
            )
            
            return run
            
        except Exception as e:
            logger.error(f"Error converting task {task.id} to Run: {e}")
            return None
    
    def _stratified_sample_by_complexity(self, tasks: List[NL2BashTask]) -> List[NL2BashTask]:
        """Apply stratified sampling to get even distribution across complexity categories."""
        import random
        random.seed(42)  # For reproducibility
        
        # Group tasks by complexity category
        tasks_by_category = {}
        for task in tasks:
            category = task.complexity_category
            if category not in tasks_by_category:
                tasks_by_category[category] = []
            tasks_by_category[category].append(task)
        
        # Log distribution before sampling
        logger.info("Original task distribution:")
        for category, task_list in tasks_by_category.items():
            logger.info(f"  {category}: {len(task_list)} tasks")
        
        # Calculate target per category for even distribution
        num_categories = len(tasks_by_category)
        target_per_category = self.sample_size // num_categories
        remainder = self.sample_size % num_categories
        
        sampled_tasks = []
        categories = sorted(tasks_by_category.keys())  # Sort for deterministic order
        
        for i, category in enumerate(categories):
            available_tasks = tasks_by_category[category]
            
            # Give remainder to first few categories
            current_target = target_per_category + (1 if i < remainder else 0)
            
            # Sample up to target, but don't exceed available
            sample_count = min(current_target, len(available_tasks))
            
            if sample_count > 0:
                sampled = random.sample(available_tasks, sample_count)
                sampled_tasks.extend(sampled)
                logger.info(f"  Sampled {sample_count} tasks from {category} (target: {current_target})")
        
        logger.info(f"Final sampled distribution: {len(sampled_tasks)} total tasks")
        return sampled_tasks 