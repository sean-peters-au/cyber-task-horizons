"""Parse NL2Bash dataset into METR standardized format."""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ...core.base_parser import BaseParser
from ...core.registry import register_parser
from ...config import (
    ENABLE_LLM_TIMING, 
    NL2BASH_LLM_PROVIDER, 
    NL2BASH_LLM_MODEL, 
    NL2BASH_BATCH_SIZE
)

logger = logging.getLogger(__name__)

@dataclass
class NL2BashTask:
    """A single NL2Bash task."""
    id: int
    nl_description: str
    bash_command: str
    word_count: int
    has_pipes: bool
    has_redirects: bool
    has_subcommands: bool
    complexity_score: float
    utilities_used: List[str]
    estimated_time_seconds: float
    timing_source: str  # "heuristic" or "llm"

@register_parser("nl2bash")
class NL2BashParser(BaseParser):
    """
    Parser that converts NL2Bash dataset into METR all_runs.jsonl format.
    
    Each NL2Bash task becomes a "human run" record with estimated completion time
    representing how long it would take a competent user to complete the task.
    """

    @property
    def dataset_name(self) -> str:
        return "nl2bash"

    def __init__(self, input_dir: Path, output_file: Path, 
                 atomic_only: bool = False, use_llm_timing: bool = None,
                 sample_size: Optional[int] = 100, random_seed: int = 42):
        """
        Args:
            input_dir: Directory containing NL2Bash data (not used directly, data comes from third-party)
            output_file: Output path for the all_runs.jsonl file
            atomic_only: If True, only include atomic (simple) commands
            use_llm_timing: If True, use LLM for time estimation. If None, use global setting.
            sample_size: If set, randomly sample this many tasks. If None, use all tasks.
            random_seed: Seed for reproducible random sampling
        """
        super().__init__(input_dir, output_file)
        self.atomic_only = atomic_only
        self.use_llm_timing = use_llm_timing if use_llm_timing is not None else ENABLE_LLM_TIMING
        self.sample_size = sample_size
        self.random_seed = random_seed
        logger.info(f"NL2BashParser initialized. Atomic only: {atomic_only}, LLM timing: {self.use_llm_timing}, Sample size: {sample_size}, Seed: {random_seed}")

    def _get_data_files(self) -> tuple[Path, Path]:
        """Get paths to the NL2Bash data files.
        
        Returns:
            Tuple of (nl_file_path, cm_file_path)
        """
        # NL2Bash data is in third-party/nl2bash/data/bash/
        project_root = Path(__file__).parent.parent.parent.parent.parent
        repo_path = project_root / "third-party" / "nl2bash"
        data_path = repo_path / "data" / "bash"
        
        nl_file = data_path / "all.nl"
        cm_file = data_path / "all.cm"
        
        if not nl_file.exists() or not cm_file.exists():
            raise FileNotFoundError(f"NL2Bash data files not found. Expected: {nl_file}, {cm_file}")
        
        return nl_file, cm_file

    def parse(self) -> List[Dict[str, Any]]:
        """
        Parse NL2Bash dataset into METR all_runs.jsonl format.
        
        Returns:
            List of run records in METR format
        """
        logger.info("Starting NL2Bash dataset parsing")
        
        try:
            # Load and analyze all tasks
            tasks = self._load_and_analyze_tasks()
            logger.info(f"Loaded {len(tasks)} tasks")
            
            # Apply LLM timing if enabled
            if self.use_llm_timing and tasks:
                self._apply_llm_timing(tasks)
            
            # Convert to METR format
            all_runs = []
            for task in tasks:
                run_record = self._convert_task_to_run(task)
                if run_record:
                    all_runs.append(run_record)
            
            logger.info(f"Successfully converted {len(all_runs)} tasks to METR format")
            return all_runs
            
        except Exception as e:
            logger.error(f"Error parsing NL2Bash dataset: {e}", exc_info=True)
            return []

    def _load_and_analyze_tasks(self) -> List[NL2BashTask]:
        """Load NL2Bash data files and create analyzed task objects."""
        nl_file, cm_file = self._get_data_files()
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
        
        # Apply stratified sampling by complexity if requested
        if self.sample_size is not None and len(tasks) > self.sample_size:
            tasks = self._stratified_sample_by_complexity(tasks)
        
        return tasks

    def _stratified_sample_by_complexity(self, tasks: List[NL2BashTask]) -> List[NL2BashTask]:
        """Perform stratified sampling by complexity groups with balanced representation."""
        import random
        from collections import defaultdict
        
        random.seed(self.random_seed)
        original_count = len(tasks)
        
        # Group tasks by complexity category
        complexity_groups = defaultdict(list)
        for task in tasks:
            category = self._get_complexity_category(task.complexity_score)
            complexity_groups[category].append(task)
        
        # Sort categories for consistent ordering
        categories = sorted(complexity_groups.keys())
        num_categories = len(categories)
        
        # Calculate more balanced sample sizes
        # Give each category a base allocation, then distribute remainder proportionally
        base_per_category = max(5, self.sample_size // (num_categories * 2))  # At least 5 per category
        base_total = base_per_category * num_categories
        remainder = max(0, self.sample_size - base_total)
        
        sampled_tasks = []
        group_info = []
        
        # First pass: give each category the base allocation
        for category in categories:
            group_tasks = complexity_groups[category]
            group_size = len(group_tasks)
            
            # Take base allocation (or all tasks if group is smaller)
            sample_size = min(base_per_category, group_size)
            
            if sample_size >= group_size:
                group_sample = group_tasks
            else:
                group_sample = random.sample(group_tasks, sample_size)
            
            sampled_tasks.extend(group_sample)
            group_info.append((category, len(group_sample), group_size))
        
        # Second pass: distribute remainder proportionally among groups with remaining tasks
        if remainder > 0:
            # Calculate how many more tasks each group can contribute
            available_groups = []
            for category, sampled_count, total_count in group_info:
                remaining = total_count - sampled_count
                if remaining > 0:
                    # Weight by log of group size to balance between common and rare groups
                    import math
                    weight = math.log(remaining + 1)
                    available_groups.append((category, remaining, weight))
            
            if available_groups:
                # Distribute remainder based on weighted availability
                total_weight = sum(weight for _, _, weight in available_groups)
                
                for category, remaining, weight in available_groups:
                    if remainder <= 0:
                        break
                    
                    # Calculate additional samples for this group
                    proportion = weight / total_weight
                    additional = min(remaining, max(1, int(remainder * proportion)))
                    
                    if additional > 0:
                        group_tasks = complexity_groups[category]
                        current_sample = [t for t in sampled_tasks if self._get_complexity_category(t.complexity_score) == category]
                        available_tasks = [t for t in group_tasks if t not in current_sample]
                        
                        if available_tasks:
                            extra_sample = random.sample(available_tasks, min(additional, len(available_tasks)))
                            sampled_tasks.extend(extra_sample)
                            remainder -= len(extra_sample)
                            
                            # Update group_info
                            for i, (cat, count, total) in enumerate(group_info):
                                if cat == category:
                                    group_info[i] = (cat, count + len(extra_sample), total)
                                    break
        
        # Shuffle the final result to mix complexity groups
        random.shuffle(sampled_tasks)
        
        # Format group info for logging
        group_info_str = ', '.join(f"{cat}: {count}/{total}" for cat, count, total in group_info)
        
        logger.info(f"Balanced stratified sampling: {len(sampled_tasks)} tasks from {original_count}")
        logger.info(f"Complexity distribution: {group_info_str} (seed: {self.random_seed})")
        
        return sampled_tasks

    def _analyze_command(self, task_id: int, nl_description: str, bash_command: str) -> NL2BashTask:
        """Analyze a bash command and create a task object."""
        word_count = len(bash_command.split())
        has_pipes = '|' in bash_command
        has_redirects = any(op in bash_command for op in ['>', '<', '>>', '<<'])
        has_subcommands = any(op in bash_command for op in ['$(', '`', '('])
        
        complexity_score = self._calculate_complexity_score(bash_command)
        utilities_used = self._extract_utilities(bash_command)
        estimated_time_seconds = self._estimate_time_heuristic(
            complexity_score, word_count, has_pipes, has_redirects, has_subcommands, utilities_used
        )
        
        return NL2BashTask(
            id=task_id,
            nl_description=nl_description,
            bash_command=bash_command,
            word_count=word_count,
            has_pipes=has_pipes,
            has_redirects=has_redirects,
            has_subcommands=has_subcommands,
            complexity_score=complexity_score,
            utilities_used=utilities_used,
            estimated_time_seconds=estimated_time_seconds,
            timing_source="heuristic"
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
        complex_utilities = ['awk', 'sed', 'grep', 'find', 'xargs', 'sort', 'uniq']
        for util in complex_utilities:
            if util in command:
                score += 1.0
        
        # Normalize by word count to get relative complexity
        word_count = len(command.split())
        if word_count > 0:
            score = score / (word_count ** 0.5)
        
        return round(score, 2)

    def _estimate_time_heuristic(self, complexity_score: float, word_count: int, 
                                has_pipes: bool, has_redirects: bool, has_subcommands: bool,
                                utilities_used: List[str]) -> float:
        """Estimate completion time using heuristics."""
        base_time = 10.0  # 10 seconds minimum
        
        # Add time based on complexity score
        complexity_time = complexity_score * 5.0
        
        # Add time for word count (reading/typing)
        word_time = word_count * 1.5
        
        # Add time for specific features
        if has_pipes:
            base_time += 15.0
        if has_redirects:
            base_time += 10.0
        if has_subcommands:
            base_time += 20.0
        
        # Add time for complex utilities
        complex_utils = {'awk', 'sed', 'find', 'xargs', 'grep'}
        for util in utilities_used:
            if util in complex_utils:
                base_time += 10.0
        
        total_time = base_time + complexity_time + word_time
        
        # Cap at reasonable maximum (5 minutes)
        return min(total_time, 300.0)

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

    def _apply_llm_timing(self, tasks: List[NL2BashTask]) -> None:
        """Apply LLM timing estimates to tasks."""
        try:
            from ...core.llm_utils import batch_process_parallel, estimate_batch_cost_realtime
            
            # Show cost estimate
            cost_info = estimate_batch_cost_realtime(
                len(tasks), NL2BASH_BATCH_SIZE, NL2BASH_LLM_PROVIDER, NL2BASH_LLM_MODEL
            )
            
            logger.info(f"LLM timing cost estimate: ${cost_info['estimated_total_cost']:.2f} "
                       f"({cost_info['num_batches']} batches)")
            
            # Ask for user confirmation
            print(f"LLM time estimation will cost approximately ${cost_info['estimated_total_cost']:.2f}")
            print(f"Provider: {cost_info['provider']}, Model: {cost_info['model']}")
            response = input("Proceed with LLM timing? (y/n): ")
            
            if response.lower() != 'y':
                logger.info("LLM timing cancelled by user")
                return
            
            # Process in batches
            def process_batch(batch: List[NL2BashTask]) -> List[Dict[int, float]]:
                return [self._estimate_batch_with_llm(batch)]
            
            def progress_callback(current: int, total: int):
                logger.info(f"Processed LLM batch {current}/{total}")
            
            logger.info(f"Starting LLM timing estimation for {len(tasks)} tasks")
            batch_results = batch_process_parallel(
                tasks, process_batch, NL2BASH_BATCH_SIZE, max_workers=4, progress_callback=progress_callback
            )
            
            # Apply LLM estimates
            all_estimates = {}
            for batch_result in batch_results:
                all_estimates.update(batch_result)
            
            for task in tasks:
                if task.id in all_estimates:
                    task.estimated_time_seconds = all_estimates[task.id]
                    task.timing_source = "llm"
            
            logger.info(f"Applied LLM timing to {len(all_estimates)} tasks")
            
        except ImportError:
            logger.warning("LLM utilities not available, skipping LLM timing")
        except Exception as e:
            logger.warning(f"LLM timing failed: {e}")

    def _estimate_batch_with_llm(self, tasks: List[NL2BashTask]) -> Dict[int, float]:
        """Estimate timing for a batch of tasks using LLM."""
        try:
            from ...core.llm_utils import LLMClient, LLMConfig
            
            config = LLMConfig(
                provider=NL2BASH_LLM_PROVIDER,
                model=NL2BASH_LLM_MODEL,
                max_tokens=4096,
                temperature=0.1
            )
            
            client = LLMClient(config)
            system_prompt, user_prompt = self._create_timing_prompt(tasks)
            response = client.call(user_prompt, system_prompt)
            
            # Parse response
            estimates = json.loads(response)
            result = {}
            for estimate in estimates:
                task_id = estimate['id']
                seconds = max(5.0, min(600.0, float(estimate['seconds'])))  # Sanity bounds
                result[task_id] = seconds
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM batch estimation failed: {e}")
            return {}

    def _create_timing_prompt(self, tasks: List[NL2BashTask]) -> tuple[str, str]:
        """Create prompts for LLM timing estimation."""
        system_prompt = """You are an expert Linux system administrator. Estimate how long it would take a competent Linux user to complete each bash command task.

Consider: reading the description, thinking about the command, typing it (including trial and error), and command familiarity.

Return ONLY a JSON array: [{"id": 1, "seconds": 25}, {"id": 2, "seconds": 45}, ...]"""

        user_prompt = "Estimate completion times for these bash tasks:\n\n"
        for task in tasks:
            user_prompt += f"ID {task.id}: {task.nl_description}\n"
            user_prompt += f"Command: {task.bash_command}\n"
            user_prompt += f"Complexity: {task.complexity_score}\n\n"
        
        user_prompt += f"\nReturn JSON array with {len(tasks)} estimates in seconds."
        return system_prompt, user_prompt

    def _convert_task_to_run(self, task: NL2BashTask) -> Dict[str, Any]:
        """Convert a NL2BashTask to METR all_runs format."""
        try:
            complexity_category = self._get_complexity_category(task.complexity_score)
            task_family = f"nl2bash_{complexity_category}"
            task_id = f"{task_family}/task_{task.id}"
            run_id = f"human_nl2bash_{complexity_category}_task_{task.id}_{task.timing_source}"
            
            human_minutes = task.estimated_time_seconds / 60.0
            
            return {
                "task_id": task_id,
                "task_family": task_family,
                "run_id": run_id,
                "alias": f"Human (NL2Bash {task.timing_source.upper()})",
                "model": "human",
                "score_binarized": 1,  # All tasks are solvable by definition
                "score_cont": 1.0,
                "fatal_error_from": None,
                "human_minutes": human_minutes,
                "human_score": 1.0,
                "human_source": f"nl2bash_{task.timing_source}_estimates",
                "task_source": "nl2bash_dataset",
                "generation_cost": 0.0,
                "human_cost": None,
                "time_limit": None,
                "started_at": None,
                "completed_at": None,
                "task_version": "1.0",
                "equal_task_weight": None,
                "invsqrt_task_weight": None,
                
                # NL2Bash-specific metadata
                "nl_description": task.nl_description,
                "bash_command": task.bash_command,
                "word_count": task.word_count,
                "has_pipes": task.has_pipes,
                "has_redirects": task.has_redirects,
                "has_subcommands": task.has_subcommands,
                "complexity_score": task.complexity_score,
                "utilities_used": task.utilities_used,
                "timing_source": task.timing_source,
                
                # Raw task reference
                "_raw_task_id": task.id,
                "_raw_complexity_category": complexity_category
            }
            
        except Exception as e:
            logger.error(f"Error converting task {task.id} to run format: {e}", exc_info=True)
            return None

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

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("data/processed/nl2bash")
    
    # Ask user about LLM timing
    print(f"NL2Bash Parser Configuration:")
    print(f"  LLM Provider: {NL2BASH_LLM_PROVIDER}")
    print(f"  LLM Model: {NL2BASH_LLM_MODEL}")
    print(f"  Batch Size: {NL2BASH_BATCH_SIZE}")
    print(f"  LLM Timing Enabled: {ENABLE_LLM_TIMING}")
    print()
    
    parser = NL2BashParser(Path("data/raw/nl2bash"), output_dir / "all_runs.jsonl")
    all_runs = parser.parse()
    
    print("NL2Bash dataset parsing completed:")
    for run in all_runs:
        print(f"  {run['task_id']}: {run['nl_description']}")
        print(f"    → {run['bash_command']}")
        print(f"    (complexity: {run['complexity_score']}, time: {run['human_minutes']:.2f}m [{run['timing_source']}], utils: {run['utilities_used']})")
        print()
    
    print(f"\nAll runs saved to {output_dir}") 