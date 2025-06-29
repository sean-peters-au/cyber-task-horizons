"""
Base class for dataset benchmarking.

Benches run AI models on tasks and produce Run objects that can be
compared against human baselines, maintaining METR schema compatibility.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timezone
import uuid
import json
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from inspect_ai.log import read_eval_log, list_eval_logs, EvalLog, EvalSample

from .run import Run

logger = logging.getLogger(__name__)


@dataclass
class BenchResult:
    """
    Result from a benchmark evaluation.
    
    Contains both individual Run objects (METR format) and aggregate statistics.
    """
    dataset_name: str
    model_name: str
    model_alias: str  # Display name for the model
    runs: List[Run]  # Individual run results in METR format
    summary_stats: Dict[str, Any]  # Aggregated statistics
    metadata: Dict[str, Any]  # Evaluation metadata (duration, params, etc.)
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "model_alias": self.model_alias,
            "runs": [run.to_jsonl_dict() for run in self.runs],
            "summary_stats": self.summary_stats,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "success": self.success,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchResult':
        """Create from dictionary."""
        runs = [Run.from_dict(run_data) for run_data in data["runs"]]
        return cls(
            dataset_name=data["dataset_name"],
            model_name=data["model_name"],
            model_alias=data["model_alias"],
            runs=runs,
            summary_stats=data["summary_stats"],
            metadata=data["metadata"],
            timestamp=data["timestamp"],
            success=data["success"],
            error_message=data.get("error_message"),
        )


class Bench(ABC):
    """
    Abstract base class for dataset benchmarking.
    
    Benches evaluate AI models on datasets and produce Run objects
    that conform to the METR schema. Results can be compared against
    human baselines loaded from processed data.
    
    Key principles:
    - Input: Task definitions and model configuration
    - Output: List[Run] strictly conforming to METR schema
    - Each task gets one run per model evaluation
    - Results are directly comparable to human baselines
    """
    
    def __init__(self, dataset_dir: Path, output_dir: Path):
        """
        Initialize the benchmark runner.
        
        Args:
            dataset_dir: Directory containing dataset files (typically data/raw/<dataset>/)
            output_dir: Directory to store evaluation results (typically results/benchmarks/<dataset>/)
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load human baseline for comparison
        self.human_baseline: List[Run] = []
        self._load_human_baseline()
        
        logger.info(f"Initialized {self.__class__.__name__} with dataset_dir: {self.dataset_dir}, output_dir: {self.output_dir}")
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        Returns the dataset identifier.
        
        This should match the name used in the CLI and directory structure.
        Examples: "nl2bash", "cybench"
        
        Returns:
            Dataset name as a slug (lowercase, no spaces)
        """
        pass
    
    def _load_human_baseline(self) -> None:
        """Load human baseline runs for comparison."""
        baseline_filename = f"{self.dataset_name}_human_runs.jsonl"
        baseline_file = self.dataset_dir.parent.parent / "processed" / self.dataset_name / baseline_filename
        if baseline_file.exists():
            self.human_baseline = Run.load_jsonl(str(baseline_file))
            # Filter to unique tasks only
            unique_tasks = {}
            for run in self.human_baseline:
                if run.task_id not in unique_tasks:
                    unique_tasks[run.task_id] = run
            self.human_baseline = list(unique_tasks.values())
            logger.info(f"Loaded {len(self.human_baseline)} human baseline tasks")
        else:
            logger.warning(f"No human baseline found at {baseline_file}")
    
    @abstractmethod
    def list_available_tasks(self) -> List[str]:
        """
        List all available task IDs for this dataset.
        
        Returns:
            List of task identifiers
        """
        pass
    
    @abstractmethod
    def run_evaluation(
        self,
        model_name: str,
        model_alias: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
        **kwargs
    ) -> BenchResult:
        """
        Run evaluation for the specified model and tasks.
        
        Args:
            model_name: Model identifier (e.g., "openai/gpt-4")
            model_alias: Display name for the model (defaults to model_name)
            task_ids: Optional list of specific tasks to run (None = all tasks)
            **kwargs: Additional evaluation parameters
            
        Returns:
            BenchResult with evaluation results
        """
        pass
    
    def _get_human_minutes_for_task(self, task_id: str) -> float:
        """Helper to find human_minutes for a given task_id from the loaded baseline."""
        for run in self.human_baseline:
            if run.task_id == task_id:
                return run.human_minutes
        logger.warning(f"No human baseline minutes found for task: {task_id} in {self.dataset_name}. Defaulting to 0.0.")
        return 0.0

    def _get_task_family_for_task(self, task_id: str) -> str:
        """Helper to find task_family for a given task_id from the loaded baseline, or default."""
        for run in self.human_baseline:
            if run.task_id == task_id:
                return run.task_family
        # Fallback if not in baseline (should be rare if baseline is comprehensive)
        # Basic heuristic: take first part of task_id or default to dataset_name
        parts = task_id.split('/')
        return parts[0] if len(parts) > 1 else self.dataset_name

    def _create_failed_run(self, task_id: str, model_name: str, model_alias: str, error: str) -> Run:
        """Create a Run object for a failed task."""
        human_minutes = self._get_human_minutes_for_task(task_id)
        task_family = self._get_task_family_for_task(task_id)
        
        # Ensure a unique run_id for this failed attempt
        # Using a UUID suffix to ensure uniqueness if called multiple times for same task/model
        unique_suffix = uuid.uuid4().hex[:8]
        run_id_str = f"{model_name.replace('/', '_')}_{task_id.replace('/', '_')}_failed_{unique_suffix}"

        return Run(
            task_id=task_id,
            task_family=task_family,
            run_id=run_id_str,
            alias=model_alias,
            model=model_name,
            score_binarized=0,
            score_cont=0.0,
            human_minutes=human_minutes,
            fatal_error_from=error[:1000],  # Truncate potentially very long errors
            human_source="baseline",
            task_source=self.dataset_name,
            generation_cost=0.0,
            started_at=0.0,
            completed_at=0.0,
        )
    
    def _create_zero_imputed_result(
        self, 
        model_name: str, 
        model_alias: str, 
        task_ids: List[str], 
        start_time: datetime,
        reason: str
    ) -> BenchResult:
        """
        Create a BenchResult with zero scores for all tasks.
        
        Used when a model is incompatible with benchmark requirements
        (e.g., tool-requiring benchmarks with models that don't support function calling).
        
        Args:
            model_name: Model identifier
            model_alias: Display name for the model
            task_ids: List of task IDs to create zero runs for
            start_time: Evaluation start time
            reason: Reason for zero imputation (logged and stored in metadata)
            
        Returns:
            BenchResult with zero-scored runs for all tasks
        """
        logger.warning(f"⚠️  ZERO IMPUTATION: {reason}")
        logger.warning(f"   Model: {model_name}")
        logger.warning(f"   Dataset: {self.dataset_name}")
        logger.warning(f"   Tasks affected: {len(task_ids)}")
        
        runs = []
        for task_id in task_ids:
            human_minutes = self._get_human_minutes_for_task(task_id)
            task_family = self._get_task_family_for_task(task_id)
            
            unique_suffix = uuid.uuid4().hex[:8]
            run_id_str = f"{model_name.replace('/', '_')}_{task_id.replace('/', '_')}_imputed_{unique_suffix}"
            
            run = Run(
                task_id=task_id,
                task_family=task_family,
                run_id=run_id_str,
                alias=model_alias,
                model=model_name,
                score_binarized=0,
                score_cont=0.0,
                human_minutes=human_minutes,
                fatal_error_from=f"Zero-imputed: {reason}",
                human_source="baseline",
                task_source=self.dataset_name,
                generation_cost=0.0,
                started_at=0.0,
                completed_at=0.0,
            )
            runs.append(run)
        
        # Calculate summary stats (will show 0% success rate)
        summary_stats = self._calculate_summary_stats(runs)
        summary_stats["zero_imputed"] = True
        summary_stats["imputation_reason"] = reason
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        return BenchResult(
            dataset_name=self.dataset_name,
            model_name=model_name,
            model_alias=model_alias,
            runs=runs,
            summary_stats=summary_stats,
            metadata={
                "zero_imputed": True,
                "imputation_reason": reason,
                "duration_seconds": duration,
                "num_tasks": len(task_ids),
            },
            timestamp=start_time.isoformat(),
            success=True,  # Success=True because imputation worked as intended
            error_message=None
        )
    
    def _calculate_summary_stats(self, runs: List[Run]) -> Dict[str, Any]:
        """Calculate summary statistics from runs."""
        if not runs:
            return {"error": "No runs to analyze"}
        
        successful_runs = [r for r in runs if r.score_binarized == 1]
        
        stats = {
            "total_tasks": len(runs),
            "successful_tasks": len(successful_runs),
            "success_rate": len(successful_runs) / len(runs) if runs else 0.0,
            "total_generation_cost": sum(r.generation_cost for r in runs),
            "mean_score_cont": sum(r.score_cont or 0 for r in runs) / len(runs) if runs else 0.0,
        }
        
        # Compare to human baseline if available
        if self.human_baseline:
            baseline_success = sum(1 for r in self.human_baseline if r.score_binarized == 1)
            stats["human_success_rate"] = baseline_success / len(self.human_baseline) if self.human_baseline else 0.0
            stats["relative_performance"] = stats["success_rate"] / stats["human_success_rate"] if stats["human_success_rate"] > 0 else 0.0
        
        return stats
    
    def save_result(self, result: BenchResult) -> Path:
        """
        Save benchmark result to JSON and JSONL formats.
        
        Args:
            result: BenchResult to save
            
        Returns:
            Path to saved result directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.output_dir / f"{result.model_name.replace('/', '_')}_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full result as JSON
        json_path = result_dir / "result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save runs in METR format JSONL
        jsonl_path = result_dir / "runs.jsonl"
        Run.save_jsonl(result.runs, str(jsonl_path))
        
        # Save summary statistics
        summary_path = result_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset": result.dataset_name,
                "model": result.model_name,
                "alias": result.model_alias,
                "timestamp": result.timestamp,
                "stats": result.summary_stats,
                "metadata": result.metadata,
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark result saved to: {result_dir}")
        return result_dir
    
    def load_result(self, result_path: Path) -> BenchResult:
        """
        Load a previously saved benchmark result.
        
        Args:
            result_path: Path to result directory or result.json file
            
        Returns:
            BenchResult object
        """
        if result_path.is_dir():
            result_path = result_path / "result.json"
            
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return BenchResult.from_dict(data)

    def find_existing_eval_log(self, model_name: str, log_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Find the most recent .eval log file for a given model.
        
        Args:
            model_name: The model identifier
            log_dir: Directory to search for logs (defaults to output_dir/inspect_logs)
            
        Returns:
            Path to the most recent .eval file, or None if not found
        """
        if log_dir is None:
            log_dir = self.output_dir / "inspect_logs"
        
        if not log_dir.exists():
            return None
            
        # Look for .eval files matching the model name
        eval_files = list(log_dir.glob(f"*{model_name.replace('/', '_')}*.eval"))
        
        if not eval_files:
            return None
            
        # Return the most recent file
        return max(eval_files, key=lambda p: p.stat().st_mtime)
    
    def find_all_eval_logs(self, model_name: str, log_dir: Optional[Path] = None) -> List[Path]:
        """
        Find all .eval log files for a given model using the inspect_ai API.
        
        Args:
            model_name: The model identifier
            log_dir: Directory to search for logs (defaults to output_dir/inspect_logs)
            
        Returns:
            List of paths to .eval files, sorted from oldest to newest
        """
        if log_dir is None:
            log_dir = self.output_dir / "inspect_logs"
        
        logger.info(f"Searching for .eval logs for model '{model_name}' in: {log_dir.resolve()}")

        if not log_dir.exists():
            logger.warning(f"Log directory not found: {log_dir.resolve()}")
            return []

        try:
            # Use inspect_ai's API to list logs
            all_logs = list_eval_logs(str(log_dir))
            logger.info(f"Found {len(all_logs)} total .eval files. Checking headers for model name...")

            matching_files = []
            for log_info in all_logs:
                try:
                    # Read only the header to efficiently check the model name
                    # Pass the EvalLogInfo object directly to read_eval_log
                    log_header = read_eval_log(log_info, header_only=True)
                    if log_header.eval and log_header.eval.model == model_name:
                        # The location is a file URI, parse it to get a Path
                        parsed_uri = urlparse(log_header.location)
                        matching_files.append(Path(parsed_uri.path))
                except Exception as e:
                    logger.warning(f"Could not read or parse header from log info object: {e}")

            if not matching_files:
                logger.warning(f"No log files found for model '{model_name}'.")
            else:
                logger.info(f"Found {len(matching_files)} log file(s) for model '{model_name}'.")

            # Sort by modification time (oldest first)
            return sorted(matching_files, key=lambda p: p.stat().st_mtime)

        except Exception as e:
            logger.error(f"Failed to list or read eval logs using inspect_ai API: {e}")
            return []
    
    def extract_completed_task_ids(self, eval_log_path: Path) -> Set[str]:
        """
        Extract completed task IDs from an inspect .eval log file.
        A task is considered "completed" if it either succeeded OR failed legitimately
        (i.e., the model got a chance to try but gave the wrong answer).
        Tasks with technical errors will be excluded so they can be re-run during resume.
        
        Args:
            eval_log_path: Path to the .eval log file
            
        Returns:
            Set of completed task IDs (successful or legitimately failed, but not errored)
        """
        completed_tasks = set()
        
        try:
            # Pass path as a string to be safe
            log = read_eval_log(str(eval_log_path))
            for sample in log.samples:
                # Check if this sample had a technical error
                has_error = False
                if hasattr(sample, 'error') and sample.error:
                    has_error = True
                elif hasattr(sample, 'messages') and sample.messages:
                    # Check if the last message indicates an error
                    last_message = sample.messages[-1] if sample.messages else None
                    if last_message and hasattr(last_message, 'role') and last_message.role == 'error':
                        has_error = True
                
                # If there was a technical error, don't mark as completed (so it gets re-run)
                if has_error:
                    continue
                
                # If no error, check if we have a valid score (success or legitimate failure)
                score_cont, score_binarized = self._extract_score_from_sample(sample)
                if sample.scores:  # Has some kind of score, whether success or failure
                    completed_tasks.add(sample.id)
                    
            logger.info(f"Found {len(completed_tasks)} completed tasks (successful or legitimately failed) in {eval_log_path}")
            
        except Exception as e:
            logger.error(f"Error reading eval log {eval_log_path} with inspect_ai: {e}")
            
        return completed_tasks
    
    def _extract_score_from_sample(self, sample: EvalSample) -> tuple[float, int]:
        """Extract score from an inspect_ai EvalSample object."""
        if not sample.scores:
            return 0.0, 0
        
        for key in ['includes', 'accuracy', 'score', 'correct']:
            if key in sample.scores:
                score_obj = sample.scores[key]
                if hasattr(score_obj, 'value'):
                    score_value = score_obj.value
                    if isinstance(score_value, str):
                        val, binarized = (1.0, 1) if score_value == 'C' else (0.0, 0)
                        return val, binarized
                    elif isinstance(score_value, (int, float)):
                        val = float(score_value)
                        return (val, 1 if val > 0 else 0)
                elif isinstance(score_obj, (float, int)):
                    val = float(score_obj)
                    return (val, 1 if val > 0 else 0)
        return 0.0, 0

    def extract_completed_runs(self, eval_log_paths: List[Path], model_name: str, model_alias: str) -> List[Run]:
        """
        Extract completed runs from multiple eval log files.
        
        Args:
            eval_log_paths: List of paths to .eval log files
            model_name: The model identifier
            model_alias: The model alias
            
        Returns:
            List of Run objects from completed evaluations
        """
        completed_runs = []
        seen_task_ids = set()
        
        # Process files from newest to oldest to get the most recent result for each task
        for log_path in reversed(eval_log_paths):
            try:
                # Pass path as a string to be safe
                log = read_eval_log(str(log_path))
                if log.status != "success" and not log.samples:
                    logger.warning(f"Skipping log with status '{log.status}' and no samples: {log_path.name}")
                    continue

                for sample in log.samples:
                    task_id = sample.id
                    if not task_id or task_id in seen_task_ids:
                        continue
                        
                    seen_task_ids.add(task_id)
                    
                    # Convert sample to Run object
                    score_cont, score = self._extract_score_from_sample(sample)
                    
                    # Create Run object
                    run = Run(
                        task_id=task_id,
                        task_family=self._get_task_family_for_task(task_id),
                        run_id=f"{model_name.replace('/', '_')}_{task_id}_{uuid.uuid4().hex[:8]}",
                        alias=model_alias,
                        model=model_name,
                        score_binarized=score,
                        score_cont=score_cont,
                        human_minutes=self._get_human_minutes_for_task(task_id),
                        human_source="baseline",
                        task_source=self.dataset_name,
                        generation_cost=self._get_generation_cost(sample),
                        started_at=self._get_timestamp(log.eval.created),
                        completed_at=self._get_timestamp(log.stats.completed_at),
                    )
                    completed_runs.append(run)
                    
            except Exception as e:
                logger.error(f"Error extracting runs from {log_path} using inspect_ai: {e}")
                
        logger.info(f"Extracted {len(completed_runs)} completed runs from {len(eval_log_paths)} log files")
        return completed_runs

    def _get_timestamp(self, value: Any) -> float:
        """Safely convert a datetime object or an ISO string to a UNIX timestamp."""
        if not value:
            return 0.0
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str):
            try:
                # Handle ISO format strings, including those with 'Z' for UTC
                return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
            except ValueError:
                logger.warning(f"Could not parse timestamp string: {value}")
                return 0.0
        logger.warning(f"Unhandled timestamp type: {type(value)}")
        return 0.0

    def _get_generation_cost(self, sample: EvalSample) -> float:
        """Safely extract generation cost from a sample, handling different formats."""
        if not sample.model_usage:
            return 0.0
        
        # Handle case where model_usage is an object (newer format)
        if hasattr(sample.model_usage, 'total_tokens'):
            return float(sample.model_usage.total_tokens)
        
        # Handle case where model_usage is a dict (older format)
        if isinstance(sample.model_usage, dict) and 'total_tokens' in sample.model_usage:
            return float(sample.model_usage['total_tokens'])
            
        return 0.0 