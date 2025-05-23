import subprocess
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import re

from ...core.base_bench import BaseBench, BenchmarkResult

logger = logging.getLogger(__name__)

# CyBench Evaluation Constants
DEFAULT_MAX_ITERATIONS = 15
DEFAULT_MAX_INPUT_TOKENS = 6000
DEFAULT_MAX_OUTPUT_TOKENS = 2000
EVALUATION_TIMEOUT_SECONDS = 7200  # 2 hours

class CyBenchBench(BaseBench):
    """
    Benchmark runner for CyBench dataset using their native evaluation system.
    
    This class wraps CyBench's run_benchmark.py script and parses their 
    log outputs into our standardized BenchmarkResult format.
    """
    
    # CyBench's supported models (from their run_benchmark.py)
    AVAILABLE_MODELS = [
        "openai/gpt-4-0613",
        "openai/gpt-4-32k-0613", 
        "openai/gpt-4-turbo-2024-04-09",
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-3.5-turbo-0125",
        "together/llama-3-8b",
        "together/llama-3-70b",
        "together/llama-3-70b-chat",
        "together/llama-3.1-70b-instruct-turbo",
        "together/llama-3.1-405b-instruct-turbo",
        "together/mixtral-8x22b",
        "together/mixtral-8x22b-instruct-v0.1",
        "mistralai/mistral-large-2407",
        "together/qwen2-72b-instruct",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-5-sonnet-20240620",
        "google/gemini-1.0-pro-001",
        "google/gemini-1.5-pro-001", 
        "google/gemini-1.5-pro-preview-0409",
        "01-ai/yi-large",
        # New models added
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-3-7-sonnet-20250219",
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-opus-4-20250514",
        "openai/o4-mini-2025-04-16",
        "openai/o3-mini-2025-01-31",
        "openai/o1-2024-12-17",
        "google/gemini-2.5-flash-preview-05-20",
        "google/gemini-2.5-pro-preview-05-06",
        "google/gemini-2.0-flash"
    ]
    
    def __init__(self, output_dir: Path, cybench_repo_path: Path):
        """
        Initialize CyBench benchmark runner.
        
        Args:
            output_dir: Directory to store our standardized evaluation results
            cybench_repo_path: Path to the local CyBench repository clone
        """
        super().__init__("cybench", output_dir)
        self.cybench_repo_path = Path(cybench_repo_path)
        
        # Validate CyBench repo exists
        if not self.cybench_repo_path.exists():
            raise FileNotFoundError(f"CyBench repository not found at: {cybench_repo_path}")
        
        run_benchmark_script = self.cybench_repo_path / "run_benchmark.py"
        if not run_benchmark_script.exists():
            raise FileNotFoundError(f"run_benchmark.py not found in: {cybench_repo_path}")
            
        logger.info(f"CyBenchBench initialized with repo: {self.cybench_repo_path}")
    
    def list_available_tasks(self) -> List[str]:
        """List all available CyBench tasks."""
        task_list_file = self.cybench_repo_path / "task_list.txt"
        subtask_list_file = self.cybench_repo_path / "subtask_list.txt"
        
        tasks = []
        
        # Read main task list
        if task_list_file.exists():
            with open(task_list_file, 'r', encoding='utf-8') as f:
                tasks.extend([line.strip() for line in f if line.strip()])
        
        # Read subtask list (if different from main tasks)
        if subtask_list_file.exists():
            with open(subtask_list_file, 'r', encoding='utf-8') as f:
                subtasks = [line.strip() for line in f if line.strip()]
                # Add subtasks if they're not already in the main list
                for subtask in subtasks:
                    if subtask not in tasks:
                        tasks.append(subtask)
        
        logger.info(f"Found {len(tasks)} available CyBench tasks")
        return tasks
    
    def validate_model_name(self, model_name: str) -> bool:
        """Validate that the model is supported by CyBench."""
        return model_name in self.AVAILABLE_MODELS
    
    def run_evaluation(
        self,
        model_name: str,
        tasks: Optional[List[str]] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        unguided_mode: bool = False,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run CyBench evaluation using their native system.
        
        Args:
            model_name: Model identifier (must be in AVAILABLE_MODELS)
            tasks: Optional list of specific tasks (None = all tasks)
            max_iterations: Maximum iterations per task
            unguided_mode: Whether to run in unguided mode
            **kwargs: Additional parameters
            
        Returns:
            BenchmarkResult with parsed evaluation results
        """
        start_time = datetime.now()
        
        # Validate model
        if not self.validate_model_name(model_name):
            error_msg = f"Model '{model_name}' not supported. Available models: {self.AVAILABLE_MODELS}"
            logger.error(error_msg)
            return BenchmarkResult(
                dataset_name=self.dataset_name,
                model_name=model_name,
                task_results=[],
                summary_stats={},
                metadata={"error": error_msg},
                raw_output_path=None,
                timestamp=start_time.isoformat(),
                success=False,
                error_message=error_msg
            )
        
        # Build command for CyBench's run_benchmark.py
        cmd = [
            "python3", "run_benchmark.py",
            "--model", model_name,
            "--max_iterations", str(max_iterations),
            "--max_input_tokens", str(DEFAULT_MAX_INPUT_TOKENS),
            "--max_output_tokens", str(DEFAULT_MAX_OUTPUT_TOKENS)
        ]
        
        if unguided_mode:
            cmd.append("--unguided_mode")
        
        # If specific tasks requested, create a custom task list file
        task_list_file = None
        if tasks:
            task_list_file = self.cybench_repo_path / f"custom_task_list_{int(time.time())}.txt"
            with open(task_list_file, 'w', encoding='utf-8') as f:
                for task in tasks:
                    f.write(f"{task}\n")
            cmd.extend(["--task_list", str(task_list_file)])
        
        logger.info(f"Running CyBench evaluation: {' '.join(cmd)}")
        
        try:
            # Run CyBench evaluation
            result = subprocess.run(
                cmd,
                cwd=self.cybench_repo_path,
                capture_output=True,
                text=True,
                timeout=EVALUATION_TIMEOUT_SECONDS
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results from CyBench logs
            cybench_logs_dir = self.cybench_repo_path / "logs"
            task_results, summary_stats = self._parse_cybench_logs(cybench_logs_dir, model_name)
            
            # Create our standardized result
            benchmark_result = BenchmarkResult(
                dataset_name=self.dataset_name,
                model_name=model_name,
                task_results=task_results,
                summary_stats=summary_stats,
                metadata={
                    "cybench_command": " ".join(cmd),
                    "duration_seconds": duration,
                    "max_iterations": max_iterations,
                    "unguided_mode": unguided_mode,
                    "cybench_stdout": result.stdout[-2000:],  # Last 2000 chars
                    "cybench_stderr": result.stderr[-2000:] if result.stderr else None,
                    "cybench_returncode": result.returncode
                },
                raw_output_path=cybench_logs_dir,
                timestamp=start_time.isoformat(),
                success=(result.returncode == 0 and len(task_results) > 0),
                error_message=result.stderr if result.returncode != 0 else None
            )
            
            # Save our standardized result
            self.save_result(benchmark_result)
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            error_msg = f"CyBench evaluation timed out after {EVALUATION_TIMEOUT_SECONDS} seconds"
            logger.error(error_msg)
            return BenchmarkResult(
                dataset_name=self.dataset_name,
                model_name=model_name,
                task_results=[],
                summary_stats={},
                metadata={"error": error_msg, "duration_seconds": EVALUATION_TIMEOUT_SECONDS},
                raw_output_path=None,
                timestamp=start_time.isoformat(),
                success=False,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error running CyBench evaluation: {e}"
            logger.error(error_msg, exc_info=True)
            return BenchmarkResult(
                dataset_name=self.dataset_name,
                model_name=model_name,
                task_results=[],
                summary_stats={},
                metadata={"error": error_msg},
                raw_output_path=None,
                timestamp=start_time.isoformat(),
                success=False,
                error_message=error_msg
            )
        finally:
            # Clean up custom task list file
            if task_list_file and task_list_file.exists():
                task_list_file.unlink()
    
    def _parse_cybench_logs(self, logs_dir: Path, model_name: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parse CyBench log files into our standardized format.
        
        Args:
            logs_dir: Path to CyBench's logs directory
            model_name: Model name to filter logs
            
        Returns:
            Tuple of (task_results, summary_stats)
        """
        task_results = []
        
        if not logs_dir.exists():
            logger.warning(f"CyBench logs directory not found: {logs_dir}")
            return task_results, {}
        
        # Find all log files for this model
        model_slug = model_name.replace("/", "_")
        log_files = list(logs_dir.glob(f"**/*{model_slug}*.json"))
        
        logger.info(f"Found {len(log_files)} CyBench log files for model {model_name}")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                # Extract key information from CyBench log format
                task_result = {
                    "task_name": log_data.get("task", {}).get("name", "unknown"),
                    "task_path": log_data.get("task", {}).get("path", ""),
                    "success": log_data.get("num_correct_subtasks", 0) > 0,
                    "score": log_data.get("num_correct_subtasks", 0) / max(log_data.get("num_subtasks", 1), 1),
                    "subtasks_completed": log_data.get("num_correct_subtasks", 0),
                    "total_subtasks": log_data.get("num_subtasks", 0),
                    "time_taken_ms": log_data.get("time_taken_in_ms", 0),
                    "iterations_used": log_data.get("usage", {}).get("iterations_used", 0),
                    "input_tokens": log_data.get("usage", {}).get("input_tokens_used", 0),
                    "output_tokens": log_data.get("usage", {}).get("output_tokens_used", 0),
                    "raw_log_path": str(log_file)
                }
                
                task_results.append(task_result)
                
            except Exception as e:
                logger.warning(f"Failed to parse CyBench log file {log_file}: {e}")
        
        # Calculate summary statistics
        if task_results:
            total_tasks = len(task_results)
            successful_tasks = sum(1 for r in task_results if r["success"])
            
            summary_stats = {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": successful_tasks / total_tasks,
                "average_score": sum(r["score"] for r in task_results) / total_tasks,
                "total_time_ms": sum(r["time_taken_ms"] for r in task_results),
                "total_iterations": sum(r["iterations_used"] for r in task_results),
                "total_input_tokens": sum(r["input_tokens"] for r in task_results),
                "total_output_tokens": sum(r["output_tokens"] for r in task_results),
                "average_iterations_per_task": sum(r["iterations_used"] for r in task_results) / total_tasks
            }
        else:
            summary_stats = {
                "total_tasks": 0,
                "successful_tasks": 0, 
                "success_rate": 0.0,
                "average_score": 0.0
            }
        
        return task_results, summary_stats 