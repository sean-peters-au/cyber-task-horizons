"""
CyBench-specific benchmark runner, aligned with the new Bench base class.
"""
import subprocess
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timezone
import uuid

from human_ttc_eval.core import Bench, BenchResult, Run, register_bench
from human_ttc_eval.config import (
    CYBENCH_MAX_ITERATIONS,
    CYBENCH_MAX_INPUT_TOKENS,
    CYBENCH_MAX_OUTPUT_TOKENS,
    CYBENCH_EVALUATION_TIMEOUT_SECONDS,
    CYBENCH_REPO_PATH,
    CYBENCH_IMPUTE_ZERO
)

logger = logging.getLogger(__name__)

@register_bench("cybench")
class CybenchBench(Bench):
    """
    Benchmark runner for CyBench dataset using their native evaluation system.
    This class overrides run_evaluation to call CyBench's run_benchmark.py script
    and parses its log outputs into a list of Run objects and a BenchResult.
    """

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

    def __init__(self, dataset_dir: Path, output_dir: Path, cybench_repo_path_override: Optional[Path] = None):
        super().__init__(dataset_dir, output_dir)
        self.cybench_repo_path = Path(cybench_repo_path_override or CYBENCH_REPO_PATH)
        if not self.cybench_repo_path.is_dir():
            raise FileNotFoundError(f"CyBench repository not found at: {self.cybench_repo_path}")
        
        run_benchmark_script = self.cybench_repo_path / "run_benchmark.py"
        if not run_benchmark_script.is_file():
            raise FileNotFoundError(f"run_benchmark.py not found in CyBench repo: {self.cybench_repo_path}")
        logger.info(f"CyBenchBench initialized, using CyBench repo: {self.cybench_repo_path}")

    @property
    def dataset_name(self) -> str:
        return "cybench"

    def list_available_tasks(self) -> List[str]:
        task_list_file = self.cybench_repo_path / "task_list.txt"
        subtask_list_file = self.cybench_repo_path / "subtask_list.txt"
        tasks = []
        if task_list_file.is_file():
            with open(task_list_file, 'r', encoding='utf-8') as f:
                tasks.extend([line.strip() for line in f if line.strip()])
        if subtask_list_file.is_file():
            with open(subtask_list_file, 'r', encoding='utf-8') as f:
                subtasks = [line.strip() for line in f if line.strip()]
                for subtask in subtasks:
                    if subtask not in tasks:
                        tasks.append(subtask)
        if not tasks:
            logger.warning("No tasks found from task_list.txt or subtask_list.txt in CyBench repo.")
        return tasks

    def validate_model_name(self, model_name: str) -> bool:
        """Validate that the model is supported by CyBench."""
        return model_name in self.AVAILABLE_MODELS

    def run_evaluation(
        self, 
        model_name: str, 
        model_alias: Optional[str] = None, 
        task_ids: Optional[List[str]] = None,
        max_iterations: Optional[int] = None, # Allow overriding config
        **kwargs # Catches other potential params from base class or CLI
    ) -> BenchResult:
        eval_start_time = datetime.now(timezone.utc)
        model_alias = model_alias or model_name

        # Check for zero-imputation first
        if model_name in CYBENCH_IMPUTE_ZERO:
            effective_task_ids = task_ids or self.list_available_tasks()
            return self._create_zero_imputed_result(
                model_name=model_name,
                model_alias=model_alias,
                task_ids=effective_task_ids,
                start_time=eval_start_time,
                reason=f"Model '{model_name}' cannot run tool-requiring CyBench tasks"
            )

        if not self.validate_model_name(model_name):
            error_msg = f"Model '{model_name}' not supported by CyBench. Available: {self.AVAILABLE_MODELS}"
            logger.error(error_msg)
            return BenchResult(
                dataset_name=self.dataset_name, model_name=model_name, model_alias=model_alias,
                runs=[], summary_stats={},
                metadata={"error": error_msg}, timestamp=eval_start_time.isoformat(),
                success=False, error_message=error_msg
            )

        cmd = [
            "python3", str(self.cybench_repo_path / "run_benchmark.py"),
            "--model", model_name,
            "--max_iterations", str(max_iterations or CYBENCH_MAX_ITERATIONS),
            "--max_input_tokens", str(CYBENCH_MAX_INPUT_TOKENS),
            "--max_output_tokens", str(CYBENCH_MAX_OUTPUT_TOKENS),
            "--unguided_mode" # Assuming unguided mode is standard for our benchmarking
        ]

        temp_task_list_path: Optional[Path] = None
        effective_task_ids = task_ids or self.list_available_tasks()
        if not effective_task_ids:
            logger.warning("No tasks specified or found for CyBench evaluation.")
            return BenchResult(
                dataset_name=self.dataset_name, model_name=model_name, model_alias=model_alias,
                runs=[], summary_stats={"message": "No tasks to run"},
                metadata={}, timestamp=eval_start_time.isoformat(),
                success=True, error_message=None # Success=True as it didn't fail, just nothing to do
            )

        if task_ids is not None: # Use this condition to decide if --task_list is needed
            temp_task_list_path = self.cybench_repo_path / f"temp_task_list_{model_name.replace('/', '_')}_{int(time.time())}.txt"
            with open(temp_task_list_path, 'w', encoding='utf-8') as f:
                for t_id in effective_task_ids:
                    f.write(f"{t_id}\n")
            cmd.extend(["--task_list", str(temp_task_list_path)])
        
        runs: List[Run] = []
        raw_output_path_for_result = self.cybench_repo_path / "logs" # Main log dir
        metadata_for_result: Dict[str, Any] = {"cybench_command": " ".join(cmd)}

        try:
            process = subprocess.run(
                cmd, cwd=self.cybench_repo_path, capture_output=True, text=True,
                timeout=CYBENCH_EVALUATION_TIMEOUT_SECONDS * len(effective_task_ids) # Adjust timeout based on #tasks
            )
            eval_end_time = datetime.now(timezone.utc)
            duration_seconds = (eval_end_time - eval_start_time).total_seconds()
            metadata_for_result["duration_seconds"] = duration_seconds
            metadata_for_result["cybench_stdout_tail"] = process.stdout[-2000:]
            metadata_for_result["cybench_stderr_tail"] = process.stderr[-2000:] if process.stderr else None
            metadata_for_result["cybench_returncode"] = process.returncode

            if process.returncode != 0:
                logger.error(f"CyBench script failed with code {process.returncode}. Stderr: {process.stderr}")
                # Attempt to parse any partial logs anyway
            
            parsed_runs, _ = self._parse_and_create_runs(raw_output_path_for_result, model_name, model_alias, effective_task_ids)
            runs.extend(parsed_runs)
            self._latest_batch_runs = runs

            # Fill in any missing tasks with failure runs if they were expected but not in logs
            logged_task_ids = {r.task_id for r in runs}
            for t_id in effective_task_ids:
                if t_id not in logged_task_ids:
                    logger.warning(f"Task '{t_id}' was in the input list but no log found. Marking as failed.")
                    runs.append(self._create_failed_run(t_id, model_name, model_alias, "No log file found after benchmark execution"))
            
            success_flag = True # Overall success unless critical script error
            error_message_for_result = None
            if process.returncode != 0: # If script had non-zero exit, consider it a failure for BenchResult success status
                success_flag = False
                error_message_for_result = f"CyBench script error (code {process.returncode}): {process.stderr[:500] if process.stderr else 'Unknown error'}"
                if not runs: # Also log if no runs were parsed on top of script error
                    logger.error(f"CyBench script failed (code {process.returncode}) AND no runs were parsed.")

        except subprocess.TimeoutExpired:
            duration_seconds = (datetime.now(timezone.utc) - eval_start_time).total_seconds()
            error_msg = f"CyBench evaluation timed out after {duration_seconds:.2f}s (limit per task might have been exceeded globally)."
            logger.error(error_msg)
            metadata_for_result["error"] = error_msg
            metadata_for_result["duration_seconds"] = duration_seconds

            # Create failed runs for all expected tasks on timeout
            for t_id in effective_task_ids:
                runs.append(self._create_failed_run(t_id, model_name, model_alias, "Global evaluation timeout"))
            success_flag = False
            error_message_for_result = error_msg
        except Exception as e:
            duration_seconds = (datetime.now(timezone.utc) - eval_start_time).total_seconds()
            error_msg = f"Unexpected error during CyBench evaluation: {e}"
            logger.error(error_msg, exc_info=True)
            metadata_for_result["error"] = error_msg
            metadata_for_result["duration_seconds"] = duration_seconds
            for t_id in effective_task_ids:
                 runs.append(self._create_failed_run(t_id, model_name, model_alias, f"Unexpected error: {str(e)[:100]}"))
            success_flag = False
            error_message_for_result = error_msg
        finally:
            if temp_task_list_path and temp_task_list_path.exists():
                temp_task_list_path.unlink()
        
        summary_stats = self._calculate_summary_stats(runs) # Use base class method
        
        bench_result_obj = BenchResult(
            dataset_name=self.dataset_name, model_name=model_name, model_alias=model_alias,
            runs=runs, summary_stats=summary_stats, metadata=metadata_for_result,
            timestamp=eval_start_time.isoformat(), success=success_flag, error_message=error_message_for_result
        )
        self.save_result(bench_result_obj) # Use base class method
        return bench_result_obj

    def _parse_and_create_runs(
        self, 
        cybench_logs_dir: Path, 
        model_name: str, 
        model_alias: str,
        expected_task_ids: List[str] # To help filter relevant logs if needed
    ) -> tuple[List[Run], Dict[str, Any]]:
        """
        Parses CyBench log files and creates Run objects.
        Adapts logic from the old _parse_cybench_logs.
        """
        created_runs: List[Run] = []
        if not cybench_logs_dir.is_dir():
            logger.warning(f"CyBench logs directory not found for parsing: {cybench_logs_dir}")
            return [], {}

        model_basename = model_name.split("/")[-1]
        log_files = list(cybench_logs_dir.glob(f"**/{model_basename}_*.json"))
        logger.info(f"Found {len(log_files)} potential log files in {cybench_logs_dir} for model basename '{model_basename}'.")

        parsed_task_ids = set()

        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                task_path_from_log = log_data.get("task", {}).get("path")
                if not task_path_from_log:
                    logger.warning(f"Log file {log_file} missing task path. Skipping.")
                    continue
                
                if task_path_from_log not in expected_task_ids:
                    continue
                
                if task_path_from_log in parsed_task_ids:
                    logger.warning(f"Duplicate log found for task '{task_path_from_log}' (file: {log_file}). Taking first encountered.")
                    continue
                parsed_task_ids.add(task_path_from_log)

                human_minutes = self._get_human_minutes_for_task(task_path_from_log)
                task_family = self._get_task_family(task_path_from_log) # Usually self.dataset_name

                # Scores
                num_correct = log_data.get("num_correct_subtasks", 0)
                num_total_subtasks = max(log_data.get("num_subtasks", 1), 1) # Avoid division by zero
                score_cont = num_correct / num_total_subtasks
                score_binarized = 1 if num_correct > 0 else 0

                # Timings from logs (if available, else fallback)
                time_taken_ms = log_data.get("time_taken_in_ms")
                run_started_at = 0.0 # Relative to this run attempt
                run_completed_at = float(time_taken_ms / 1000.0) if time_taken_ms is not None else None

                generation_cost = 0.0 

                run_id = f"{model_name.replace('/', '_')}_{task_path_from_log.replace('/', '_')}_{uuid.uuid4().hex[:8]}"

                run = Run(
                    task_id=task_path_from_log,
                    task_family=task_family,
                    run_id=run_id,
                    alias=model_alias,
                    model=model_name,
                    score_binarized=score_binarized,
                    score_cont=score_cont,
                    human_minutes=human_minutes,
                    started_at=run_started_at,
                    completed_at=run_completed_at, 
                    generation_cost=generation_cost,
                    human_source="baseline", # From loaded human baseline
                    task_source=self.dataset_name,
                    fatal_error_from=log_data.get("error_message") # If CyBench itself logged an error for the task
                )
                created_runs.append(run)
            except Exception as e:
                logger.warning(f"Error parsing CyBench log file {log_file}: {e}", exc_info=True)
        
        return created_runs, {}

    def _get_human_minutes_for_task(self, task_id: str) -> float:
        for run in self.human_baseline:
            if run.task_id == task_id:
                return run.human_minutes
        logger.warning(f"No human baseline minutes found for task: {task_id}. Defaulting to 0.")
        return 0.0
    
    def _get_task_family(self, task_id: str) -> str:
        return self.dataset_name 