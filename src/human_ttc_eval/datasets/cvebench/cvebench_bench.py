"""
CVE-Bench benchmark runner using the cvebench package for AI model evaluation.

Evaluates how well AI models can exploit real-world web application vulnerabilities.
The cvebench package handles Docker orchestration and grading for 8 attack types.

Architecture:
- Imports cvebench task factory from cvebench.cvebench
- Uses cloned cve-bench repo for challenge data files (compose.yml, etc.)
- Docker sandbox for running CVE challenge containers

Requirements:
- Cloned cve-bench repo: make third-party
- Docker for running CVE challenge containers
"""

import json
import os
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from human_ttc_eval.core.bench import Bench, BenchResult
from human_ttc_eval.core.run import Run
from human_ttc_eval.core.registry import register_bench
from human_ttc_eval.core.local_models import (
    validate_local_server,
    LOCAL_MODEL_CONFIGS
)
from human_ttc_eval import config

import inspect_ai
from inspect_ai import eval as inspect_eval

logger = logging.getLogger(__name__)


@register_bench("cvebench")
class CvebenchBench(Bench):
    """
    Benchmark runner for CVE-Bench dataset using the cvebench package.
    
    Unlike other benchmarks that build Tasks manually from cloned repository data,
    CVE-Bench imports a pre-built Task from the cvebench package, which
    handles all Docker container orchestration and grading for 8 attack types.
    """

    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "cvebench"

    def __init__(self, dataset_dir: Path, output_dir: Path):
        """Initialize the benchmark runner."""
        super().__init__(dataset_dir, output_dir)
        
        # Configuration from config.py
        self.token_limit = config.CVEBENCH_TOKEN_LIMIT
        self.sandbox_type = config.CVEBENCH_SANDBOX_TYPE
        self.default_variant = config.CVEBENCH_DEFAULT_VARIANT
        self.message_limit = config.CVEBENCH_MESSAGE_LIMIT
    
    def list_available_tasks(self) -> List[str]:
        """
        List all available task IDs for CVE-Bench.
        
        Returns:
            List of CVE identifiers from the prepared dataset
        """
        tasks_file = config.DATA_DIR / "processed" / self.dataset_name / f"{self.dataset_name}_tasks.jsonl"
        
        if not tasks_file.exists():
            logger.warning(f"Tasks file not found: {tasks_file}")
            return []
        
        task_ids = []
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task = json.loads(line)
                        if 'task_id' in task:
                            task_ids.append(task['task_id'])
        except Exception as e:
            logger.error(f"Error loading task IDs: {e}")
        
        return task_ids

    def run_evaluation(
        self, 
        model_name: str, 
        model_alias: Optional[str] = None, 
        task_ids: Optional[List[str]] = None,
        **kwargs
    ) -> BenchResult:
        """
        Run CVE-Bench evaluation using the cvebench package.
        
        Args:
            model_name: Model identifier (e.g., "openai/gpt-4o")
            model_alias: Display name for the model (defaults to model_name)
            task_ids: Optional list of specific CVE IDs to run (None = all CVEs)
            **kwargs: Additional evaluation parameters including:
                - resume: Whether to resume from previous evaluation
                - variant: Override default variant ("one_day" or "zero_day")
            
        Returns:
            BenchResult with evaluation results
        """
        start_time = datetime.now(timezone.utc)
        model_alias = model_alias or model_name
        is_human_eval = model_name.lower() == "human"
        is_local = model_name in LOCAL_MODEL_CONFIGS
        
        # Get variant from kwargs or use default
        variant = kwargs.get('variant', self.default_variant)
        
        # Resume logic
        resume = kwargs.get('resume', False)
        completed_task_ids = set()
        completed_runs: List[Run] = []
        eval_logs: List[Path] = []
        if resume and not is_human_eval:
            eval_logs = self.find_all_eval_logs(model_name)
            if eval_logs:
                logger.info(f"Found {len(eval_logs)} existing evaluation log(s) for resumption.")
                for log_path in eval_logs:
                    completed_task_ids.update(self.extract_completed_task_ids(log_path))
                if completed_task_ids:
                    logger.info(f"Found {len(completed_task_ids)} completed tasks across all logs.")
                    completed_runs = self.extract_completed_runs(eval_logs, model_name, model_alias)

        if not is_human_eval:
            # Check for zero-imputation first (models that can't run agentic tasks)
            if model_name in config.CVEBENCH_IMPUTE_ZERO:
                effective_task_ids = task_ids or self.list_available_tasks()
                return self._create_zero_imputed_result(
                    model_name=model_name,
                    model_alias=model_alias,
                    task_ids=effective_task_ids,
                    start_time=start_time,
                    reason=f"Model '{model_name}' cannot run agentic CVE-Bench tasks"
                )

            # Validate model format
            if "/" not in model_name:
                error_msg = f"Model name must be in provider/model format, got: {model_name}"
                logger.error(error_msg)
                return self._create_error_result(model_name, model_alias, start_time, error_msg)
            
            # Check if this is a local model and if the server is running
            if is_local and not validate_local_server(model_name):
                error_msg = f"Local server not running for {model_name}. Run 'make start-local-model-servers' first."
                logger.error(error_msg)
                return self._create_error_result(model_name, model_alias, start_time, error_msg)
        
        # Load tasks
        all_tasks_for_stats = self._load_tasks(task_ids)
        tasks_to_run = all_tasks_for_stats
        
        # Filter out completed tasks if resuming
        if completed_task_ids:
            tasks_to_run = [t for t in all_tasks_for_stats if t.get('task_id') not in completed_task_ids]
            logger.info(f"Filtered tasks: {len(all_tasks_for_stats)} -> {len(tasks_to_run)} to run.")

            if not tasks_to_run:
                logger.info("All tasks already completed. Returning merged results from previous runs.")
                summary_stats = self._calculate_summary_stats(completed_runs)
                summary_stats.update(self._calculate_cvebench_stats(completed_runs, all_tasks_for_stats))
                
                return BenchResult(
                    dataset_name=self.dataset_name, model_name=model_name, model_alias=model_alias,
                    runs=completed_runs, summary_stats=summary_stats,
                    metadata={
                        "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                        "num_tasks": len(completed_runs), "resumed": True,
                        "num_eval_logs_merged": len(eval_logs),
                        "variant": variant,
                    },
                    timestamp=start_time.isoformat(), success=True
                )
        
        if not tasks_to_run:
            error_msg = "No tasks loaded for evaluation"
            logger.error(error_msg)
            return self._create_error_result(model_name, model_alias, start_time, error_msg)
        
        logger.info(f"Starting CVE-Bench evaluation with {len(tasks_to_run)} CVEs on model: {model_name}")
        logger.info(f"Variant: {variant}, Sandbox: {self.sandbox_type}")
        
        try:
            # Import cvebench directly (not via inspect_evals wrapper)
            from cvebench.cvebench import cvebench
            
            # Get challenges directory from cloned cve-bench repo
            challenges_dir = config.CVEBENCH_CHALLENGES_DIR
            if not challenges_dir.exists():
                error_msg = f"CVE-Bench challenges not found at {challenges_dir}. Run 'make third-party' to clone the cve-bench repo."
                logger.error(error_msg)
                return self._create_error_result(model_name, model_alias, start_time, error_msg)
            
            # Set up CVE-Bench environment variables for Docker compose
            cvebench_repo = config.CVEBENCH_REPO_PATH.resolve()
            
            # Get cvebench version for Docker image tags
            import cvebench as cvebench_module
            cvebench_version = getattr(cvebench_module, '__version__', '2.0.0')
            os.environ["CVEBENCH_TAG"] = cvebench_version
            os.environ["CVEBENCH_VERSION"] = "critical"
            
            # Force x86_64 platform for Docker (CVE-Bench images are amd64 only)
            os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"
            os.environ["CVEBENCH_VERSION_DIR"] = str(cvebench_repo / "src" / "critical")
            os.environ["CVEBENCH_CHALLENGE_DIR"] = str(challenges_dir.resolve())
            os.environ["CVEBENCH_METADATA_DIR"] = str(cvebench_repo / "src" / "critical" / "metadata")
            os.environ["CVEBENCH_NVD_DIR"] = str(cvebench_repo / "src" / "critical" / "nvd")
            os.environ["CVEBENCH_VERSION_DOCKER_DIR"] = str(cvebench_repo / "src" / "critical" / "docker")
            os.environ["CVEBENCH_DOCKER_DIR"] = str(cvebench_repo / "src" / "common" / "docker")
            os.environ["CVEBENCH_EVALUATIONS_DIR"] = str(cvebench_repo / "src" / "common" / "evaluations")
            os.environ["CVEBENCH_SANDBOXES_DIR"] = str(cvebench_repo / "src" / "common" / "sandboxes")
            
            logger.info(f"Using challenges from: {challenges_dir}")
            logger.info(f"CVEBENCH_DOCKER_DIR: {os.environ['CVEBENCH_DOCKER_DIR']}")
            
            # Build list of CVE challenges to run
            challenges = [t['task_id'] for t in tasks_to_run]
            
            # Create the cvebench task directly (bypassing inspect_evals wrapper)
            # Note: cvebench v2.0.0 doesn't accept sandbox_type parameter
            cve_task = cvebench(
                challenges=challenges,
                variants=variant,
                challenges_dir=str(challenges_dir.resolve()),
            )
            
            # Prepare eval parameters
            eval_params = {
                "log_dir": str(self.output_dir / "inspect_logs"),
                "token_limit": self.token_limit,
                "message_limit": self.message_limit,
            }
            
            if is_human_eval:
                logger.info("Human evaluation mode not directly supported for CVE-Bench.")
                logger.info("Use the CVE-Bench repo directly for manual exploitation.")
                error_msg = "Human evaluation not supported for CVE-Bench via this interface"
                return self._create_error_result(model_name, model_alias, start_time, error_msg)
            else:
                eval_params["model"] = model_name
                # Add base URL for local models
                if is_local:
                    local_config = LOCAL_MODEL_CONFIGS[model_name]
                    eval_params["model_base_url"] = local_config["base_url"]
            
            # CVE-Bench requires lower concurrency due to container orchestration
            max_connections = 1
            
            # Run evaluation using inspect_eval
            eval_result = inspect_eval(
                cve_task, 
                retry_on_error=3,  # Retry failed samples up to 3 times
                fail_on_error=0.1,  # Tolerate up to 10% sample failures
                max_connections=max_connections,
                message_limit=self.message_limit,
                time_limit=config.INSPECT_TIME_LIMIT,
                **eval_params
            )
            
            # Parse results into Run objects
            new_runs = self._parse_inspect_results(eval_result, tasks_to_run, model_name, model_alias)
            
            # Merge with previous runs if resuming
            runs = new_runs + completed_runs
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(runs)
            
            # Add CVE-Bench specific stats
            summary_stats.update(self._calculate_cvebench_stats(runs, all_tasks_for_stats))
            
            # Create successful result
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            result = BenchResult(
                dataset_name=self.dataset_name,
                model_name=model_name,
                model_alias=model_alias,
                runs=runs,
                summary_stats=summary_stats,
                metadata={
                    "duration_seconds": duration,
                    "num_tasks": len(runs),
                    "inspect_ai_version": inspect_ai.__version__,
                    "token_limit": self.token_limit,
                    "is_local_model": is_local,
                    "sandbox_type": self.sandbox_type,
                    "variant": variant,
                    "resumed": resume,
                    "num_completed_from_previous": len(completed_runs),
                    "num_new_completions": len(new_runs),
                },
                timestamp=start_time.isoformat(),
                success=True,
                error_message=None
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_result(model_name, model_alias, start_time, error_msg)
    
    def _load_tasks(self, task_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load tasks from the prepared dataset."""
        tasks_file = config.DATA_DIR / "processed" / self.dataset_name / f"{self.dataset_name}_tasks.jsonl"
        
        if not tasks_file.exists():
            logger.error(f"Tasks file not found: {tasks_file}")
            return []
        
        all_tasks = []
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task = json.loads(line)
                        all_tasks.append(task)
        except Exception as e:
            logger.error(f"Error loading tasks: {e}")
            return []
        
        # Filter by task_ids if specified
        if task_ids:
            task_id_set = set(task_ids)
            final_tasks = [t for t in all_tasks if t.get('task_id') in task_id_set]
            logger.info(f"Filtered to {len(final_tasks)} tasks matching specified CVE IDs")
            return final_tasks
        
        return all_tasks
    
    def _parse_inspect_results(
        self, 
        eval_result, 
        tasks: List[Dict[str, Any]],
        model_name: str,
        model_alias: str
    ) -> List[Run]:
        """Parse inspect_ai results into Run objects."""
        runs: List[Run] = []
        
        # Extract samples from eval_result
        samples = []
        if hasattr(eval_result, '__iter__') and hasattr(eval_result, '__len__'):
            for eval_log in eval_result:
                if hasattr(eval_log, 'samples') and eval_log.samples:
                    samples.extend(eval_log.samples)
        elif hasattr(eval_result, 'samples'):
            samples = eval_result.samples
        else:
            logger.warning("Could not extract samples from eval_result")
            return runs
        
        # Create task lookup
        tasks_by_id = {task['task_id']: task for task in tasks}
        
        # Convert samples to Run objects
        for sample in samples:
            task_id = getattr(sample, 'id', None)
            if not task_id or task_id not in tasks_by_id:
                continue
                
            task_data = tasks_by_id[task_id]
            
            # Extract score (CVE exploitation is binary - success or failure)
            score_value, score_binarized = self._extract_score_from_sample(sample)
            
            # Get metadata from task
            metadata = task_data.get('dataset_task_metadata', {})
            category = metadata.get('category', 'unknown')
            
            # Create Run object
            run = Run(
                task_id=task_id,
                task_family=f"cvebench_{category}",
                run_id=f"{model_name.replace('/', '_')}_{task_id}_{uuid.uuid4().hex[:8]}",
                alias=model_alias,
                model=model_name,
                score_binarized=score_binarized,
                score_cont=score_value,
                human_minutes=self._get_human_minutes_for_task(task_id),
                human_source="baseline",
                task_source=self.dataset_name,
                started_at=0.0,
                completed_at=0.0,
                generation_cost=0.0,
                fatal_error_from=None
            )
            runs.append(run)
        
        return runs
    
    def _extract_score_from_sample(self, sample) -> tuple:
        """Extract score from inspect_ai sample.
        
        Returns:
            tuple: (continuous_score, binary_score) where binary_score is 0 or 1
        """
        if not hasattr(sample, 'scores') or not sample.scores:
            return 0.0, 0
        
        # Try different score keys that inspect_ai might use
        # CVE-Bench uses custom graders, so we check various keys
        score_keys = ['grader', 'accuracy', 'score', 'correct', 'exploit_success']
        
        for key in score_keys:
            if key in sample.scores:
                score_obj = sample.scores[key]
                if hasattr(score_obj, 'value'):
                    # Handle inspect_ai score objects
                    if isinstance(score_obj.value, str):
                        # 'C' = Correct/Success, 'I' = Incorrect/Failure
                        return (1.0, 1) if score_obj.value == 'C' else (0.0, 0)
                    elif isinstance(score_obj.value, (int, float)):
                        score_val = float(score_obj.value)
                        return (score_val, 1 if score_val > 0 else 0)
                elif isinstance(score_obj, (float, int)):
                    score_val = float(score_obj)
                    return (score_val, 1 if score_val > 0 else 0)
        
        return 0.0, 0
    
    def _calculate_cvebench_stats(self, runs: List[Run], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate CVE-Bench specific statistics."""
        # Group by attack category
        category_stats: Dict[str, Dict[str, float]] = {}
        
        for run in runs:
            # Find task to get category
            task = next((t for t in tasks if t['task_id'] == run.task_id), None)
            if task:
                metadata = task.get('dataset_task_metadata', {})
                category = metadata.get('category', 'unknown')
                
                if category not in category_stats:
                    category_stats[category] = {
                        'total': 0,
                        'exploited': 0
                    }
                
                category_stats[category]['total'] += 1
                if run.score_binarized == 1:
                    category_stats[category]['exploited'] += 1
        
        # Calculate exploit rates per category
        for category, stats in category_stats.items():
            stats['exploit_rate'] = stats['exploited'] / stats['total'] if stats['total'] > 0 else 0.0
        
        # Calculate application breakdown
        application_stats: Dict[str, int] = {}
        for task in tasks:
            metadata = task.get('dataset_task_metadata', {})
            application = metadata.get('application', 'unknown')
            if application not in application_stats:
                application_stats[application] = 0
            application_stats[application] += 1
        
        return {
            'category_breakdown': category_stats,
            'application_breakdown': application_stats,
            'total_categories': len(category_stats),
            'total_applications': len(application_stats),
            'total_cves': len(tasks),
        }
    
    def _create_error_result(self, model_name: str, model_alias: str, start_time: datetime, error_msg: str) -> BenchResult:
        """Create a BenchResult for error cases."""
        return BenchResult(
            dataset_name=self.dataset_name,
            model_name=model_name,
            model_alias=model_alias,
            runs=[],
            summary_stats={"error": error_msg},
            metadata={
                "error": error_msg,
                "timestamp": start_time.isoformat()
            },
            timestamp=start_time.isoformat(),
            success=False,
            error_message=error_msg
        )

