"""
NL2Bash benchmark runner using inspect_ai for AI model evaluation.

Evaluates how well AI models can generate bash commands from natural
language descriptions, using functional equivalence scoring rather
than exact match.
"""

import json
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from human_ttc_eval.core.bench import Bench, BenchResult
from human_ttc_eval.core.run import Run
from human_ttc_eval.core.registry import register_bench
from human_ttc_eval.core.local_models import (
    validate_local_server,
    LOCAL_MODEL_CONFIGS
)

import inspect_ai
from inspect_ai import eval as inspect_eval
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message

logger = logging.getLogger(__name__)


@register_bench("nl2bash")
class NL2BashBench(Bench):
    """
    Benchmark runner for NL2Bash dataset using inspect_ai.
    
    Evaluates AI models on their ability to generate correct bash commands
    from natural language descriptions using LLM-based functional equivalence scoring.
    """
    
    @property
    def dataset_name(self) -> str:
        """Returns the dataset identifier."""
        return "nl2bash"
    
    def list_available_tasks(self) -> List[str]:
        """
        List all available task IDs for NL2Bash.
        
        Returns:
            List of task identifiers from the prepared dataset
        """
        tasks_file = self.dataset_dir.parent.parent / "processed" / self.dataset_name / f"{self.dataset_name}_tasks.jsonl"
        
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
        Run NL2Bash evaluation using inspect_ai.
        
        Args:
            model_name: Model identifier (e.g., "openai/gpt-4" or "openai/gpt2" for local)
            model_alias: Display name for the model (defaults to model_name)
            task_ids: Optional list of specific tasks to run (None = all tasks)
            **kwargs: Additional evaluation parameters
            
        Returns:
            BenchResult with evaluation results
        """
        start_time = datetime.now(timezone.utc)
        model_alias = model_alias or model_name
        
        # Validate model format
        if "/" not in model_name:
            error_msg = f"Model name must be in provider/model format, got: {model_name}"
            logger.error(error_msg)
            return self._create_error_result(model_name, model_alias, start_time, error_msg)
        
        # Check if this is a local model and get configuration
        is_local = False
        local_config = None
        if model_name in LOCAL_MODEL_CONFIGS:
            is_local = True
            local_config = LOCAL_MODEL_CONFIGS[model_name]
        
        # Validate local server is running
        if not validate_local_server(model_name):
            error_msg = f"Local server not running for {model_name}. Run 'make start-local-model-server MODEL={model_name}' first."
            logger.error(error_msg)
            return self._create_error_result(model_name, model_alias, start_time, error_msg)
        
        # Load tasks
        tasks = self._load_tasks(task_ids)
        if not tasks:
            error_msg = "No tasks loaded for evaluation"
            logger.error(error_msg)
            return self._create_error_result(model_name, model_alias, start_time, error_msg)
        
        logger.info(f"Starting NL2Bash evaluation with {len(tasks)} tasks on model: {model_name}")
        
        try:
            # Create inspect_ai task
            inspect_task = self._create_inspect_task(tasks, model_name)
            
            # For local models, keep the provider prefix and use model_base_url to route to local server
            eval_model_name = model_name
            
            # Prepare eval parameters
            eval_params = {
                "model": eval_model_name,
                "log_dir": str(self.output_dir / "inspect_logs")
            }
            
            # Add base URL for local models
            if is_local:
                eval_params["model_base_url"] = local_config["base_url"]
            
            # Run evaluation
            eval_result = inspect_eval(
                inspect_task, 
                retry_on_error=3,  # Retry failed samples up to 3 times
                fail_on_error=0.1,  # Tolerate up to 10% sample failures
                **eval_params
            )
            
            # Parse results into Run objects
            runs = self._parse_inspect_results(eval_result, tasks, model_name, model_alias)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(runs)
            
            # Add NL2Bash-specific stats
            summary_stats.update(self._calculate_nl2bash_stats(runs, tasks))
            
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
                    "num_tasks": len(tasks),
                    "inspect_ai_version": inspect_ai.__version__,
                    "scoring_method": "llm_functional_equivalence",
                    "log_dir": str(self.output_dir / "inspect_logs"),
                    "is_local_model": is_local
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
        tasks_file = self.dataset_dir.parent.parent / "processed" / self.dataset_name / f"{self.dataset_name}_tasks.jsonl"
        
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
            filtered_tasks = [t for t in all_tasks if t.get('task_id') in task_id_set]
            logger.info(f"Filtered to {len(filtered_tasks)} tasks from {len(all_tasks)} total")
            return filtered_tasks
        
        return all_tasks
    
    def _create_inspect_task(self, tasks: List[Dict[str, Any]], model_name: str):
        """Create inspect_ai Task for NL2Bash evaluation."""
        # Check if this is a completion model (like GPT-2) vs a chat model
        is_completion_model = any(model_part in model_name.lower() 
                                for model_part in ['gpt2', 'gpt-2', 'davinci-002'])
        
        if is_completion_model:
            return self._create_completion_task(tasks)
        else:
            return self._create_chat_task(tasks)
    
    def _create_completion_task(self, tasks: List[Dict[str, Any]]):
        """Create task for completion models like GPT-2."""
        samples = []
        
        # Few-shot prefix that will be prepended to each user input
        few_shot_prefix = (
            "Natural language: Show all files in the current directory\n"
            "Bash command: ls\n\n"
            "Natural language: Display the contents of README.txt\n"
            "Bash command: cat README.txt\n\n"
            "Natural language: Find all Python files in the current directory\n"
            "Bash command: find . -name '*.py'\n\n"
            "Natural language: Create a new directory called 'backup'\n"
            "Bash command: mkdir backup\n\n"
            "Natural language: "
        )
        
        for task_data in tasks:
            metadata = task_data.get('dataset_task_metadata', {})
            nl_description = metadata.get('nl_description', '')
            
            # Combine few-shot examples with the specific task
            completion_input = few_shot_prefix + nl_description + "\nBash command:"
            
            sample = Sample(
                id=task_data['task_id'],
                input=completion_input,
                target=metadata.get('bash_command', ''),
                metadata={
                    'complexity_category': metadata.get('complexity_category'),
                    'complexity_score': metadata.get('complexity_score'),
                    'human_minutes': task_data.get('human_minutes'),
                    'task_family': task_data.get('task_family')
                }
            )
            samples.append(sample)
        
        @task
        def nl2bash_completion_task():
            return Task(
                dataset=samples,
                plan=[
                    generate(max_tokens=50, stop_seqs=["\n", "\n\n", "Natural language:"])
                ],
                scorer=self._create_llm_scorer()
            )
        
        return nl2bash_completion_task()
    
    def _create_chat_task(self, tasks: List[Dict[str, Any]]):
        """Create task for chat models like GPT-4."""
        samples = []
        for task_data in tasks:
            metadata = task_data.get('dataset_task_metadata', {})
            
            sample = Sample(
                id=task_data['task_id'],
                input=metadata.get('nl_description', ''),
                target=metadata.get('bash_command', ''),
                metadata={
                    'complexity_category': metadata.get('complexity_category'),
                    'complexity_score': metadata.get('complexity_score'),
                    'human_minutes': task_data.get('human_minutes'),
                    'task_family': task_data.get('task_family')
                }
            )
            samples.append(sample)
        
        @task
        def nl2bash_chat_task():
            return Task(
                dataset=samples,
                plan=[
                    system_message(
                        "You are an expert Linux system administrator tasked with generating a precise bash command. "
                        "Carefully analyze the natural language description provided by the user. "
                        "Your generated command should accurately and completely fulfill all aspects of the user's request. "
                        "Pay close attention to any specific file names, directory paths, variables (like $1, $HOST), or other literal values mentioned in the description, and use them exactly as provided in your command. "
                        "Ensure your command is safe and directly achieves the described task. "
                        "Respond with ONLY the bash command, no explanations, apologies, or any surrounding text."
                    ),
                    generate()
                ],
                scorer=self._create_llm_scorer()
            )
        
        return nl2bash_chat_task()
    
    def _create_llm_scorer(self):
        """Create LLM scorer for functional equivalence evaluation."""
        from inspect_ai.scorer import scorer, Score, Target, accuracy
        from human_ttc_eval.core.llm_utils import LLMClient, LLMConfig
        
        @scorer(metrics=[accuracy()])
        def llm_bash_scorer():
            async def score(state, target: Target):
                # Extract generated command
                if state.output and state.output.completion:
                    generated = state.output.completion.strip()
                else:
                    return Score(value=0.0, explanation="No output generated")
                
                target_cmd = target.text.strip()
                task_description = state.input_text
                
                # Use LLM to evaluate functional equivalence
                try:
                    config_obj = LLMConfig(
                        provider="openai",
                        model="o4-mini-2025-04-16",
                        max_tokens=100,
                        temperature=0.0
                    )
                    
                    client = LLMClient(config_obj)
                    
                    system_prompt = (
                        "You are an expert Linux system administrator and AI evaluation assistant. "
                        "Your task is to evaluate a bash command generated by an AI model based on a natural language (NL) task description. "
                        "A target command is provided as an example of a correct solution, but the generated command does not need to match it exactly. "
                        "The key is functional equivalence: does the generated command correctly and completely achieve the goal stated in the NL task description?"
                        "Consider different ways to achieve the same result (e.g., different flag orders, equivalent commands, alternative utilities that produce the same outcome)."
                        "Be precise in your scoring. If the command is only partially correct or misses aspects of the NL description, reflect that in the score."
                        "Respond with ONLY a single floating-point score between 0.0 and 1.0, where:\n"
                        "1.0 = Functionally equivalent to the NL description, or a clear improvement on the target command for achieving the NL task.\n"
                        "0.8 = Mostly correct and functionally equivalent, with only very minor deviations or perhaps a slightly less optimal but still correct approach to the NL task.\n"
                        "0.5 = Partially correct; achieves some aspects of the NL task but misses key requirements or has significant flaws.\n"
                        "0.2 = Tangentially related or attempts the task but is fundamentally incorrect or incomplete for the NL task.\n"
                        "0.0 = Completely incorrect, unsafe, or irrelevant to the NL task."
                    )
                    
                    user_prompt = (
                        f"Task: {task_description}\n"
                        f"Target: {target_cmd}\n"
                        f"Generated: {generated}\n"
                        f"Score:"
                    )
                    
                    response = client.call(user_prompt, system_prompt)
                    
                    # Parse score
                    try:
                        score_value = float(response.strip())
                        score_value = max(0.0, min(1.0, score_value))
                    except ValueError:
                        # Exact match fallback
                        score_value = 1.0 if generated == target_cmd else 0.0
                        logger.warning(f"Failed to parse LLM score: {response}")
                    
                    return Score(value=score_value, explanation=f"LLM score: {score_value}")
                    
                except Exception as e:
                    logger.warning(f"LLM scoring failed: {e}, using exact match")
                    score_value = 1.0 if generated == target_cmd else 0.0
                    return Score(value=score_value, explanation=f"Exact match: {score_value}")
            
            return score
        
        return llm_bash_scorer()
    
    def _parse_inspect_results(
        self, 
        eval_result, 
        tasks: List[Dict[str, Any]],
        model_name: str,
        model_alias: str
    ) -> List[Run]:
        """Parse inspect_ai results into Run objects."""
        runs = []
        
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
            
            # Extract score
            score_value = 0.0
            if hasattr(sample, 'scores') and sample.scores:
                if 'llm_bash_scorer' in sample.scores:
                    score_obj = sample.scores['llm_bash_scorer']
                    if hasattr(score_obj, 'value'):
                        score_value = score_obj.value
                    elif isinstance(score_obj, (float, int)):
                        score_value = float(score_obj)
            
            # Create Run object
            run = Run(
                task_id=task_id,
                task_family=task_data.get('task_family', self.dataset_name),
                run_id=f"{model_name.replace('/', '_')}_{task_id.replace('/', '_')}_{uuid.uuid4().hex[:8]}",
                alias=model_alias,
                model=model_name,
                score_binarized=1 if score_value >= 0.8 else 0,  # 0.8+ considered success
                score_cont=score_value,
                human_minutes=self._get_human_minutes_for_task(task_id),
                human_source="baseline",
                task_source=self.dataset_name,
                started_at=0.0,
                completed_at=0.0,  # inspect_ai doesn't provide timing
                generation_cost=0.0,  # Would need token counting
                fatal_error_from=None
            )
            runs.append(run)
        
        return runs
    
    def _calculate_nl2bash_stats(self, runs: List[Run], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate NL2Bash-specific statistics."""
        # Group by complexity category
        complexity_stats = {}
        
        for run in runs:
            # Find task to get complexity category
            task = next((t for t in tasks if t['task_id'] == run.task_id), None)
            if task:
                metadata = task.get('dataset_task_metadata', {})
                category = metadata.get('complexity_category', 'unknown')
                
                if category not in complexity_stats:
                    complexity_stats[category] = {
                        'total': 0,
                        'successful': 0,
                        'scores': []
                    }
                
                complexity_stats[category]['total'] += 1
                if run.score_binarized == 1:
                    complexity_stats[category]['successful'] += 1
                complexity_stats[category]['scores'].append(run.score_cont)
        
        # Calculate success rates per category
        for category, stats in complexity_stats.items():
            stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0.0
            stats['avg_score'] = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0.0
            del stats['scores']  # Remove raw scores from final stats
        
        return {
            'complexity_breakdown': complexity_stats,
            'scoring_method': 'llm_functional_equivalence',
            'success_threshold': 0.8
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