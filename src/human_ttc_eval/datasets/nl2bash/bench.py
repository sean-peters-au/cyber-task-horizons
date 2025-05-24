"""
NL2Bash benchmark using inspect_ai for AI model evaluation.

This benchmark evaluates how well AI models can generate bash commands
from natural language descriptions using the NL2Bash dataset.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from ...core.inspect_bench import InspectBench
from ...core.base_bench import BenchmarkResult
from ...core.registry import register_bench

logger = logging.getLogger(__name__)

@register_bench("nl2bash")
class NL2BashBench(InspectBench):
    """
    Benchmark runner for NL2Bash dataset using inspect_ai.
    
    Evaluates AI models on their ability to generate correct bash commands
    from natural language descriptions.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize NL2Bash benchmark runner.
        
        Args:
            output_dir: Directory to store evaluation results
        """
        super().__init__("nl2bash", output_dir)
        
        # Load tasks from the standard location
        project_root = Path(__file__).parent.parent.parent.parent.parent
        tasks_file = project_root / "data" / "processed" / "nl2bash" / "all_tasks.jsonl"
        
        if not tasks_file.exists():
            raise FileNotFoundError(f"NL2Bash tasks file not found: {tasks_file}")
        
        self.tasks = self._load_tasks(tasks_file)
        logger.info(f"Loaded {len(self.tasks)} NL2Bash tasks")
    
    def _load_tasks(self, tasks_file: Path) -> List[Dict[str, Any]]:
        """Load NL2Bash tasks from the parsed JSONL file."""
        tasks = []
        with open(tasks_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    tasks.append(task)
        return tasks
    
    def list_available_tasks(self) -> List[str]:
        """List all available NL2Bash task IDs."""
        return [task["task_id"] for task in self.tasks]
    
    def _create_inspect_task(self):
        """Create inspect_ai Task for NL2Bash evaluation."""
        from inspect_ai import Task, task
        from inspect_ai.dataset import Sample
        from inspect_ai.solver import generate, system_message
        
        # Convert tasks to inspect_ai samples
        samples = []
        for task_data in self.tasks:
            sample = Sample(
                id=task_data["task_id"],
                input=task_data["nl_description"],
                target=task_data["bash_command"],
                metadata={
                    "complexity_category": task_data.get("_raw_complexity_category"),
                    "complexity_score": task_data.get("complexity_score"),
                    "human_minutes": task_data.get("human_minutes"),
                    "task_family": task_data["task_family"]
                }
            )
            samples.append(sample)
        
        @task
        def nl2bash_task():
            return Task(
                dataset=samples,
                plan=[
                    system_message(
                        "You are an expert Linux system administrator. "
                        "Given a natural language description of what a user wants to accomplish, "
                        "generate the appropriate bash command. "
                        "Respond with ONLY the bash command, no explanations or formatting."
                    ),
                    generate()
                ],
                scorer=[
                    self._create_llm_scorer()
                ]
            )
        
        return nl2bash_task()
    
    def _create_llm_scorer(self):
        """Create LLM scorer using o4-mini for bash command evaluation."""
        from inspect_ai.scorer import scorer, Score, Target, accuracy
        
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
                    from ...core.llm_utils import LLMClient, LLMConfig
                    
                    config = LLMConfig(
                        provider="openai",
                        model="o4-mini-2025-04-16"
                    )
                    
                    client = LLMClient(config)
                    
                    system_prompt = (
                        "You are an expert Linux system administrator. "
                        "Evaluate if the generated bash command is functionally equivalent to the target command for the given task. "
                        "Consider different ways to achieve the same result (e.g., different flag orders, equivalent commands). "
                        "Respond with ONLY a score between 0.0 and 1.0 where:\n"
                        "1.0 = Functionally equivalent or better\n"
                        "0.8 = Mostly correct, minor differences\n"
                        "0.5 = Partially correct\n"
                        "0.0 = Incorrect or completely wrong"
                    )
                    
                    user_prompt = (
                        f"Task: {task_description}\n"
                        f"Target command: {target_cmd}\n"
                        f"Generated command: {generated}\n\n"
                        f"Score:"
                    )
                    
                    response = client.call(user_prompt, system_prompt)
                    
                    # Parse score from response
                    try:
                        score_value = float(response.strip())
                        score_value = max(0.0, min(1.0, score_value))  # Clamp to [0,1]
                    except ValueError:
                        # If parsing fails, fall back to exact match
                        score_value = 1.0 if generated == target_cmd else 0.0
                        logger.warning(f"Failed to parse LLM score: {response}, using exact match")
                    
                    explanation = f"LLM score: {score_value} (target: {target_cmd})"
                    return Score(value=score_value, explanation=explanation)
                    
                except Exception as e:
                    logger.warning(f"LLM scoring failed: {e}, falling back to exact match")
                    # Fall back to exact match if LLM fails
                    score_value = 1.0 if generated == target_cmd else 0.0
                    return Score(value=score_value, explanation=f"Exact match fallback: {score_value}")
            
            return score
        
        return llm_bash_scorer()
    
    def _parse_inspect_results(self, eval_result) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parse inspect_ai evaluation results with NL2Bash-specific logic."""
        # Use the base parsing first
        task_results, summary_stats = super()._parse_inspect_results(eval_result)
        
        # Add NL2Bash-specific parsing
        try:
            samples = []
            if hasattr(eval_result, '__iter__') and hasattr(eval_result, '__len__'):
                for eval_log in eval_result:
                    if hasattr(eval_log, 'samples') and eval_log.samples:
                        samples.extend(eval_log.samples)
            else:
                # Fallback methods
                samples = getattr(eval_result, 'samples', [])
            
            # Enhanced task results with NL2Bash-specific fields
            enhanced_task_results = []
            for i, sample_result in enumerate(samples):
                if i >= len(self.tasks):
                    break  # Safety check
                    
                task_data = self.tasks[i]
                
                # Extract LLM score
                scores = sample_result.scores
                llm_score = 0.0
                if scores and 'llm_bash_scorer' in scores:
                    llm_score = scores['llm_bash_scorer'].value
                
                # Get generated output
                generated_command = ""
                if sample_result.output and sample_result.output.completion:
                    generated_command = sample_result.output.completion.strip()
                
                task_result = {
                    "task_id": task_data["task_id"],
                    "task_family": task_data["task_family"],
                    "complexity_category": task_data.get("_raw_complexity_category", "unknown"),
                    "nl_description": task_data["nl_description"],
                    "target_command": task_data["bash_command"],
                    "generated_command": generated_command,
                    "llm_score": llm_score,
                    "success": llm_score >= 0.8,  # Consider 0.8+ as success
                    "complexity_score": task_data.get("complexity_score", 0.0),
                    "human_minutes": task_data.get("human_minutes", 0.0)
                }
                
                enhanced_task_results.append(task_result)
            
            # Calculate NL2Bash-specific summary statistics
            if enhanced_task_results:
                total_tasks = len(enhanced_task_results)
                successful_tasks = sum(1 for r in enhanced_task_results if r["success"])
                
                # Group by complexity
                complexity_stats = {}
                for result in enhanced_task_results:
                    category = result["complexity_category"]
                    if category not in complexity_stats:
                        complexity_stats[category] = {"total": 0, "successful": 0, "avg_score": 0.0}
                    
                    complexity_stats[category]["total"] += 1
                    if result["success"]:
                        complexity_stats[category]["successful"] += 1
                    complexity_stats[category]["avg_score"] += result["llm_score"]
                
                # Finalize complexity stats
                for category in complexity_stats:
                    stats = complexity_stats[category]
                    stats["success_rate"] = stats["successful"] / stats["total"]
                    stats["avg_score"] = stats["avg_score"] / stats["total"]
                
                enhanced_summary_stats = {
                    "total_tasks": total_tasks,
                    "successful_tasks": successful_tasks,
                    "success_rate": successful_tasks / total_tasks,
                    "average_llm_score": sum(r["llm_score"] for r in enhanced_task_results) / total_tasks,
                    "complexity_breakdown": complexity_stats
                }
            else:
                enhanced_summary_stats = {
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "success_rate": 0.0,
                    "average_llm_score": 0.0,
                    "complexity_breakdown": {}
                }
            
            return enhanced_task_results, enhanced_summary_stats
            
        except Exception as e:
            logger.error(f"Failed to parse NL2Bash results: {e}", exc_info=True)
            # Fall back to base parsing
            return task_results, summary_stats
    
    def get_complexity_breakdown(self, result: BenchmarkResult):
        """
        Get NL2Bash-specific complexity analysis.
        
        Args:
            result: BenchmarkResult from NL2Bash evaluation
            
        Returns:
            Dictionary with complexity-based analysis
        """
        if result.framework == "inspect_ai":
            # Use inspect_ai analysis tools
            try:
                analysis = self.get_inspect_analysis(result)
                df = analysis.get("samples_dataframe")
                
                if df is not None:
                    # Group by complexity category
                    complexity_analysis = df.groupby('metadata.complexity_category').agg({
                        'score_accuracy': ['mean', 'count']
                    }).round(3)
                    
                    return {
                        "complexity_breakdown": complexity_analysis.to_dict(),
                        "inspect_view_command": analysis.get("inspect_view_command")
                    }
            except Exception as e:
                logger.warning(f"Could not generate complexity breakdown: {e}")
        
        # Fall back to summary stats if inspect analysis fails
        return result.summary_stats.get("complexity_breakdown", {}) 