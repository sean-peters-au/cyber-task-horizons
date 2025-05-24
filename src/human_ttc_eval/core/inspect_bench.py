"""
Base class for benchmarks using inspect_ai framework.

This provides common functionality for all inspect_ai-based evaluations
while keeping framework-specific logic separate from other benchmark types.
"""

import logging
from abc import ABC
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from .base_bench import BaseBench, BenchmarkResult

logger = logging.getLogger(__name__)


class InspectBench(BaseBench, ABC):
    """
    Base class for benchmarks using inspect_ai framework.
    
    Provides common inspect_ai functionality while allowing dataset-specific
    customization in subclasses.
    """
    
    def run_evaluation(self, model_name: str, **kwargs) -> BenchmarkResult:
        """
        Run inspect_ai evaluation with framework-specific result preservation.
        
        Args:
            model_name: Model identifier (provider/model format)
            **kwargs: Additional parameters passed to specific implementation
            
        Returns:
            BenchmarkResult with both unified and native inspect_ai results
        """
        start_time = datetime.now()
        
        if not self.validate_model_name(model_name):
            error_msg = f"Model '{model_name}' not supported or incorrectly formatted"
            return self._create_error_result(model_name, start_time, error_msg)
        
        try:
            # Import inspect_ai
            import inspect_ai as ai
            from inspect_ai import eval
            
            # Create inspect_ai task (implemented by subclasses)
            inspect_task = self._create_inspect_task()
            
            # Run evaluation
            logger.info(f"Starting inspect_ai evaluation for model: {model_name}")
            eval_result = eval(
                inspect_task,
                model=model_name,
                log_dir=str(self.output_dir / "inspect_logs")
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results for unified interface
            task_results, summary_stats = self._parse_inspect_results(eval_result)
            
            # Get native log path
            native_log_path = None
            if hasattr(eval_result, '__iter__') and hasattr(eval_result, '__len__'):
                if len(eval_result) > 0:
                    first_log = eval_result[0]
                    if hasattr(first_log, 'location'):
                        native_log_path = first_log.location
            elif hasattr(eval_result, 'logs') and eval_result.logs:
                native_log_path = eval_result.logs[0]
            
            # Extract serializable subset of eval_result for native_results
            native_results = self._extract_serializable_eval_data(eval_result)
            
            # Create result with both unified and native data
            benchmark_result = BenchmarkResult(
                dataset_name=self.dataset_name,
                model_name=model_name,
                task_results=task_results,
                summary_stats=summary_stats,
                metadata={
                    "duration_seconds": duration,
                    "sample_size": len(task_results) if task_results else 0,
                    "inspect_ai_version": ai.__version__
                },
                raw_output_path=self.output_dir / "inspect_logs",
                timestamp=start_time.isoformat(),
                success=True,
                error_message=None,
                framework="inspect_ai",
                native_results=native_results,
                native_log_path=native_log_path
            )
            
            # Save result
            self.save_result(benchmark_result)
            return benchmark_result
            
        except ImportError as e:
            error_msg = f"inspect_ai not available: {e}"
            return self._create_error_result(model_name, start_time, error_msg)
        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_result(model_name, start_time, error_msg)
    
    def _create_inspect_task(self):
        """Create inspect_ai Task for evaluation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_inspect_task")
    
    def _parse_inspect_results(self, eval_result) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Parse inspect_ai evaluation results into unified format.
        
        This provides a default implementation that can be overridden by subclasses
        for dataset-specific parsing logic.
        """
        try:
            # Import here to avoid dependency issues
            from inspect_ai.log import read_eval_log
            
            # Extract samples from eval_result
            samples = []
            
            # Handle different eval_result types
            if hasattr(eval_result, '__iter__') and hasattr(eval_result, '__len__'):
                for eval_log in eval_result:
                    if hasattr(eval_log, 'samples') and eval_log.samples:
                        samples.extend(eval_log.samples)
            elif hasattr(eval_result, 'logs') and eval_result.logs:
                # Original approach - if eval_result has logs attribute
                log_path = eval_result.logs[0]
                eval_log = read_eval_log(log_path)
                samples = eval_log.samples
            elif hasattr(eval_result, 'samples'):
                # Direct samples access
                samples = eval_result.samples
            else:
                # Fallback: try to access samples directly if available
                samples = getattr(eval_result, 'samples', [])
            
            task_results = []
            for i, sample_result in enumerate(samples):
                # Extract basic information that should be common across datasets
                task_result = {
                    "sample_id": getattr(sample_result, 'id', i),
                    "input": getattr(sample_result, 'input', ''),
                    "target": getattr(sample_result, 'target', ''),
                    "output": getattr(sample_result.output, 'completion', '') if sample_result.output else '',
                    "scores": dict(sample_result.scores) if hasattr(sample_result, 'scores') and sample_result.scores else {},
                    "metadata": dict(sample_result.metadata) if hasattr(sample_result, 'metadata') and sample_result.metadata else {}
                }
                task_results.append(task_result)
            
            # Calculate basic summary statistics
            summary_stats = {
                "total_samples": len(task_results),
                "completed_samples": len([r for r in task_results if r.get("output")]),
            }
            
            # Add score-based statistics if available
            if task_results and any(r.get("scores") for r in task_results):
                # This is generic - subclasses can override for dataset-specific metrics
                accuracy_scores = [
                    score.value for r in task_results 
                    for score_name, score in r.get("scores", {}).items()
                    if score_name == "accuracy" and hasattr(score, 'value')
                ]
                if accuracy_scores:
                    summary_stats["accuracy"] = sum(accuracy_scores) / len(accuracy_scores)
                    summary_stats["success_rate"] = summary_stats["accuracy"]
            
            return task_results, summary_stats
            
        except Exception as e:
            logger.error(f"Failed to parse inspect_ai results: {e}", exc_info=True)
            return [], {"error": f"Result parsing failed: {e}"}
    
    def get_inspect_analysis(self, result: BenchmarkResult):
        """
        Get inspect_ai specific analysis helpers.
        
        Args:
            result: BenchmarkResult from an inspect_ai evaluation
            
        Returns:
            Dictionary with inspect_ai analysis tools and data
        """
        if result.framework != "inspect_ai":
            raise ValueError("This method only works with inspect_ai results")
        
        analysis = {
            "framework": "inspect_ai",
            "native_results_available": result.native_results is not None,
            "native_log_path": result.native_log_path,
        }
        
        if result.native_log_path:
            analysis["inspect_view_command"] = f"inspect view {result.native_log_path.parent}"
            
            try:
                from inspect_ai.analysis import samples_df
                df = samples_df(result.native_log_path)
                analysis["samples_dataframe"] = df
                analysis["sample_count"] = len(df)
            except Exception as e:
                logger.warning(f"Could not load samples dataframe: {e}")
                analysis["samples_dataframe_error"] = str(e)
        
        return analysis
    
    def validate_model_name(self, model_name: str) -> bool:
        """Validate model name for inspect_ai compatibility."""
        return "/" in model_name
    
    def _create_error_result(self, model_name: str, start_time: datetime, error_msg: str) -> BenchmarkResult:
        """Create a BenchmarkResult for error cases."""
        return BenchmarkResult(
            dataset_name=self.dataset_name,
            model_name=model_name,
            task_results=[],
            summary_stats={},
            metadata={"error": error_msg},
            raw_output_path=None,
            timestamp=start_time.isoformat(),
            success=False,
            error_message=error_msg,
            framework="inspect_ai",
            native_results=None,
            native_log_path=None
        )

    def _extract_serializable_eval_data(self, eval_result):
        """
        Extract a serializable subset of eval_result for native_results.
        
        Args:
            eval_result: The inspect_ai EvalResult/EvalLog object
            
        Returns:
            Dict with JSON-serializable inspect_ai-specific data
        """
        try:
            native_data = {
                "framework": "inspect_ai",
                "eval_type": type(eval_result).__name__,
            }
            
            # Basic eval metadata
            if hasattr(eval_result, 'eval_id'):
                native_data["eval_id"] = str(eval_result.eval_id)
            
            if hasattr(eval_result, 'model'):
                native_data["model"] = str(eval_result.model)
                
            if hasattr(eval_result, 'dataset'):
                native_data["dataset"] = str(eval_result.dataset)
            
            # Stats and metrics
            if hasattr(eval_result, 'stats'):
                stats = eval_result.stats
                if stats:
                    stats_dict = {}
                    # Extract basic numeric stats that are JSON serializable
                    for attr in ['total_time', 'total_cost', 'samples_completed']:
                        if hasattr(stats, attr):
                            value = getattr(stats, attr)
                            if isinstance(value, (int, float, str)):
                                stats_dict[attr] = value
                    native_data["stats"] = stats_dict
            
            # Sample count info
            if hasattr(eval_result, 'samples'):
                native_data["sample_count"] = len(eval_result.samples)
            
            # Log paths
            if hasattr(eval_result, 'logs'):
                native_data["log_paths"] = [str(path) for path in eval_result.logs]
            
            # Config info (convert to string to avoid complex object serialization)
            if hasattr(eval_result, 'eval_config'):
                native_data["eval_config_str"] = str(eval_result.eval_config)
                
            return native_data
            
        except Exception as e:
            logger.warning(f"Failed to extract serializable eval data: {e}")
            # Return minimal data if extraction fails
            return {
                "framework": "inspect_ai", 
                "extraction_error": str(e),
                "eval_type": type(eval_result).__name__
            }
    