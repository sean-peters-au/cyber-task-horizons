from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Standardized result format for all benchmark runs."""
    dataset_name: str
    model_name: str
    task_results: List[Dict[str, Any]]  # List of individual task results
    summary_stats: Dict[str, Any]       # Aggregated statistics
    metadata: Dict[str, Any]            # Run metadata (duration, params, etc.)
    raw_output_path: Optional[Path]     # Path to raw evaluation logs/outputs
    timestamp: str
    success: bool
    error_message: Optional[str] = None

class BaseBench(ABC):
    """
    Base class for benchmark evaluation systems.
    
    Supports both external evaluation systems (like CyBench) and 
    inspect_ai-based evaluations through a common interface.
    """
    
    def __init__(self, dataset_name: str, output_dir: Path):
        """
        Initialize the benchmark runner.
        
        Args:
            dataset_name: Name of the dataset being evaluated
            output_dir: Directory to store evaluation results and logs
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized {self.__class__.__name__} for dataset: {dataset_name}")
        
    @abstractmethod
    def run_evaluation(
        self, 
        model_name: str, 
        tasks: Optional[List[str]] = None,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run evaluation for the specified model and tasks.
        
        Args:
            model_name: Name/identifier of the model to evaluate
            tasks: Optional list of specific tasks to run (None = all tasks)
            **kwargs: Additional evaluation parameters (max_iterations, etc.)
            
        Returns:
            BenchmarkResult with standardized evaluation results
        """
        pass
    
    @abstractmethod
    def list_available_tasks(self) -> List[str]:
        """
        List all available tasks for this dataset.
        
        Returns:
            List of task identifiers
        """
        pass
    
    @abstractmethod
    def validate_model_name(self, model_name: str) -> bool:
        """
        Validate that the model name is supported by this evaluation system.
        
        Args:
            model_name: Model identifier to validate
            
        Returns:
            True if model is supported, False otherwise
        """
        pass
    
    def save_result(self, result: BenchmarkResult) -> Path:
        """
        Save benchmark result to standardized JSON format.
        
        Args:
            result: BenchmarkResult to save
            
        Returns:
            Path to saved result file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_{result.model_name.replace('/', '_')}_{timestamp}.json"
        result_path = self.output_dir / filename
        
        # Convert result to dict for JSON serialization
        result_dict = {
            "dataset_name": result.dataset_name,
            "model_name": result.model_name,
            "task_results": result.task_results,
            "summary_stats": result.summary_stats,
            "metadata": result.metadata,
            "raw_output_path": str(result.raw_output_path) if result.raw_output_path else None,
            "timestamp": result.timestamp,
            "success": result.success,
            "error_message": result.error_message
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Benchmark result saved to: {result_path}")
        return result_path
    
    def load_result(self, result_path: Path) -> BenchmarkResult:
        """
        Load a previously saved benchmark result.
        
        Args:
            result_path: Path to saved result JSON file
            
        Returns:
            BenchmarkResult object
        """
        with open(result_path, 'r', encoding='utf-8') as f:
            result_dict = json.load(f)
            
        return BenchmarkResult(
            dataset_name=result_dict["dataset_name"],
            model_name=result_dict["model_name"],
            task_results=result_dict["task_results"],
            summary_stats=result_dict["summary_stats"],
            metadata=result_dict["metadata"],
            raw_output_path=Path(result_dict["raw_output_path"]) if result_dict["raw_output_path"] else None,
            timestamp=result_dict["timestamp"],
            success=result_dict["success"],
            error_message=result_dict.get("error_message")
        ) 