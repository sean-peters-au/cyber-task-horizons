"""
Unit tests for the Bench base class.

Tests the new benchmarking interface that produces Run objects.
"""

import pytest
from pathlib import Path
import tempfile
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from human_ttc_eval.core.bench import Bench, BenchResult
from human_ttc_eval.core.run import Run


class MockBench(Bench):
    """Mock implementation of Bench for testing the base class functionality."""
    
    @property
    def dataset_name(self) -> str:
        return "mock_dataset"
    
    def list_available_tasks(self) -> List[str]:
        """Return a fixed list of task IDs."""
        return ["task_1", "task_2", "task_3"]
    
    def run_evaluation(
        self,
        model_name: str,
        model_alias: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
        **kwargs
    ) -> BenchResult:
        """Mock evaluation that creates a few Run objects."""
        if model_alias is None:
            model_alias = model_name

        tasks_to_run = task_ids if task_ids is not None else self.list_available_tasks()
        runs: List[Run] = []
        eval_start_time = datetime.now(timezone.utc)

        for task_id in tasks_to_run:
            human_minutes = self._get_human_minutes_for_task(task_id)
            task_family = self._get_task_family_for_task(task_id)
            score = 1 if task_id != "task_2" else 0
            score_cont = 1.0 if task_id != "task_2" else 0.3 
            gen_cost = 0.05 if task_id == "task_1" else (0.03 if task_id == "task_2" else 0.04)

            runs.append(Run(
                task_id=task_id,
                task_family=task_family,
                run_id=f"{model_name.replace('/', '_')}_{task_id}_mockrun",
                alias=model_alias,
                model=model_name,
                score_binarized=score,
                score_cont=score_cont,
                human_minutes=human_minutes,
                human_source="baseline",
                task_source=self.dataset_name,
                generation_cost=gen_cost,
                started_at=0.0,
                completed_at=float(kwargs.get(f"{task_id}_time_sec", 60.0))
            ))
        
        eval_end_time = datetime.now(timezone.utc)
        duration_seconds = (eval_end_time - eval_start_time).total_seconds()

        summary_stats = self._calculate_summary_stats(runs)
        metadata = {
            "duration_seconds": duration_seconds,
            "num_tasks_evaluated": len(tasks_to_run),
            "custom_kwargs": kwargs
        }

        return BenchResult(
            dataset_name=self.dataset_name,
            model_name=model_name,
            model_alias=model_alias,
            runs=runs,
            summary_stats=summary_stats,
            metadata=metadata,
            timestamp=eval_start_time.isoformat(),
            success=True,
            error_message=None
        )


def create_test_human_baseline() -> List[Run]:
    """Create human baseline runs for testing."""
    return [
        Run(
            task_id="task_1", task_family="mock_family", run_id="human_task_1", alias="Human", 
            model="human", score_binarized=1, human_minutes=10.0, human_score=1.0, 
            human_source="test", task_source="mock_dataset", started_at=0.0, completed_at=600.0
        ),
        Run(
            task_id="task_2", task_family="mock_family", run_id="human_task_2", alias="Human", 
            model="human", score_binarized=1, human_minutes=15.0, human_score=1.0, 
            human_source="test", task_source="mock_dataset", started_at=0.0, completed_at=900.0
        ),
        Run(
            task_id="task_3", task_family="mock_family", run_id="human_task_3", alias="Human", 
            model="human", score_binarized=0, human_minutes=20.0, human_score=0.0, 
            human_source="test", task_source="mock_dataset", started_at=0.0, completed_at=1200.0
        ),
    ]


@pytest.fixture
def temp_output_dirs():
    """Creates temporary directories for testing that mimics project structure."""
    with tempfile.TemporaryDirectory() as temp_dir_base:
        base_path = Path(temp_dir_base)
        
        # Mimic structure for dataset related files (raw, processed)
        # dataset_root will be effectively 'temp_dir_base/data'
        # and processed files will go into 'temp_dir_base/data/processed/mock_dataset'
        dataset_root_for_files = base_path / "data"
        
        # Mimic structure for benchmark output files
        # bench_output_dir will be 'temp_dir_base/results/benchmarks/mock_dataset'
        # This is where Bench.save_result would typically save.
        # However, MockBench's output_dir is initialized at a higher level,
        # so we will return 'base_path / "results"' for bench.output_dir
        # and the tests can construct the full path if needed.
        bench_overall_output_dir = base_path / "results"
        
        # Ensure subdirectories for processed data exist, as Bench class expects
        # dataset_dir.parent.parent / "processed" / self.dataset_name
        # For MockBench, dataset_dir is data/raw/mock_dataset
        # so dataset_dir.parent.parent is 'data'
        (dataset_root_for_files / "processed" / "mock_dataset").mkdir(parents=True, exist_ok=True)
        (dataset_root_for_files / "raw" / "mock_dataset").mkdir(parents=True, exist_ok=True)
        
        # The Bench class takes output_dir which is usually results/benchmarks/<dataset_name>
        # but for testing, we'll pass the parent 'results' and let the Bench class create subdirs
        yield dataset_root_for_files, bench_overall_output_dir


class TestBench:
    """Test the Bench base class."""
    
    def test_initialization(self):
        """Test successful initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "data" / "raw" / "mock_dataset"
            dataset_dir.mkdir(parents=True)
            output_dir = Path(temp_dir) / "results"
            bench = MockBench(dataset_dir, output_dir)
            assert bench.dataset_dir == dataset_dir
            assert bench.output_dir == output_dir
            assert bench.dataset_name == "mock_dataset"
            assert output_dir.exists()
    
    def test_list_available_tasks(self):
        """Test listing available tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "data" / "raw" / "mock_dataset"
            dataset_dir.mkdir(parents=True)
            output_dir = Path(temp_dir) / "results"
            bench = MockBench(dataset_dir, output_dir)
            tasks = bench.list_available_tasks()
            assert len(tasks) == 3
            assert "task_1" in tasks

    def test_run_evaluation_all_tasks(self):
        """Test running evaluation on all tasks via MockBench.run_evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "data" / "raw" / "mock_dataset"
            dataset_dir.mkdir(parents=True)
            output_dir = Path(temp_dir) / "results"
            bench = MockBench(dataset_dir, output_dir)
            result = bench.run_evaluation(model_name="test_model", model_alias="Test Model")
            
            assert isinstance(result, BenchResult)
            assert result.dataset_name == "mock_dataset"
            assert result.model_name == "test_model"
            assert len(result.runs) == 3
            assert result.success
            assert result.summary_stats["total_tasks"] == 3
            assert result.summary_stats["successful_tasks"] == 2
            assert result.summary_stats["success_rate"] == pytest.approx(2/3)
            assert result.summary_stats["total_generation_cost"] == pytest.approx(0.05 + 0.03 + 0.04)

    def test_run_evaluation_specific_tasks(self):
        """Test running evaluation on specific tasks via MockBench.run_evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "data" / "raw" / "mock_dataset"
            dataset_dir.mkdir(parents=True)
            output_dir = Path(temp_dir) / "results"
            bench = MockBench(dataset_dir, output_dir)
            result = bench.run_evaluation("test_model", task_ids=["task_1", "task_3"])
            
            assert len(result.runs) == 2
            assert all(run.task_id in ["task_1", "task_3"] for run in result.runs)
            assert result.summary_stats["successful_tasks"] == 2
            assert result.summary_stats["success_rate"] == 1.0

    def test_load_human_baseline(self, temp_output_dirs):
        """Test loading human baseline runs."""
        dataset_root, _ = temp_output_dirs
        mock_raw_dir = dataset_root / "raw" / "mock_dataset"
        bench_output_base_dir = dataset_root.parent / "results"
        processed_dir = dataset_root / "processed" / "mock_dataset"
        
        baseline = create_test_human_baseline()
        processed_dir.mkdir(parents=True, exist_ok=True)
        Run.save_jsonl(baseline, str(processed_dir / "mock_dataset_human_runs.jsonl"))
        
        bench = MockBench(mock_raw_dir, bench_output_base_dir)
        assert len(bench.human_baseline) == 3
        assert all(run.model == "human" for run in bench.human_baseline)

    def test_calculate_summary_stats_with_baseline(self, temp_output_dirs):
        """Test summary statistics calculation with human baseline comparison."""
        dataset_root, _ = temp_output_dirs
        mock_raw_dir = dataset_root / "raw" / "mock_dataset"
        bench_output_base_dir = dataset_root.parent / "results"
        processed_dir = dataset_root / "processed" / "mock_dataset"

        baseline = create_test_human_baseline()
        processed_dir.mkdir(parents=True, exist_ok=True)
        Run.save_jsonl(baseline, str(processed_dir / "mock_dataset_human_runs.jsonl"))
        
        bench = MockBench(mock_raw_dir, bench_output_base_dir)
        result = bench.run_evaluation("test_model", task_ids=["task_1", "task_2"])
        
        assert "human_success_rate" in result.summary_stats
        assert result.summary_stats["human_success_rate"] == pytest.approx(2/3)

    def test_create_failed_run(self, temp_output_dirs):
        """Test creating a run for a failed task, ensuring human_minutes are fetched."""
        dataset_root, _ = temp_output_dirs
        mock_raw_dir = dataset_root / "raw" / "mock_dataset"
        bench_output_base_dir = dataset_root.parent / "results"
        processed_dir = dataset_root / "processed" / "mock_dataset"

        baseline = create_test_human_baseline()
        processed_dir.mkdir(parents=True, exist_ok=True)
        Run.save_jsonl(baseline, str(processed_dir / "mock_dataset_human_runs.jsonl"))

        bench = MockBench(mock_raw_dir, bench_output_base_dir)
        run = bench._create_failed_run("task_1", "test_model", "Test Model", "System crash")
        
        assert run.task_id == "task_1"
        assert run.score_binarized == 0
        assert run.fatal_error_from == "System crash"
        assert run.human_minutes == 10.0

    def test_save_and_load_result(self, temp_output_dirs):
        """Test saving and loading benchmark results."""
        dataset_root, bench_overall_output_dir = temp_output_dirs
        mock_raw_dir = dataset_root / "raw" / "mock_dataset"
        
        processed_dir = dataset_root / "processed" / "mock_dataset"
        processed_dir.mkdir(parents=True, exist_ok=True)
        baseline = create_test_human_baseline()
        Run.save_jsonl(baseline, str(processed_dir / "mock_dataset_human_runs.jsonl"))
        
        bench = MockBench(mock_raw_dir, bench_overall_output_dir)
        original_result = bench.run_evaluation("test_model", "Test Model")
        
        saved_dir_path = bench.save_result(original_result)
        assert saved_dir_path.parent == bench_overall_output_dir
        assert saved_dir_path.exists()
        assert (saved_dir_path / "result.json").exists()
        assert (saved_dir_path / "runs.jsonl").exists()
        assert (saved_dir_path / "summary.json").exists()
        
        loaded_result = bench.load_result(saved_dir_path)
        assert loaded_result.dataset_name == original_result.dataset_name
        assert len(loaded_result.runs) == len(original_result.runs)

    def test_bench_result_serialization(self):
        """Test BenchResult to_dict and from_dict methods."""
        sample_run = Run(
            task_id="test_task", task_family="test_family", run_id="test_run", 
            alias="TestAlias", model="test_model", score_binarized=1, human_minutes=10.0,
            human_source="test", task_source="test_dataset"
        )
        result = BenchResult(
            dataset_name="test_dataset", model_name="test_model", model_alias="Test Model",
            runs=[sample_run], summary_stats={"test": 123}, metadata={"param": "value"},
            timestamp=datetime.now(timezone.utc).isoformat(), success=True, error_message=None,
        )
        result_dict = result.to_dict()
        loaded_result = BenchResult.from_dict(result_dict)
        assert loaded_result.dataset_name == result.dataset_name
        assert loaded_result.runs[0].task_id == result.runs[0].task_id

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            class BadBenchMissingDatasetName(Bench):
                def list_available_tasks(self): return []
                def run_evaluation(self, model_name, model_alias=None, task_ids=None, **kwargs): pass
            BadBenchMissingDatasetName(Path("."), Path("."))

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            class BadBenchMissingListTasks(Bench):
                @property
                def dataset_name(self): return "bad"
                def run_evaluation(self, model_name, model_alias=None, task_ids=None, **kwargs): pass
            BadBenchMissingListTasks(Path("."), Path("."))

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            class BadBenchMissingRunEval(Bench):
                @property
                def dataset_name(self): return "bad"
                def list_available_tasks(self): return []
            BadBenchMissingRunEval(Path("."), Path("."))

    def test_path_coercion(self):
        """Test that paths are coerced to Path objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir_str = f"{temp_dir}/data/raw/mock_dataset"
            Path(dataset_dir_str).mkdir(parents=True)
            output_dir_str = f"{temp_dir}/results"
            
            bench = MockBench(dataset_dir_str, output_dir_str)
            assert isinstance(bench.dataset_dir, Path)
            assert isinstance(bench.output_dir, Path) 