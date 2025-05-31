"""
Unit tests for the CybenchBench class.
"""
import pytest
import tempfile
from pathlib import Path
import json
import subprocess
from unittest.mock import patch, MagicMock

from human_ttc_eval.datasets.cybench.cybench_bench import CybenchBench
from human_ttc_eval.core import Run
from human_ttc_eval import config

# Store original CYBENCH_REPO_PATH and restore it after tests
ORIGINAL_CYBENCH_REPO_PATH = config.CYBENCH_REPO_PATH

@pytest.fixture
def mock_cybench_repo(tmp_path_factory):
    """Creates a temporary mock CyBench repository structure."""
    repo_dir = tmp_path_factory.mktemp("mock_cybench_repo")
    (repo_dir / "run_benchmark.py").touch()
    (repo_dir / "task_list.txt").write_text("benchmark/tasks/task1\nbenchmark/tasks/task2\n")
    (repo_dir / "subtask_list.txt").write_text("benchmark/tasks/task3\n") # A subtask
    (repo_dir / "analytics").mkdir(exist_ok=True) # For completeness, though not directly used by bench
    logs_dir = repo_dir / "logs"
    logs_dir.mkdir()
    
    # Monkeypatch the config to point to this mock repo
    config.CYBENCH_REPO_PATH = str(repo_dir)
    yield repo_dir
    # Restore original config path
    config.CYBENCH_REPO_PATH = ORIGINAL_CYBENCH_REPO_PATH

@pytest.fixture
def temp_output_dirs(tmp_path_factory):
    """Provides temporary dataset_root (like 'data/') and benchmark output_dir."""
    with tempfile.TemporaryDirectory() as input_dir_str: # This will be the root for dataset related paths
        with tempfile.TemporaryDirectory() as output_dir_str:
            dataset_root = Path(input_dir_str) # This is like the top-level 'data' directory
            
            # Ensure the structure Bench expects for processed data exists under this root
            (dataset_root / "processed" / "cybench").mkdir(parents=True, exist_ok=True)
            
            bench_output_dir_root = Path(output_dir_str) # This is like the top-level 'results' directory
            final_bench_output_dir = bench_output_dir_root / "benchmarks" / "cybench"
            final_bench_output_dir.mkdir(parents=True, exist_ok=True)
            
            yield dataset_root, final_bench_output_dir

@pytest.fixture
def cybench_bench_instance(mock_cybench_repo, temp_output_dirs, sample_human_baseline_runs_file):
    """Provides a CybenchBench instance with mocked repo, temp output dirs, and pre-existing baseline file."""
    dataset_root, bench_output_dir = temp_output_dirs # dataset_root is like 'data/'
    
    # dataset_dir for Bench class should point to where raw data would be: data/raw/cybench
    mock_raw_dir = dataset_root / "raw" / "cybench"
    mock_raw_dir.mkdir(parents=True, exist_ok=True)

    # sample_human_baseline_runs_file fixture ensures the baseline file is created 
    # relative to dataset_root / "processed" / "cybench" / ...
    return CybenchBench(dataset_dir=mock_raw_dir, output_dir=bench_output_dir, cybench_repo_path_override=mock_cybench_repo)

@pytest.fixture
def sample_human_baseline_runs_file(temp_output_dirs):
    """Creates a sample human baseline file and returns its path."""
    dataset_root, _ = temp_output_dirs # dataset_root is like 'data/'
    # Baseline file should be in data/processed/cybench/
    baseline_file_path = dataset_root / "processed" / "cybench" / "cybench_human_runs.jsonl"
    
    runs_data = [
        Run(task_id="benchmark/tasks/task1", task_family="cybench", run_id="h_t1", alias="H_T1", model="human", score_binarized=1, human_minutes=10.0, human_source="test", task_source="cybench", started_at=0.0, completed_at=600.0).to_jsonl_dict(),
        Run(task_id="benchmark/tasks/task2", task_family="cybench", run_id="h_t2", alias="H_T2", model="human", score_binarized=1, human_minutes=20.0, human_source="test", task_source="cybench", started_at=0.0, completed_at=1200.0).to_jsonl_dict(),
        Run(task_id="benchmark/tasks/task3", task_family="cybench", run_id="h_t3", alias="H_T3", model="human", score_binarized=0, human_minutes=5.0, human_source="test", task_source="cybench", started_at=0.0, completed_at=300.0).to_jsonl_dict(),
    ]
    with open(baseline_file_path, 'w') as f:
        for r_data in runs_data:
            f.write(json.dumps(r_data) + '\n')
    return baseline_file_path


class TestCybenchBench:
    def test_initialization(self, cybench_bench_instance, mock_cybench_repo, temp_output_dirs):
        """Test successful initialization."""
        _, bench_output_dir = temp_output_dirs
        assert cybench_bench_instance.cybench_repo_path == mock_cybench_repo
        assert cybench_bench_instance.output_dir == bench_output_dir
        assert cybench_bench_instance.dataset_name == "cybench"

    def test_initialization_repo_not_found(self, tmp_path):
        """Test initialization fails if CyBench repo doesn't exist."""
        non_existent_repo_path = tmp_path / "non_existent_repo"
        # config.CYBENCH_REPO_PATH = str(tmp_path / "non_existent_repo") # No longer needed
        with pytest.raises(FileNotFoundError, match="CyBench repository not found"):
            CybenchBench(dataset_dir=tmp_path, output_dir=tmp_path, cybench_repo_path_override=non_existent_repo_path)
        # config.CYBENCH_REPO_PATH = ORIGINAL_CYBENCH_REPO_PATH # No longer needed if not changed

    def test_initialization_script_not_found(self, tmp_path):
        """Test init fails if run_benchmark.py is missing."""
        mock_repo_no_script = tmp_path / "mock_cybench_repo_no_script"
        mock_repo_no_script.mkdir()
        # (run_benchmark.py is not created here)
        # config.CYBENCH_REPO_PATH = str(mock_repo) # No longer needed
        with pytest.raises(FileNotFoundError, match="run_benchmark.py not found"):
            CybenchBench(dataset_dir=tmp_path, output_dir=tmp_path, cybench_repo_path_override=mock_repo_no_script)
        # config.CYBENCH_REPO_PATH = ORIGINAL_CYBENCH_REPO_PATH # No longer needed

    def test_list_available_tasks(self, cybench_bench_instance):
        """Test listing available tasks from mock repo files."""
        tasks = cybench_bench_instance.list_available_tasks()
        assert len(tasks) == 3
        assert "benchmark/tasks/task1" in tasks
        assert "benchmark/tasks/task2" in tasks
        assert "benchmark/tasks/task3" in tasks # From subtask_list.txt

    def test_validate_model_name(self, cybench_bench_instance):
        """Test model name validation."""
        assert cybench_bench_instance.validate_model_name("openai/gpt-4o-2024-05-13") is True
        assert cybench_bench_instance.validate_model_name("non_existent/model") is False

    @patch("subprocess.run")
    def test_run_evaluation_success(self, mock_subprocess_run, cybench_bench_instance, mock_cybench_repo, sample_human_baseline_runs_file):
        """Test a successful run_evaluation call, mocking subprocess and log parsing."""
        # Ensure human baseline is loaded by re-initializing after file creation
        cybench_bench_instance._load_human_baseline() # Manually trigger reload for this test setup

        model_name = "openai/gpt-4o-2024-05-13"
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        
        # Create mock CyBench log files that _parse_and_create_runs would find
        logs_dir = mock_cybench_repo / "logs"
        task1_log_dir = logs_dir / "benchmark_tasks_task1" / "openai" / model_name.split("/")[-1] / "ts1"
        task1_log_dir.mkdir(parents=True)
        task1_log_content = {
            "task": {"path": "benchmark/tasks/task1", "name": "Task One"},
            "num_correct_subtasks": 1, "num_subtasks": 1, "time_taken_in_ms": 10000
        }
        with open(task1_log_dir / f"{model_name.split('/')[-1]}_details.json", 'w') as f:
            json.dump(task1_log_content, f)

        task2_log_dir = logs_dir / "benchmark_tasks_task2" / "openai" / model_name.split("/")[-1] / "ts2"
        task2_log_dir.mkdir(parents=True)
        task2_log_content = {
            "task": {"path": "benchmark/tasks/task2", "name": "Task Two"},
            "num_correct_subtasks": 0, "num_subtasks": 2, "time_taken_in_ms": 5000
        }
        with open(task2_log_dir / f"{model_name.split('/')[-1]}_details.json", 'w') as f:
            json.dump(task2_log_content, f)
        
        # Run evaluation for specific tasks to make log finding more deterministic in test
        tasks_to_run = ["benchmark/tasks/task1", "benchmark/tasks/task2"]
        result = cybench_bench_instance.run_evaluation(model_name=model_name, task_ids=tasks_to_run)

        assert result.success is True
        assert len(result.runs) == 2
        
        run1 = next(r for r in result.runs if r.task_id == "benchmark/tasks/task1")
        assert run1.model == model_name
        assert run1.score_binarized == 1
        assert run1.score_cont == 1.0
        assert run1.human_minutes == 10.0 # From sample_human_baseline_runs_file
        assert run1.completed_at == 10.0 # 10000ms

        run2 = next(r for r in result.runs if r.task_id == "benchmark/tasks/task2")
        assert run2.score_binarized == 0
        assert run2.score_cont == 0.0
        assert run2.human_minutes == 20.0
        assert run2.completed_at == 5.0 # 5000ms

        mock_subprocess_run.assert_called_once()
        cmd_args = mock_subprocess_run.call_args[0][0]
        assert model_name in cmd_args
        assert "--task_list" in cmd_args
        # Check if the temp task list file was created and then removed (harder to check removal without more mocks)

    @patch("subprocess.run")
    def test_run_evaluation_script_failure(self, mock_subprocess_run, cybench_bench_instance, sample_human_baseline_runs_file):
        """Test run_evaluation when the CyBench script fails."""
        cybench_bench_instance._load_human_baseline()
        model_name = "openai/gpt-4o-2024-05-13"
        mock_subprocess_run.return_value = MagicMock(returncode=1, stdout="Error output", stderr="Script crashed")
        
        result = cybench_bench_instance.run_evaluation(model_name=model_name, task_ids=["benchmark/tasks/task1"])
        
        assert result.success is False
        assert "Script crashed" in result.error_message
        assert len(result.runs) == 1 # Should create a failed run object for the task
        assert result.runs[0].task_id == "benchmark/tasks/task1"
        assert result.runs[0].score_binarized == 0
        assert result.runs[0].fatal_error_from is not None

    @patch("subprocess.run")
    def test_run_evaluation_timeout(self, mock_subprocess_run, cybench_bench_instance, sample_human_baseline_runs_file):
        """Test run_evaluation when the CyBench script times out."""
        cybench_bench_instance._load_human_baseline()
        model_name = "openai/gpt-4o-2024-05-13"
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="test_cmd", timeout=10)
        
        result = cybench_bench_instance.run_evaluation(model_name=model_name, task_ids=["benchmark/tasks/task1"])
        
        assert result.success is False
        assert "timed out" in result.error_message.lower()
        assert len(result.runs) == 1 # Should create a failed run for the task
        assert result.runs[0].task_id == "benchmark/tasks/task1"
        assert result.runs[0].fatal_error_from == "Global evaluation timeout"

    def test_get_human_minutes_for_task(self, cybench_bench_instance):
        """Test fetching human minutes for a task."""
        cybench_bench_instance._load_human_baseline()
        assert cybench_bench_instance._get_human_minutes_for_task("benchmark/tasks/task1") == 10.0
        assert cybench_bench_instance._get_human_minutes_for_task("benchmark/tasks/task2") == 20.0
        assert cybench_bench_instance._get_human_minutes_for_task("non_existent_task") == 0.0 # Default

    def test_get_task_family(self, cybench_bench_instance):
        """Test task family determination."""
        assert cybench_bench_instance._get_task_family("benchmark/org/event/cat/task") == "cybench"

    def test_parse_and_create_runs_no_logs_dir(self, cybench_bench_instance, tmp_path):
        """Test parsing when the main logs directory doesn't exist."""
        runs, _ = cybench_bench_instance._parse_and_create_runs(tmp_path / "non_existent_logs", "model", "alias", [])
        assert len(runs) == 0

    def test_parse_and_create_runs_no_matching_logs(self, cybench_bench_instance, mock_cybench_repo):
        """Test parsing when no logs match the model or tasks."""
        # Logs dir exists but is empty or has no relevant logs
        runs, _ = cybench_bench_instance._parse_and_create_runs(mock_cybench_repo / "logs", "test_model", "alias", ["benchmark/tasks/task1"])
        assert len(runs) == 0

    # More detailed tests for _parse_and_create_runs can be added to cover edge cases in log content
    # For example: missing fields in log JSON, different subtask counts, etc. 