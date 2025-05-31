"""
Unit tests for the CybenchPrepare class.
Ensures correct transformation of raw CyBench data into strictly schema-compliant Run objects
and associated Task objects.
"""
import pytest
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

from human_ttc_eval.datasets.cybench.cybench_prepare import CybenchPrepare
from human_ttc_eval.core.run import Run
from human_ttc_eval.core.task import Task
from human_ttc_eval import config

# Store original DATA_DIR and override it for tests
ORIGINAL_DATA_DIR = config.DATA_DIR

@pytest.fixture(autouse=True)
def mock_config_paths_for_cybench_prepare(tmp_path_factory):
    """Fixture to temporarily change config.DATA_DIR for CybenchPrepare tests."""
    temp_data_root = tmp_path_factory.mktemp("test_data_root_cybench_prepare")
    config.DATA_DIR = temp_data_root
    # Ensure 'raw' and 'processed' directories exist under the temp_data_root
    (config.DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    yield
    config.DATA_DIR = ORIGINAL_DATA_DIR

def create_dummy_cybench_raw_file(
    raw_data_dir: Path, 
    filename: str, 
    records: List[Dict[str, Any]]
) -> Path:
    """Helper to create a dummy raw data file for CyBench."""
    dataset_raw_dir = raw_data_dir # In CybenchPrepare, raw_data_dir is already dataset specific
    dataset_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file_path = dataset_raw_dir / filename
    with open(raw_file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    return raw_file_path

@pytest.fixture
def cybench_preparer_fixture():
    """Fixture to get an instance of CybenchPrepare."""
    # CybenchPrepare's __init__ creates its own subdirectories if they don't exist
    # under config.DATA_DIR / "raw" / "cybench" and config.DATA_DIR / "processed" / "cybench"
    return CybenchPrepare()

class TestCybenchPrepare:
    """Test suite for the CybenchPrepare class."""

    # Sample raw records based on CybenchPrepare's expected fields
    RAW_RECORD_1 = {
        "task_id": "task/foo/bar",
        "name": "Challenge Bar",
        "category": "FooCategory",
        "event": "Event2024",
        "ctf_name": "FooCTF",
        "fastest_solve_time_seconds": 120, # 2 minutes
        "readme_data": {"difficulty": "Easy"}
    }
    RAW_RECORD_2 = {
        "task_id": "task/alpha/beta",
        "name": "Challenge Beta",
        "category": "AlphaCategory",
        "event": "Event2023",
        "ctf_name": "AlphaCTF",
        "fastest_solve_time_seconds": 300, # 5 minutes
        "readme_data": {"difficulty": "Medium"}
    }
    RAW_RECORD_MINIMAL = { # Record with only essential fields
        "task_id": "task/minimal/one",
        "fastest_solve_time_seconds": 60
    }

    def test_initialization(self, cybench_preparer_fixture: CybenchPrepare):
        """Test the initialization of CybenchPrepare."""
        preparer = cybench_preparer_fixture
        assert preparer.dataset_name == "cybench"
        expected_raw_dir = config.DATA_DIR / "raw" / "cybench"
        expected_processed_dir = config.DATA_DIR / "processed" / "cybench"
        
        assert preparer.raw_data_dir == expected_raw_dir
        assert preparer.processed_data_dir == expected_processed_dir
        assert preparer.default_raw_input_filename == "cybench_raw_data.jsonl"
        assert preparer.default_human_runs_filename == "cybench_human_runs.jsonl"
        assert preparer.default_tasks_filename == "cybench_tasks.jsonl"

        # Check that directories were created by __init__ if they didn't exist
        assert expected_raw_dir.exists()
        assert expected_processed_dir.exists()

    def test_get_dataset_task_metadata(self, cybench_preparer_fixture: CybenchPrepare):
        """Test that get_dataset_task_metadata returns an empty dict."""
        preparer = cybench_preparer_fixture
        # Create a dummy Run object to pass to the method
        dummy_run = Run(
            task_id="any_task_id", task_family="any_family", run_id="any_run",
            alias="any_alias", model="human", score_binarized=1, human_minutes=10,
            human_source="s", task_source="ds"
        )
        metadata = preparer.get_dataset_task_metadata(dummy_run)
        assert metadata == {}

    def test_prepare_single_valid_record(self, cybench_preparer_fixture: CybenchPrepare):
        """Test prepare with a single valid raw record."""
        preparer = cybench_preparer_fixture
        create_dummy_cybench_raw_file(
            preparer.raw_data_dir,
            preparer.default_raw_input_filename,
            [self.RAW_RECORD_1]
        )
        runs = preparer.prepare()
        assert len(runs) == 1
        run = runs[0]
        
        assert run.task_id == self.RAW_RECORD_1["task_id"]
        assert run.task_family == "foo"
        assert run.run_id == f"human_{self.RAW_RECORD_1['task_id'].replace('/ ', '_')}"
        assert run.alias == "Human Baseline (CyBench)"
        assert run.model == "human"
        assert run.score_binarized == 1
        assert run.score_cont == 1.0
        assert run.human_minutes == 2.0
        assert run.human_source == "cybench"
        assert run.task_source == "cybench"
        assert run.started_at == 0.0
        assert run.completed_at == 120.0

    def test_prepare_multiple_valid_records(self, cybench_preparer_fixture: CybenchPrepare):
        """Test prepare with multiple valid raw records."""
        preparer = cybench_preparer_fixture
        create_dummy_cybench_raw_file(
            preparer.raw_data_dir,
            preparer.default_raw_input_filename,
            [self.RAW_RECORD_1, self.RAW_RECORD_2]
        )
        runs = preparer.prepare()
        assert len(runs) == 2
        assert runs[0].task_id == self.RAW_RECORD_1["task_id"]
        assert runs[1].task_id == self.RAW_RECORD_2["task_id"]
        assert runs[1].human_minutes == 5.0
        assert runs[1].completed_at == 300.0

    def test_prepare_minimal_valid_record(self, cybench_preparer_fixture: CybenchPrepare):
        """Test prepare with a minimal but valid raw record."""
        preparer = cybench_preparer_fixture
        create_dummy_cybench_raw_file(
            preparer.raw_data_dir,
            preparer.default_raw_input_filename,
            [self.RAW_RECORD_MINIMAL]
        )
        runs = preparer.prepare()
        assert len(runs) == 1
        run = runs[0]
        
        assert run.task_id == self.RAW_RECORD_MINIMAL["task_id"]
        # Defaults used for name, category, event; task_family from task_id
        assert run.task_family == "minimal"
        assert run.human_minutes == 1.0
        assert run.completed_at == 60.0

    def test_prepare_raw_file_not_found(self, cybench_preparer_fixture: CybenchPrepare, caplog):
        """Test prepare when the raw data file is not found."""
        preparer = cybench_preparer_fixture
        # Ensure the file does not exist
        raw_file = preparer.raw_data_dir / preparer.default_raw_input_filename
        if raw_file.exists():
            raw_file.unlink()

        with caplog.at_level(logging.ERROR):
            runs = preparer.prepare()
        
        assert f"Raw CyBench data file not found: {raw_file}" in caplog.text
        assert len(runs) == 0

    def test_prepare_skip_record_missing_task_id(self, cybench_preparer_fixture: CybenchPrepare, caplog):
        """Test that records with missing 'task_id' are skipped."""
        preparer = cybench_preparer_fixture
        invalid_record = self.RAW_RECORD_1.copy()
        del invalid_record["task_id"]
        
        create_dummy_cybench_raw_file(
            preparer.raw_data_dir,
            preparer.default_raw_input_filename,
            [invalid_record, self.RAW_RECORD_2] # Mix valid and invalid
        )
        with caplog.at_level(logging.WARNING):
            runs = preparer.prepare()
        
        assert "Skipping record in" in caplog.text and "due to missing 'task_id'" in caplog.text
        assert len(runs) == 1 # Only RAW_RECORD_2 should be processed
        assert runs[0].task_id == self.RAW_RECORD_2["task_id"]

    def test_prepare_skip_record_missing_solve_time(self, cybench_preparer_fixture: CybenchPrepare, caplog):
        """Test that records with missing 'fastest_solve_time_seconds' are skipped."""
        preparer = cybench_preparer_fixture
        invalid_record = self.RAW_RECORD_1.copy()
        del invalid_record["fastest_solve_time_seconds"]
        
        create_dummy_cybench_raw_file(
            preparer.raw_data_dir,
            preparer.default_raw_input_filename,
            [invalid_record, self.RAW_RECORD_2]
        )
        with caplog.at_level(logging.WARNING):
            runs = preparer.prepare()
        
        expected_log_msg = f"Skipping task '{self.RAW_RECORD_1['task_id']}' due to missing 'fastest_solve_time_seconds'."
        assert expected_log_msg in caplog.text
        assert len(runs) == 1
        assert runs[0].task_id == self.RAW_RECORD_2["task_id"]

    def test_prepare_skip_record_invalid_solve_time_format(self, cybench_preparer_fixture: CybenchPrepare, caplog):
        """Test that records with invalid 'fastest_solve_time_seconds' format are skipped."""
        preparer = cybench_preparer_fixture
        invalid_record = self.RAW_RECORD_1.copy()
        invalid_record["fastest_solve_time_seconds"] = "not_a_number"
        
        create_dummy_cybench_raw_file(
            preparer.raw_data_dir,
            preparer.default_raw_input_filename,
            [invalid_record, self.RAW_RECORD_2]
        )
        with caplog.at_level(logging.WARNING):
            runs = preparer.prepare()
        
        expected_log_msg = f"Skipping task '{self.RAW_RECORD_1['task_id']}' due to invalid format for 'fastest_solve_time_seconds': not_a_number"
        assert expected_log_msg in caplog.text
        assert len(runs) == 1
        assert runs[0].task_id == self.RAW_RECORD_2["task_id"]

    def test_prepare_malformed_json_line_recovery(self, cybench_preparer_fixture: CybenchPrepare, caplog):
        """Test recovery from malformed JSON lines in the raw data file."""
        preparer = cybench_preparer_fixture
        raw_file_path = preparer.raw_data_dir / preparer.default_raw_input_filename
        
        # Manually create file with malformed line
        preparer.raw_data_dir.mkdir(parents=True, exist_ok=True)
        with open(raw_file_path, 'w') as f:
            f.write(json.dumps(self.RAW_RECORD_1) + '\n')
            f.write("this is not valid json\n")
            f.write(json.dumps(self.RAW_RECORD_2) + '\n')
            f.write(json.dumps(self.RAW_RECORD_MINIMAL) + '\n')
        
        with caplog.at_level(logging.WARNING):
            runs = preparer.prepare()
        
        assert "Skipping malformed JSON line 2" in caplog.text
        assert len(runs) == 3 # RAW_RECORD_1, RAW_RECORD_2, and RAW_RECORD_MINIMAL
        assert runs[0].task_id == self.RAW_RECORD_1["task_id"]
        assert runs[1].task_id == self.RAW_RECORD_2["task_id"]
        assert runs[2].task_id == self.RAW_RECORD_MINIMAL["task_id"]

    def test_run_method_integration_cybench(self, cybench_preparer_fixture: CybenchPrepare):
        """Test the full run() method integration for CybenchPrepare."""
        preparer = cybench_preparer_fixture
        create_dummy_cybench_raw_file(
            preparer.raw_data_dir,
            preparer.default_raw_input_filename,
            [self.RAW_RECORD_1, self.RAW_RECORD_2, self.RAW_RECORD_MINIMAL]
        )
        
        # These records represent 3 unique tasks
        # RAW_RECORD_1: task/foo/bar
        # RAW_RECORD_2: task/alpha/beta
        # RAW_RECORD_MINIMAL: task/minimal/one
        
        prepared_runs = preparer.run() # Calls prepare(), save_runs(), save_tasks()
        
        assert len(prepared_runs) == 3 # All 3 records are valid and should be prepared
        
        # Verify human_runs.jsonl
        human_runs_file = preparer.processed_data_dir / preparer.default_human_runs_filename
        assert human_runs_file.exists()
        loaded_human_runs = Run.load_jsonl(human_runs_file)
        assert len(loaded_human_runs) == 3
        
        expected_equal_weight = 1/3
        for run in loaded_human_runs:
            assert run.model == "human"
            assert run.equal_task_weight == pytest.approx(expected_equal_weight)
            assert run.invsqrt_task_weight is not None # calculation depends on all human_minutes

        # Verify tasks.jsonl
        tasks_file = preparer.processed_data_dir / preparer.default_tasks_filename
        assert tasks_file.exists()
        loaded_tasks = Task.load_jsonl(tasks_file)
        assert len(loaded_tasks) == 3 # 3 unique tasks

        # Check content of one task for correctness
        task1_loaded = next((t for t in loaded_tasks if t.task_id == self.RAW_RECORD_1["task_id"]), None)
        assert task1_loaded is not None
        assert task1_loaded.task_family == "foo"
        assert task1_loaded.human_minutes == 2.0
        assert task1_loaded.equal_task_weight == pytest.approx(expected_equal_weight)
        
        # Find the corresponding run to check invsqrt_task_weight consistency
        run1_for_task1 = next(r for r in loaded_human_runs if r.task_id == self.RAW_RECORD_1["task_id"])
        assert task1_loaded.invsqrt_task_weight == run1_for_task1.invsqrt_task_weight
        assert task1_loaded.dataset_task_metadata == {} # CyBench returns empty

        task_minimal_loaded = next((t for t in loaded_tasks if t.task_id == self.RAW_RECORD_MINIMAL["task_id"]), None)
        assert task_minimal_loaded is not None
        assert task_minimal_loaded.task_family == "minimal"
        assert task_minimal_loaded.human_minutes == 1.0
        assert task_minimal_loaded.equal_task_weight == pytest.approx(expected_equal_weight)
        assert task_minimal_loaded.dataset_task_metadata == {}
