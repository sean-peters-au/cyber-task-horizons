"""
Unit tests for the Prepare base class.

Tests the new data preparation interface that works with Run objects.
"""

import pytest
from pathlib import Path
import json
import tempfile
from typing import List, Dict, Any, Optional
import logging
from unittest.mock import Mock

from human_ttc_eval.core.prepare import Prepare
from human_ttc_eval.core.run import Run
from human_ttc_eval.core.task import Task
from human_ttc_eval import config

# Store original DATA_DIR and override it for tests
ORIGINAL_DATA_DIR = config.DATA_DIR

@pytest.fixture(autouse=True)
def mock_config_paths_for_prepare(tmp_path_factory):
    """Fixture to temporarily change config.DATA_DIR for Prepare tests."""
    temp_data_root = tmp_path_factory.mktemp("test_data_root_prepare")
    config.DATA_DIR = temp_data_root
    yield
    config.DATA_DIR = ORIGINAL_DATA_DIR

# Mock data
def create_mock_runs(count=3, dataset_name="mock_dataset", include_non_human=False) -> List[Run]:
    runs = []
    for i in range(count):
        model_name = "human" if not include_non_human or i % 2 == 0 else f"ai_model_{i}"
        runs.append(Run(
            task_id=f"task_id_{i}", 
            task_family=f"fam_{(i % 2)}", 
            run_id=f"run_{i}", 
            alias=f"{dataset_name}_task_{i}", 
            model=model_name,
            score_binarized=i % 2,
            human_minutes=10.0 + i * 5,
            human_source=f"{dataset_name}_source",
            task_source=dataset_name
        ))
    return runs

class MockPrepare(Prepare):
    """Mock implementation of Prepare for testing."""
    MOCK_DATASET_NAME = "mock_prepare_dataset"
    MOCK_METADATA = {"custom_field": "custom_value"}

    def __init__(self, dataset_name: str = MOCK_DATASET_NAME, num_mock_runs: int = 3):
        super().__init__(dataset_name)
        self.num_mock_runs = num_mock_runs
        dummy_raw_file = self.raw_data_dir / self.default_raw_input_filename
        dummy_raw_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_raw_file.touch(exist_ok=True)

    def prepare(self) -> List[Run]:
        # Return human runs for this mock, as prepare() is expected to give human data
        return create_mock_runs(self.num_mock_runs, self.dataset_name, include_non_human=False)

    def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]:
        if representative_run.task_id == "task_id_1": # Example: only task_id_1 gets special metadata
            return self.MOCK_METADATA.copy() # Return a copy to prevent modification of class attr
        return {}

@pytest.fixture
def temp_data_dirs():
    with tempfile.TemporaryDirectory() as base_dir_str:
        base_dir = Path(base_dir_str)
        original_data_dir = config.DATA_DIR
        config.DATA_DIR = base_dir
        
        (config.DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
        (config.DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
        
        yield base_dir
        
        config.DATA_DIR = original_data_dir

class TestPrepare:
    """Tests for the Prepare base class."""

    def test_initialization_success(self, temp_data_dirs):
        ds_name = "init_success_ds"
        # MockPrepare will create its own dataset-specific subdirectories under temp_data_dirs/raw and temp_data_dirs/processed
        preparer = MockPrepare(dataset_name=ds_name)
        assert preparer.dataset_name == ds_name
        assert preparer.raw_data_dir == config.DATA_DIR / "raw" / ds_name
        assert preparer.processed_data_dir == config.DATA_DIR / "processed" / ds_name
        assert preparer.default_human_runs_filename == f"{ds_name}_human_runs.jsonl"
        assert preparer.default_tasks_filename == f"{ds_name}_tasks.jsonl"
        assert preparer.processed_data_dir.exists()
        # Check that MockPrepare also created its dummy raw file's directory
        assert (config.DATA_DIR / "raw" / ds_name).exists()

    def test_initialization_raw_dir_does_not_exist_warning(self, caplog):
        # This test needs to ensure the specific <DATA_DIR>/raw/<dataset_name> does NOT exist
        # before super().__init__ is called, but MockPrepare then creates it.
        with tempfile.TemporaryDirectory() as base_dir_str:
            original_data_dir = config.DATA_DIR
            config.DATA_DIR = Path(base_dir_str)
            
            ds_name_warn = "warn_ds"

            with caplog.at_level(logging.WARNING):
                _ = MockPrepare(dataset_name=ds_name_warn)
            
            expected_raw_dir = config.DATA_DIR / "raw" / ds_name_warn
            assert f"Raw data directory {str(expected_raw_dir)} does not exist" in caplog.text
            
            config.DATA_DIR = original_data_dir # Restore

    def test_save_runs_and_tasks_flow(self, temp_data_dirs):
        ds_name = "save_flow_ds"
        preparer = MockPrepare(dataset_name=ds_name, num_mock_runs=3)
        
        prepared_runs = preparer.prepare()
        # Correct aliases based on create_mock_runs
        for i, run in enumerate(prepared_runs):
            run.alias = f"{ds_name}_task_{i}"

        assert len(prepared_runs) == 3
        assert all(run.model == "human" for run in prepared_runs)

        human_runs_path = preparer.save_runs(prepared_runs)
        assert human_runs_path.exists()
        assert human_runs_path.name == preparer.default_human_runs_filename
        
        loaded_runs = Run.load_jsonl(human_runs_path)
        assert len(loaded_runs) == 3
        # MockPrepare with num_mock_runs=3 calls create_mock_runs(3, ...)
        # create_mock_runs(3) creates task_id_0, task_id_1, task_id_2. These are 3 unique tasks.
        # So, equal_task_weight should be 1/3.
        for run in loaded_runs:
            assert run.equal_task_weight == pytest.approx(1/3) 
            assert run.invsqrt_task_weight is not None 
        for run in prepared_runs: 
            assert run.equal_task_weight == pytest.approx(1/3)

        tasks_path = preparer.save_tasks(prepared_runs)
        assert tasks_path.exists()
        assert tasks_path.name == preparer.default_tasks_filename
        
        loaded_tasks = Task.load_jsonl(tasks_path)
        assert len(loaded_tasks) == 3 # task_id_0, task_id_1, task_id_2 are unique

        task0 = next(t for t in loaded_tasks if t.task_id == "task_id_0")
        task1 = next(t for t in loaded_tasks if t.task_id == "task_id_1")
        task2 = next(t for t in loaded_tasks if t.task_id == "task_id_2")

        assert task0.equal_task_weight == pytest.approx(1/3)
        assert task0.invsqrt_task_weight == prepared_runs[0].invsqrt_task_weight 
        assert task0.dataset_task_metadata == {} 

        assert task1.equal_task_weight == pytest.approx(1/3)
        assert task1.invsqrt_task_weight == prepared_runs[1].invsqrt_task_weight
        assert task1.dataset_task_metadata == MockPrepare.MOCK_METADATA 

        assert task2.equal_task_weight == pytest.approx(1/3)
        assert task2.invsqrt_task_weight == prepared_runs[2].invsqrt_task_weight
        assert task2.dataset_task_metadata == {} 

    def test_save_runs_empty_list(self, temp_data_dirs, caplog):
        ds_name = "empty_runs_save"
        preparer = MockPrepare(dataset_name=ds_name, num_mock_runs=0)
        with caplog.at_level(logging.WARNING):
            human_runs_path = preparer.save_runs([])
        assert f"No runs provided to save for dataset '{ds_name}'" in caplog.text
        assert f"No valid human runs to save for '{ds_name}'" in caplog.text 
        assert human_runs_path.exists()
        assert human_runs_path.read_text() == ""

    def test_save_tasks_empty_derived_list(self, temp_data_dirs, caplog):
        ds_name = "empty_tasks_save"
        preparer = MockPrepare(dataset_name=ds_name, num_mock_runs=0)
        with caplog.at_level(logging.WARNING):
            tasks_path = preparer.save_tasks([]) 
        assert f"No runs provided to derive tasks for dataset '{ds_name}'" in caplog.text
        assert f"No unique tasks derived from runs for '{ds_name}'" in caplog.text
        assert tasks_path.exists()
        assert tasks_path.read_text() == ""

    def test_save_runs_validation_failure(self, temp_data_dirs, caplog):
        ds_name = "invalid_run_save"
        preparer = MockPrepare(dataset_name=ds_name)
        invalid_human_run = Run(task_id=None, task_family="fam", run_id="r_invalid", model="human", score_binarized=0, human_minutes=1, human_source="s", task_source="ds", alias="invalid_alias") # Added alias
        with caplog.at_level(logging.ERROR):
            human_runs_path = preparer.save_runs([invalid_human_run])
        assert "failed validation and will be skipped" in caplog.text
        assert human_runs_path.exists()
        assert human_runs_path.read_text() == "" 

    def test_save_runs_skips_non_human_runs(self, temp_data_dirs, caplog):
        ds_name = "skip_non_human"
        preparer = MockPrepare(dataset_name=ds_name)
        runs_with_ai = [
            Run(task_id="t1", task_family="fA", run_id="rh1", model="human", score_binarized=1, human_minutes=10, human_source="s", task_source="ds", alias="human_alias_1"), # Added alias
            Run(task_id="t2", task_family="fB", run_id="rai1", model="gpt-4", score_binarized=0, human_minutes=0, human_source="s", task_source="ds", alias="ai_alias_1") # Added alias
        ]
        with caplog.at_level(logging.WARNING):
            human_runs_path = preparer.save_runs(runs_with_ai)
        assert "not a human run" in caplog.text
        assert human_runs_path.exists()
        loaded_human_runs = Run.load_jsonl(human_runs_path)
        assert len(loaded_human_runs) == 1
        assert loaded_human_runs[0].model == "human"
        assert loaded_human_runs[0].task_id == "t1"

    def test_run_method_orchestration(self, temp_data_dirs, monkeypatch):
        ds_name = "orchestration_ds"
        preparer = MockPrepare(dataset_name=ds_name, num_mock_runs=2)
        
        expected_prepared_runs = create_mock_runs(2, ds_name, include_non_human=False) 
        monkeypatch.setattr(preparer, "prepare", lambda: expected_prepared_runs)
        
        # Use unittest.mock.Mock with wraps for spying behavior
        mock_save_runs = Mock(wraps=preparer.save_runs)
        mock_save_tasks = Mock(wraps=preparer.save_tasks)
        monkeypatch.setattr(preparer, "save_runs", mock_save_runs)
        monkeypatch.setattr(preparer, "save_tasks", mock_save_tasks)

        returned_runs_from_run_method = preparer.run()

        mock_save_runs.assert_called_once_with(expected_prepared_runs)
        mock_save_tasks.assert_called_once_with(expected_prepared_runs)
        assert returned_runs_from_run_method == expected_prepared_runs
        
        assert (preparer.processed_data_dir / preparer.default_human_runs_filename).exists()
        assert (preparer.processed_data_dir / preparer.default_tasks_filename).exists()

    def test_run_method_prepare_returns_none_or_empty(self, temp_data_dirs, monkeypatch, caplog):
        ds_name = "empty_prepare_flow_ds"
        preparer = MockPrepare(dataset_name=ds_name)

        monkeypatch.setattr(preparer, "prepare", lambda: [])
        mock_save_runs_empty = Mock(wraps=preparer.save_runs) 
        mock_save_tasks_empty = Mock(wraps=preparer.save_tasks)
        monkeypatch.setattr(preparer, "save_runs", mock_save_runs_empty)
        monkeypatch.setattr(preparer, "save_tasks", mock_save_tasks_empty)

        with caplog.at_level(logging.WARNING):
            result_empty = preparer.run()
        assert "Preparation yielded no runs" in caplog.text
        mock_save_runs_empty.assert_called_once_with([])
        mock_save_tasks_empty.assert_called_once_with([])
        assert result_empty == []
        assert (preparer.processed_data_dir / preparer.default_human_runs_filename).exists() 
        assert (preparer.processed_data_dir / preparer.default_tasks_filename).exists() 
        caplog.clear()

        monkeypatch.setattr(preparer, "prepare", lambda: None)
        mock_save_runs_none = Mock(wraps=preparer.save_runs) 
        mock_save_tasks_none = Mock(wraps=preparer.save_tasks)
        monkeypatch.setattr(preparer, "save_runs", mock_save_runs_none)
        monkeypatch.setattr(preparer, "save_tasks", mock_save_tasks_none)
        with caplog.at_level(logging.WARNING):
            result_none = preparer.run()
        assert f"Prepare method for '{ds_name}' returned None" in caplog.text
        mock_save_runs_none.assert_called_once_with([])
        mock_save_tasks_none.assert_called_once_with([])
        assert result_none == []
        assert (preparer.processed_data_dir / preparer.default_human_runs_filename).exists()
        assert (preparer.processed_data_dir / preparer.default_tasks_filename).exists()

    def test_abstract_methods_enforcement(self):
        # More flexible regex for method order in error message
        match_str = r"Can't instantiate abstract class MinimalPrepare .* abstract methods? (get_dataset_task_metadata, prepare|prepare, get_dataset_task_metadata)"
        with pytest.raises(TypeError, match=match_str):
            class MinimalPrepare(Prepare): 
                def __init__(self, dataset_name: str):
                    super().__init__(dataset_name)
            MinimalPrepare(dataset_name="abstract_test_1")

        with pytest.raises(TypeError, match=r"Can't instantiate abstract class PrepareMissingOne .* abstract method get_dataset_task_metadata"):
            class PrepareMissingOne(Prepare):
                def __init__(self, dataset_name: str):
                    super().__init__(dataset_name)
                def prepare(self) -> List[Run]: return []
            PrepareMissingOne(dataset_name="abstract_test_2")

        with pytest.raises(TypeError, match=r"Can't instantiate abstract class PrepareMissingTwo .* abstract method prepare"):
            class PrepareMissingTwo(Prepare):
                def __init__(self, dataset_name: str):
                    super().__init__(dataset_name)
                def get_dataset_task_metadata(self, representative_run: Run) -> Dict[str, Any]: return {}
            PrepareMissingTwo(dataset_name="abstract_test_3") 