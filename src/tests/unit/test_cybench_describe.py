"""
Unit tests for the CybenchDescribe class.
"""
import pytest
import tempfile
from pathlib import Path
import pandas as pd # For reading the output CSV
from typing import List, Dict
import logging

from human_ttc_eval.datasets.cybench.cybench_describe import CybenchDescribe
from human_ttc_eval.core.run import Run
from human_ttc_eval import config

def create_runs_jsonl_file(directory: Path, filename: str, runs_data: List[Dict]) -> Path:
    file_path = directory / filename
    runs_to_save = [Run.from_dict(data) for data in runs_data]
    Run.save_jsonl(runs_to_save, str(file_path))
    return file_path

@pytest.fixture
def temp_dirs(): # Simplified fixture just for output_dir, input files created per test
    with tempfile.TemporaryDirectory() as input_temp_str:
        with tempfile.TemporaryDirectory() as output_temp_str:
            yield Path(input_temp_str), Path(output_temp_str)

@pytest.fixture
def cybench_describer_fixture(temp_dirs):
    """Provides a CybenchDescribe instance with temporary directories."""
    input_base_dir, output_dir = temp_dirs
    dummy_input_files_dir = input_base_dir / "cybench_desc_fixture_input"
    dummy_input_files_dir.mkdir(parents=True, exist_ok=True)
    dummy_input_file = dummy_input_files_dir / "dummy_input.jsonl"
    with open(dummy_input_file, 'w') as f: # Create an empty but valid JSONL
        pass 
    return CybenchDescribe(input_files=[dummy_input_file], output_dir=output_dir)

class TestCybenchDescribe:

    def test_initialization(self, cybench_describer_fixture):
        describer = cybench_describer_fixture
        assert describer.dataset_name == "cybench"
        assert isinstance(describer.input_files, list)
        assert len(describer.input_files) == 1
        assert describer.output_dir.exists()

    def test_dataset_name_property(self, cybench_describer_fixture):
        describer = cybench_describer_fixture
        assert describer.dataset_name == "cybench"

    def test_extract_event_from_task_id(self, cybench_describer_fixture):
        describer = cybench_describer_fixture
        assert describer._extract_event_from_task_id("benchmark/org/EventA/cat/task") == "EventA"
        assert describer._extract_event_from_task_id("other/format") == "UnknownEvent"

    def test_generate_custom_analysis_empty_df(self, cybench_describer_fixture, caplog):
        describer = cybench_describer_fixture
        describer.df = pd.DataFrame() # Ensure df is empty
        with caplog.at_level(logging.WARNING):
            describer.generate_custom_analysis()
        assert "No data loaded" in caplog.text or "DataFrame is empty" in caplog.text

    def test_generate_custom_analysis_success(self, temp_dirs):
        input_base_dir, output_dir = temp_dirs
        dataset_name = "cybench"
        processed_dir = input_base_dir / dataset_name # Simulating data/processed/cybench
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        runs_data = [
            {"task_id": "benchmark/org1/EventA/cat1/task1", "human_minutes": 10, "score_binarized": 1, "model":"human", "task_family":"f1", "run_id":"r1", "alias":"a1", "human_source":"s", "task_source":"s"},
            {"task_id": "benchmark/org1/EventA/cat2/task2", "human_minutes": 20, "score_binarized": 0, "model":"human", "task_family":"f1", "run_id":"r2", "alias":"a2", "human_source":"s", "task_source":"s"},
            {"task_id": "benchmark/org2/EventB/cat1/task3", "human_minutes": 30, "score_binarized": 1, "model":"human", "task_family":"f2", "run_id":"r3", "alias":"a3", "human_source":"s", "task_source":"s"}
        ]
        input_file = create_runs_jsonl_file(processed_dir, f"{dataset_name}_prepared.jsonl", runs_data)
        
        describer = CybenchDescribe(input_files=[input_file], output_dir=output_dir)
        describer.load_runs() # Load data into describer.df
        describer.generate_custom_analysis()
        
        expected_csv = output_dir / f"{dataset_name}_event_summary.csv"
        assert expected_csv.exists()
        df_event = pd.read_csv(expected_csv)
        assert "EventA" in df_event["event"].values
        assert "EventB" in df_event["event"].values
        assert df_event[df_event["event"] == "EventA"]["num_tasks"].iloc[0] == 2
        assert df_event[df_event["event"] == "EventA"]["successful_tasks"].iloc[0] == 1

    def test_custom_analysis_no_event_column(self, temp_dirs, caplog):
        input_base_dir, output_dir = temp_dirs
        dataset_name = "cybench_no_event"
        processed_dir = input_base_dir / dataset_name
        processed_dir.mkdir(parents=True, exist_ok=True)

        runs_data = [
            {"task_id": "badly_formatted_task_id_1", "human_minutes": 10, "score_binarized": 1, "model":"human", "task_family":"f1", "run_id":"r1", "alias":"a1", "human_source":"s", "task_source":"s"},
            {"task_id": "no_event_info_here", "human_minutes": 20, "score_binarized": 0, "model":"human", "task_family":"f1", "run_id":"r2", "alias":"a2", "human_source":"s", "task_source":"s"},
        ]
        input_file = create_runs_jsonl_file(processed_dir, f"{dataset_name}_prepared.jsonl", runs_data)
        
        describer = CybenchDescribe(input_files=[input_file], output_dir=output_dir)
        describer.load_runs()
        with caplog.at_level(logging.WARNING):
            describer.generate_custom_analysis()
        assert "All task_ids resulted in 'UnknownEvent'" in caplog.text

    def test_run_method_calls_custom_analysis(self, temp_dirs, monkeypatch):
        input_base_dir, output_dir = temp_dirs
        dataset_name = "cybench_run_custom"
        processed_dir = input_base_dir / dataset_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        runs_data = [{"task_id": "benchmark/org/EventX/cat/task", "human_minutes": 10, "score_binarized": 1, "model":"human", "task_family":"f1", "run_id":"r1", "alias":"a1", "human_source":"s", "task_source":"s"}]
        input_file = create_runs_jsonl_file(processed_dir, f"{dataset_name}_prepared.jsonl", runs_data)
    
        describer = CybenchDescribe(input_files=[input_file], output_dir=output_dir)
        
        custom_analysis_called = False
        def mock_custom_analysis():
            nonlocal custom_analysis_called
            custom_analysis_called = True
        
        monkeypatch.setattr(describer, "generate_custom_analysis", mock_custom_analysis)
        describer.run()
        assert custom_analysis_called 