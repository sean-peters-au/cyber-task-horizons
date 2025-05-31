"""
Unit tests for the Describe base class.

Tests the new data description interface that works with Run objects.
"""

from unittest.mock import patch
import pytest
import tempfile
from pathlib import Path
from typing import List, Optional
import logging

from human_ttc_eval.core.describe import Describe
from human_ttc_eval.core.run import Run

logger = logging.getLogger(__name__)

# Minimal Run data for testing
def create_test_runs(count=3) -> List[Run]:
    runs = []
    for i in range(count):
        runs.append(Run(
            task_id=f"task_id_{i}",
            task_family=f"family_{(i % 2) + 1}", # Two families
            run_id=f"run_id_{i}",
            alias=f"Alias {i}",
            model="human",
            score_binarized=i % 2,
            human_minutes=float(10 + i * 5),
            human_source="test_source",
            task_source="test_dataset",
            started_at=0.0,
            completed_at=float(10 + i*5)*60
        ))
    return runs

def create_runs_jsonl_file(directory: Path, filename: str, runs: List[Run]) -> Path:
    file_path = directory / filename
    Run.save_jsonl(runs, str(file_path))
    return file_path

class MockDescriber(Describe):
    """Mock implementation of Describe for testing base class functionality."""
    MOCK_DATASET_NAME = "mock_describe_dataset"

    # Updated __init__ to match Describe base class
    def __init__(self, input_files: List[Path], output_dir: Path):
        super().__init__(input_files=input_files, output_dir=output_dir)
        self.custom_analysis_called = False

    @property
    def dataset_name(self) -> str:
        # This mock will always have a fixed dataset name.
        return self.MOCK_DATASET_NAME

    def generate_custom_analysis(self) -> None:
        self.custom_analysis_called = True
        logger.info(f"Mock custom analysis called for {self.dataset_name}")
        # Optionally create a dummy custom file
        (self.output_dir / f"{self.dataset_name}_custom_output.txt").write_text("Mock custom analysis data.")


class TestDescribe:
    """Tests for the Describe base class."""

    def test_initialization(self):
        """Test successful initialization."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            input_file = temp_dir / "input.jsonl"
            input_file.touch() # Create dummy file
            output_dir = temp_dir / "output"
            
            describer = MockDescriber(input_files=[input_file], output_dir=output_dir)
            
            assert describer.input_files == [input_file]
            assert describer.output_dir == output_dir
            assert output_dir.exists()
            assert describer.dataset_name == MockDescriber.MOCK_DATASET_NAME

    def test_load_runs_missing_file(self, caplog):
        """Test loading runs when an input file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            missing_file = temp_dir / "missing.jsonl"
            existing_file = temp_dir / "existing.jsonl"
            create_runs_jsonl_file(temp_dir, "existing.jsonl", create_test_runs(1))
            output_dir = temp_dir / "output"
            
            describer = MockDescriber(input_files=[missing_file, existing_file], output_dir=output_dir)
            with caplog.at_level(logging.WARNING):
                loaded_runs = describer.load_runs()
            
            assert f"Input file not found: {missing_file}" in caplog.text
            assert len(loaded_runs) == 1 # Should load from the existing file
            assert describer.df is not None
            assert len(describer.df) == 1

    def test_load_runs_success(self):
        """Test successful loading of runs from one or more files."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            output_dir = temp_dir / "output"
            
            file1_runs = create_test_runs(2)
            input_file1 = create_runs_jsonl_file(temp_dir, "input1.jsonl", file1_runs)
            file2_runs = create_test_runs(3)
            input_file2 = create_runs_jsonl_file(temp_dir, "input2.jsonl", file2_runs)
            
            describer = MockDescriber(input_files=[input_file1, input_file2], output_dir=output_dir)
            loaded_runs = describer.load_runs()
            
            assert len(loaded_runs) == 5 # 2 + 3
            assert describer.df is not None
            assert len(describer.df) == 5
            assert describer.df['human_minutes'].sum() == (10+15) + (10+15+20) # Sum of human_minutes

    def test_generate_overall_stats(self):
        """Test overall statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            input_file = create_runs_jsonl_file(temp_dir, "stats_input.jsonl", create_test_runs(5))
            output_dir = temp_dir / "output"
            describer = MockDescriber(input_files=[input_file], output_dir=output_dir)
            describer.load_runs()
            stats = describer.generate_overall_stats()
            
            assert stats["Unique Tasks"] == 5
            assert stats["Successful Tasks"] == 2 
            assert stats["Task Success Rate (%)"] == (2/5)*100
            assert stats["Human Time (min) - Total"] == (10+15+20+25+30) # 100

    def test_generate_task_family_stats(self):
        """Test task family statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            # Runs: t0 (fam1), t1 (fam2), t2 (fam1), t3 (fam2), t4 (fam1)
            # Fam1: t0, t2, t4 (3 tasks)
            # Fam2: t1, t3 (2 tasks)
            test_runs = create_test_runs(5) 
            input_file = create_runs_jsonl_file(temp_dir, "family_stats.jsonl", test_runs)
            output_dir = temp_dir / "output"
            describer = MockDescriber(input_files=[input_file], output_dir=output_dir)
            describer.load_runs()
            family_stats_df = describer.generate_task_family_stats()
            
            assert not family_stats_df.empty
            assert len(family_stats_df) == 2 # Two families
            fam1_stats = family_stats_df[family_stats_df['task_family'] == 'family_1'].iloc[0]
            assert fam1_stats['num_tasks'] == 3
            # t0 (succ), t2 (succ), t4 (succ) -> all 3 for fam1 with i%2 logic for score_binarized
            # i=0, s=0; i=1, s=1; i=2, s=0; i=3, s=1; i=4, s=0
            # t0 (fam1, s=0), t1 (fam2, s=1), t2 (fam1, s=0), t3 (fam2, s=1), t4 (fam1, s=0)
            # Fam1 tasks: t0 (score 0), t2 (score 0), t4 (score 0). Successful tasks for fam1 = 0.
            assert fam1_stats['successful_tasks'] == 0 
            assert fam1_stats['success_rate'] == 0.0

            fam2_stats = family_stats_df[family_stats_df['task_family'] == 'family_2'].iloc[0]
            assert fam2_stats['num_tasks'] == 2
            # Fam2 tasks: t1 (score 1), t3 (score 1). Successful = 2
            assert fam2_stats['successful_tasks'] == 2
            assert fam2_stats['success_rate'] == 100.0

    def test_save_summary_files(self):
        """Test saving summary files."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            input_file = create_runs_jsonl_file(temp_dir, "input.jsonl", create_test_runs(2))
            output_dir = temp_dir / "output"
            describer = MockDescriber(input_files=[input_file], output_dir=output_dir)
            describer.load_runs()
            describer.save_summary_files()
            
            assert (output_dir / 'overall_summary.csv').exists()
            assert (output_dir / 'task_family_summary.csv').exists()

    def test_run_with_data(self, monkeypatch):
        """Test the full pipeline with data."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            output_dir = temp_dir / "output"
            input_file = create_runs_jsonl_file(temp_dir, "full_run.jsonl", create_test_runs(1))
            
            describer = MockDescriber(input_files=[input_file], output_dir=output_dir)
            
            # Mock plotting methods to avoid actual plotting
            monkeypatch.setattr(describer, 'plot_human_time_distribution', lambda: None)
            monkeypatch.setattr(describer, 'plot_task_family_comparison', lambda: None)
            
            describer.run()
            
            assert (output_dir / 'overall_summary.csv').exists()
            assert describer.custom_analysis_called # from MockDescriber
            assert (output_dir / f"{describer.dataset_name}_custom_output.txt").exists()

    def test_run_without_data(self, caplog):
        """Test the full pipeline without data (empty input file)."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            output_dir = temp_dir / "output"
            input_file = create_runs_jsonl_file(temp_dir, "empty_run.jsonl", []) # Empty runs list
            
            describer = MockDescriber(input_files=[input_file], output_dir=output_dir)
            with caplog.at_level(logging.WARNING):
                describer.run()
            
            assert "No runs loaded. Creating empty summary." in caplog.text
            assert (output_dir / 'summary.txt').exists()
            # Ensure plotting and custom analysis were not called or handled gracefully
            assert not describer.custom_analysis_called
            assert not (output_dir / 'human_time_distribution.png').exists()

    @patch('matplotlib.pyplot.savefig') # Mock savefig to avoid actual file creation during plot test
    def test_plot_generation_with_matplotlib(self, mock_savefig):
        """Test that plots are generated without errors (matplotlib calls are made)."""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            output_dir = temp_dir / "output"
            input_file = create_runs_jsonl_file(temp_dir, "plot_input.jsonl", create_test_runs(5))
            
            describer = MockDescriber(input_files=[input_file], output_dir=output_dir)
            describer.load_runs()
            
            describer.plot_human_time_distribution()
            describer.plot_task_family_comparison()
            
            # Assert that savefig was called for each plot
            # Expected 2 calls: one for human_time_distribution, one for task_family_comparison
            assert mock_savefig.call_count >= 2 

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented."""
        # Test for dataset_name property
        with pytest.raises(TypeError, match="Can\'t instantiate abstract class MinimalDescribeNoName .* abstract method dataset_name"):
            class MinimalDescribeNoName(Describe):
                def __init__(self):
                    # Minimal init for testing abstract dataset_name
                    super().__init__(input_files=[Path("dummy.jsonl")], output_dir=Path("."))
                def generate_custom_analysis(self) -> None: pass
            MinimalDescribeNoName()

        # Test for generate_custom_analysis method
        with pytest.raises(TypeError, match="Can\'t instantiate abstract class MinimalDescribeNoCustom .* abstract method generate_custom_analysis"):
            class MinimalDescribeNoCustom(Describe):
                def __init__(self):
                    super().__init__(input_files=[Path("dummy.jsonl")], output_dir=Path("."))
                @property
                def dataset_name(self) -> str: return "minimal"
            MinimalDescribeNoCustom()

    # Removed test_path_coercion as Describe.__init__ now expects List[Path]
    # and Path objects are created inside from the list elements. 