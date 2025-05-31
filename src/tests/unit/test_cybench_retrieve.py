"""
Unit tests for CybenchRetrieve class.

Tests the retrieval of raw CyBench challenge data from a repository.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Optional
import logging
import builtins

from human_ttc_eval.datasets.cybench.cybench_retrieve import CybenchRetrieve
from human_ttc_eval import config

# Fixture to mock config.DATA_DIR for tests, similar to test_retrieve.py
ORIGINAL_DATA_DIR = config.DATA_DIR
ORIGINAL_CYBENCH_REPO_PATH = config.CYBENCH_REPO_PATH

@pytest.fixture(autouse=True)
def mock_cybench_config_paths(tmp_path_factory):
    """Fixture to temporarily change config.DATA_DIR and config.CYBENCH_REPO_PATH for tests."""
    temp_data_root = tmp_path_factory.mktemp("test_cybench_data_root")
    temp_repo_root = tmp_path_factory.mktemp("test_cybench_repo_root")
    config.DATA_DIR = temp_data_root
    config.CYBENCH_REPO_PATH = temp_repo_root
    yield
    config.DATA_DIR = ORIGINAL_DATA_DIR
    config.CYBENCH_REPO_PATH = ORIGINAL_CYBENCH_REPO_PATH

class TestCybenchRetrieve:
    """Test the CybenchRetrieve class."""
    
    def test_initialization(self):
        """Test successful initialization."""
        retriever = CybenchRetrieve(dataset_name="cybench")
        assert retriever.dataset_name == "cybench"
        assert retriever.cybench_repo_path == config.CYBENCH_REPO_PATH
        assert retriever.output_dir == config.DATA_DIR / "raw" / "cybench"
        assert retriever.output_dir.exists()
        assert retriever.output_filename == CybenchRetrieve.DEFAULT_OUTPUT_FILENAME
    
    def test_initialization_with_missing_repo_logs_warning(self, caplog):
        """Test initialization logs a warning if CYBENCH_REPO_PATH does not exist."""
        original_mocked_repo_path = config.CYBENCH_REPO_PATH
        non_existent_repo_path = original_mocked_repo_path.parent / "non_existent_cybench_repo"
        if non_existent_repo_path.exists():
            if non_existent_repo_path.is_dir(): non_existent_repo_path.rmdir()
            else: non_existent_repo_path.unlink()
        
        config.CYBENCH_REPO_PATH = non_existent_repo_path
        try:
            with caplog.at_level(logging.WARNING, logger="human_ttc_eval.datasets.cybench.cybench_retrieve"):
                _ = CybenchRetrieve(dataset_name="cybench_missing_repo_test")
            assert f"CyBench repository path does not exist: {non_existent_repo_path}" in caplog.text
        finally:
            config.CYBENCH_REPO_PATH = original_mocked_repo_path

    def _setup_mock_repo(self, repo_root_path: Path, task_list_content: str, fst_data_content: Optional[list] = None):
        """Helper to set up a mock CyBench repository structure."""
        repo_root_path.mkdir(parents=True, exist_ok=True)
        (repo_root_path / "analytics").mkdir(parents=True, exist_ok=True)
        (repo_root_path / "task_list.txt").write_text(task_list_content)
        if fst_data_content is not None:
            (repo_root_path / "analytics" / "CTF fastest solve times.json").write_text(json.dumps(fst_data_content))

    def test_retrieve_success(self):
        """Test successful retrieval of challenge data."""
        dataset_name = "cybench_retrieve_success"
        mock_repo_dir = config.CYBENCH_REPO_PATH 

        task_list_content = "benchmark/hackthebox/catctf/crypto/Task1\nbenchmark/other/catctf/web/Task2"
        fst_content = [
            {"challenge_dir": "benchmark/hackthebox/catctf/crypto/Task1", "challenge_fastest_solve_time": "0:10:00"},
            {"challenge_dir": "benchmark/other/catctf/web/Task2", "challenge_fastest_solve_time": "0:20:00"}
        ]
        self._setup_mock_repo(mock_repo_dir, task_list_content, fst_content)

        task1_dir = mock_repo_dir / "benchmark" / "hackthebox" / "catctf" / "crypto" / "Task1"
        task1_dir.mkdir(parents=True)
        (task1_dir / "README.md").write_text("> Author: Alice\n> Difficulty: Easy\nDescription for T1.")

        task2_dir = mock_repo_dir / "benchmark" / "other" / "catctf" / "web" / "Task2"
        task2_dir.mkdir(parents=True)
        (task2_dir / "README.md").write_text("> Author: Bob\n> Difficulty: Medium\nDescription for T2.")

        retriever = CybenchRetrieve(dataset_name=dataset_name)
        result_path = retriever.retrieve()

        expected_output_path = config.DATA_DIR / "raw" / dataset_name / CybenchRetrieve.DEFAULT_OUTPUT_FILENAME
        assert result_path == expected_output_path
        assert result_path.exists()

        with open(result_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])
            assert data1["task_id"] == "benchmark/hackthebox/catctf/crypto/Task1"
            assert data1["readme_data"]["author"] == "Alice"
            assert data1["fastest_solve_time_seconds"] == 600.0
            assert data2["task_id"] == "benchmark/other/catctf/web/Task2"
            assert data2["readme_data"]["author"] == "Bob"
            assert data2["fastest_solve_time_seconds"] == 1200.0

    def test_retrieve_custom_filename(self):
        """Test retrieval with custom output filename from __init__."""
        dataset_name = "cybench_default_retrieve"
        mock_repo_dir = config.CYBENCH_REPO_PATH
        self._setup_mock_repo(mock_repo_dir, "benchmark/dummy/task_A")
        (mock_repo_dir / "benchmark" / "dummy" / "task_A").mkdir(parents=True, exist_ok=True)
        ((mock_repo_dir / "benchmark" / "dummy" / "task_A") / "README.md").touch()

        retriever = CybenchRetrieve(dataset_name=dataset_name)
        result_path = retriever.retrieve()

        expected_output_path = config.DATA_DIR / "raw" / dataset_name / CybenchRetrieve.DEFAULT_OUTPUT_FILENAME
        assert result_path == expected_output_path
        assert result_path.exists()

    def test_retrieve_missing_task_list(self, caplog):
        """Test retrieval when task_list.txt is missing."""
        mock_repo_dir = config.CYBENCH_REPO_PATH
        task_list_file = mock_repo_dir / "task_list.txt"
        if task_list_file.exists(): task_list_file.unlink()

        retriever = CybenchRetrieve(dataset_name="cybench_missing_tl")
        with caplog.at_level(logging.ERROR, logger="human_ttc_eval.datasets.cybench.cybench_retrieve"):
            result_path = retriever.retrieve()
        
        assert result_path is None
        assert f"task_list.txt not found in CyBench repository: {task_list_file}" in caplog.text

    def test_retrieve_skips_missing_task_dirs(self, caplog):
        """Test that retrieval skips tasks with missing directories and logs warnings."""
        mock_repo_dir = config.CYBENCH_REPO_PATH
        task_list_content = "existing_category/real_task\nmissing_category/ghost_task"
        self._setup_mock_repo(mock_repo_dir, task_list_content)

        (mock_repo_dir / "existing_category" / "real_task").mkdir(parents=True)
        ((mock_repo_dir / "existing_category" / "real_task") / "README.md").write_text("Real task")

        retriever = CybenchRetrieve(dataset_name="cybench_skip_missing")
        result_path = retriever.retrieve()
        assert result_path is not None
        with open(result_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["task_id"] == "existing_category/real_task"
        assert f"Challenge directory not found: {mock_repo_dir / 'missing_category' / 'ghost_task'}" in caplog.text

    def test_parse_time_to_seconds(self):
        """Test _parse_time_to_seconds utility."""
        retriever = CybenchRetrieve()
        assert retriever._parse_time_to_seconds("1:02:03") == 3723.0
        assert retriever._parse_time_to_seconds("05:30") == 330.0
        assert retriever._parse_time_to_seconds(None) is None
        assert retriever._parse_time_to_seconds("invalid") is None
        assert retriever._parse_time_to_seconds("1:2:3:4") is None
        assert retriever._parse_time_to_seconds("1:not_int:30") is None

    def test_parse_readme_data(self):
        """Test _parse_readme_data utility."""
        retriever = CybenchRetrieve(dataset_name="test_readme_parsing")
        
        readme_content = """# Test Challenge

> Author: test_author
> Difficulty: Medium
> Points: 500

This is a test challenge description.
It has multiple lines.

## Another Section
Should not be included in description.
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md", encoding='utf-8') as tmp_file:
            tmp_file.write(readme_content)
            readme_path = Path(tmp_file.name)
        
        try:
            data = retriever._parse_readme_data(readme_path)
            assert data["author"] == "test_author", "Author not parsed correctly"
            assert data["difficulty"] == "Medium", "Difficulty not parsed correctly"
            assert data["points"] == 500, "Points not parsed correctly"
            assert data["description"] == "This is a test challenge description. It has multiple lines.", "Description not parsed correctly"
        finally:
            readme_path.unlink() # Ensure cleanup of the temp file

        # Test with markdown table (should be ignored for author, but description logic might differ)
        readme_with_table = """# Challenge

| Column1 | Column2 |
|---------|---------|
| author  | value   |

> Author: john_doe
Description after table.
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md", encoding='utf-8') as tmp_file:
            tmp_file.write(readme_with_table)
            readme_path_table = Path(tmp_file.name)
        
        try:
            data_table = retriever._parse_readme_data(readme_path_table)
            assert data_table.get("author") == "john_doe", "Author from table case failed"
            assert data_table.get("description") == "Description after table.", "Description with table case failed"
        finally:
            readme_path_table.unlink()
            
        # Test multiple authors
        readme_multi_authors = """# Challenge
> Author: alice, bob, charlie
Description for multi-author.
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md", encoding='utf-8') as tmp_file:
            tmp_file.write(readme_multi_authors)
            readme_path_multi = Path(tmp_file.name)

        try:
            data_multi = retriever._parse_readme_data(readme_path_multi)
            # The current parsing splits by ':' and takes [1], so "alice, bob, charlie" is one string.
            # This might be desired or might need further splitting in the parser if individual authors are needed.
            # For now, asserting the raw parsed value.
            assert data_multi.get("author") == "alice, bob, charlie", "Multi-author string failed"
            assert data_multi.get("description") == "Description for multi-author.", "Description for multi-author case failed"
        finally:
            readme_path_multi.unlink()
    
    def test_extract_challenge_data(self):
        """Test _extract_challenge_data utility."""
        mock_repo_dir = config.CYBENCH_REPO_PATH
        challenge_rel_path = "benchmark/event_x/ctf_y/category_z/MyTask"
        full_challenge_path = mock_repo_dir / challenge_rel_path
        full_challenge_path.mkdir(parents=True)
        (full_challenge_path / "README.md").write_text("> Author: ExtractorTest\n> Difficulty: Insane\n> Points: 1000\nActual description.")

        retriever = CybenchRetrieve()
        data = retriever._extract_challenge_data(challenge_rel_path)
        assert data is not None
        assert data["task_id"] == challenge_rel_path
        assert data["name"] == "MyTask"
        assert data["category"] == "category_z"
        assert data["event"] == "event_x"
        assert data["ctf_name"] == "ctf_y"
        assert data["readme_data"]["author"] == "ExtractorTest"
        assert data["readme_data"]["difficulty"] == "Insane"
        assert data["readme_data"]["points"] == 1000
        assert data["readme_data"]["description"] == "Actual description."

    def test_retrieve_without_fastest_solve_times(self, caplog):
        """Test retrieval when fastest solve times file is missing but repo and task list exist."""
        mock_repo_dir = config.CYBENCH_REPO_PATH
        task_list_content = "benchmark/test/task_no_fst_data"
        self._setup_mock_repo(mock_repo_dir, task_list_content)
        
        fst_file_path = mock_repo_dir / "analytics" / "CTF fastest solve times.json"
        if fst_file_path.exists(): fst_file_path.unlink()

        task_dir = mock_repo_dir / "benchmark" / "test" / "task_no_fst_data"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "README.md").write_text("A task without FST data available.")

        retriever = CybenchRetrieve(dataset_name="cybench_no_fst")
        with caplog.at_level(logging.INFO, logger="human_ttc_eval.datasets.cybench.cybench_retrieve"):
            result_path = retriever.retrieve()
        
        assert result_path is not None
        assert f"Fastest solve times file not found: {fst_file_path}" in caplog.text
        with open(result_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert "fastest_solve_time_seconds" not in data
            assert data["task_id"] == "benchmark/test/task_no_fst_data"

    def test_retrieve_error_handling(self, caplog):
        """Test retrieval when repo itself is missing (after successful init)."""
        original_mocked_repo_path = config.CYBENCH_REPO_PATH
        
        retriever = CybenchRetrieve(dataset_name="cybench_repo_disappears")
        
        disappeared_repo_path = original_mocked_repo_path.parent / "truly_gone_repo"
        if disappeared_repo_path.exists():
            if disappeared_repo_path.is_dir(): disappeared_repo_path.rmdir()
            else: disappeared_repo_path.unlink()
        
        config.CYBENCH_REPO_PATH = disappeared_repo_path
        retriever.cybench_repo_path = disappeared_repo_path

        try:
            with caplog.at_level(logging.ERROR, logger="human_ttc_eval.datasets.cybench.cybench_retrieve"):
                result_path = retriever.retrieve()
            assert result_path is None
            assert f"CyBench repository not found at {disappeared_repo_path}" in caplog.text
        finally:
            config.CYBENCH_REPO_PATH = original_mocked_repo_path

    def test_retrieve_io_error_reading_task_list(self, monkeypatch, caplog):
        """Test IOError when reading task_list.txt."""
        mock_repo_dir = config.CYBENCH_REPO_PATH
        (mock_repo_dir / "task_list.txt").touch()

        def mock_open_raises_ioerror(*args, **kwargs):
            raise IOError("Simulated read error")

        monkeypatch.setattr("builtins.open", mock_open_raises_ioerror)
        retriever = CybenchRetrieve(dataset_name="cybench_io_error_tasklist")
        
        original_open = builtins.open
        task_list_path_to_mock = mock_repo_dir / "task_list.txt"
        def specific_mock_open(file, *args, **kwargs):
            if Path(file) == task_list_path_to_mock:
                raise IOError("Simulated read error for task_list.txt")
            return original_open(file, *args, **kwargs)
        
        monkeypatch.setattr(builtins, "open", specific_mock_open)

        result = retriever.retrieve()
        assert result is None
        assert "Error reading task list" in caplog.text
        assert "Simulated read error for task_list.txt" in caplog.text

    def test_retrieve_io_error_writing_output(self, monkeypatch, caplog):
        """Test IOError when writing the output JSONL file."""
        mock_repo_dir = config.CYBENCH_REPO_PATH
        self._setup_mock_repo(mock_repo_dir, "category/some_task")
        (mock_repo_dir / "category" / "some_task").mkdir(parents=True, exist_ok=True)
        ((mock_repo_dir / "category" / "some_task") / "README.md").touch()

        retriever = CybenchRetrieve(dataset_name="cybench_io_error_output")
        expected_output_file = retriever.output_dir / retriever.output_filename

        original_open = builtins.open
        def specific_mock_open_for_write(file, mode='r', *args, **kwargs):
            if Path(file) == expected_output_file and mode == 'w':
                raise IOError("Simulated write error")
            return original_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", specific_mock_open_for_write)

        result = retriever.retrieve()
        assert result is None
        assert f"Error writing CyBench raw data to {expected_output_file}" in caplog.text
        assert "Simulated write error" in caplog.text 