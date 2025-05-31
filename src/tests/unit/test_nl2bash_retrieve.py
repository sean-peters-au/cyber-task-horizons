"""
Tests for NL2Bash dataset retriever.
"""

import pytest
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock

from human_ttc_eval.datasets.nl2bash.nl2bash_retrieve import NL2BashRetrieve
from human_ttc_eval import config


class TestNL2BashRetrieve:
    """Test cases for NL2BashRetrieve class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration paths."""
        with patch.object(config, 'NL2BASH_REPO_PATH', Path('/mock/third-party/nl2bash')):
            with patch.object(config, 'DATA_DIR', Path('/mock/data')):
                # Mock the mkdir call that happens during Retrieve.__init__
                with patch('pathlib.Path.mkdir'):
                    yield
    
    @pytest.fixture
    def retriever(self, mock_config):
        """Create a NL2BashRetrieve instance with mocked config."""
        return NL2BashRetrieve()
    
    def test_init_default(self, mock_config):
        """Test initialization with default parameters."""
        retriever = NL2BashRetrieve()
        
        assert retriever.dataset_name == "nl2bash"
        assert retriever.nl2bash_repo_path == Path('/mock/third-party/nl2bash')
        assert retriever.source_data_path == Path('/mock/third-party/nl2bash/data/bash')
        assert retriever.output_dir == Path('/mock/data/raw/nl2bash')
    
    def test_init_custom_dataset_name(self, mock_config):
        """Test initialization with custom dataset name."""
        retriever = NL2BashRetrieve("custom_nl2bash")
        
        assert retriever.dataset_name == "custom_nl2bash"
        assert retriever.output_dir == Path('/mock/data/raw/custom_nl2bash')
    
    @patch('human_ttc_eval.datasets.nl2bash.nl2bash_retrieve.shutil.copy2')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_retrieve_success(self, mock_exists, mock_open_func, mock_copy2, retriever):
        """Test successful file retrieval."""
        # Mock file existence - all files exist
        mock_exists.return_value = True
        
        # Mock file reading for line counting
        mock_open_func.side_effect = [
            mock_open(read_data="line1\nline2\nline3\n").return_value,
            mock_open(read_data="cmd1\ncmd2\ncmd3\n").return_value
        ]
        
        # Mock _ensure_repository
        with patch.object(retriever, '_ensure_repository'):
            result = retriever.retrieve()
        
        # Verify results
        assert len(result) == 2
        assert result[0] == retriever.output_dir / "all.nl"
        assert result[1] == retriever.output_dir / "all.cm"
        
        # Verify copy operations
        assert mock_copy2.call_count == 2
        mock_copy2.assert_any_call(
            retriever.source_data_path / 'all.nl',
            retriever.output_dir / 'all.nl'
        )
        mock_copy2.assert_any_call(
            retriever.source_data_path / 'all.cm',
            retriever.output_dir / 'all.cm'
        )
    
    @patch('human_ttc_eval.datasets.nl2bash.nl2bash_retrieve.shutil.copy2')
    @patch('builtins.open')
    @patch('human_ttc_eval.datasets.nl2bash.nl2bash_retrieve.logger')
    @patch('pathlib.Path.exists')
    def test_retrieve_line_count_mismatch(self, mock_exists, mock_logger, mock_open_func, mock_copy2, retriever):
        """Test retrieval with mismatched line counts."""
        # Mock file existence - all files exist
        mock_exists.return_value = True
        
        # Mock file reading with different line counts
        mock_open_func.side_effect = [
            mock_open(read_data="line1\nline2\nline3\n").return_value,
            mock_open(read_data="cmd1\ncmd2\n").return_value  # Different count
        ]
        
        with patch.object(retriever, '_ensure_repository'):
            result = retriever.retrieve()
        
        # Should still succeed but log warning
        assert len(result) == 2
        mock_logger.warning.assert_called_once_with(
            "Line count mismatch: 3 NL lines vs 2 CM lines"
        )
    
    @patch('human_ttc_eval.datasets.nl2bash.nl2bash_retrieve.subprocess.run')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_ensure_repository_clone_success(self, mock_mkdir, mock_exists, mock_run, retriever):
        """Test successful repository cloning."""
        # Mock repository doesn't exist
        mock_exists.return_value = False
        
        # Mock successful git clone
        mock_run.return_value = Mock(returncode=0)
        
        retriever._ensure_repository()
        
        # Verify mkdir was called for parent directory
        mock_mkdir.assert_called()
        
        # Verify git clone was called
        mock_run.assert_called_once_with(
            ["git", "clone", "--depth=1", "https://github.com/TellinaTool/nl2bash.git", str(retriever.nl2bash_repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
    
    @patch('human_ttc_eval.datasets.nl2bash.nl2bash_retrieve.subprocess.run')
    @patch('pathlib.Path.exists')
    def test_ensure_repository_exists_pull_success(self, mock_exists, mock_run, retriever):
        """Test repository update when repository exists."""
        # Mock repository exists
        mock_exists.return_value = True
        
        # Mock successful git pull
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        retriever._ensure_repository()
        
        # Verify git pull was called
        mock_run.assert_called_once_with(
            ["git", "pull"],
            cwd=retriever.nl2bash_repo_path,
            capture_output=True,
            text=True,
            check=False
        )
    
    def test_cleanup(self, retriever):
        """Test cleanup method (should do nothing)."""
        # Should not raise any exceptions
        retriever.cleanup()
    
    @patch('human_ttc_eval.datasets.nl2bash.nl2bash_retrieve.shutil.copy2')
    @patch('builtins.open')
    @patch('human_ttc_eval.datasets.nl2bash.nl2bash_retrieve.logger')
    @patch('pathlib.Path.exists')
    def test_retrieve_with_logging(self, mock_exists, mock_logger, mock_open_func, mock_copy2, retriever):
        """Test that retrieve method logs appropriate messages."""
        # Mock file existence - all files exist
        mock_exists.return_value = True
        
        # Mock file reading
        mock_open_func.side_effect = [
            mock_open(read_data="line1\nline2\n").return_value,
            mock_open(read_data="cmd1\ncmd2\n").return_value
        ]
        
        with patch.object(retriever, '_ensure_repository'):
            retriever.retrieve()
        
        # Check logging calls
        mock_logger.info.assert_any_call(
            f"Starting retrieval of NL2Bash dataset to {retriever.output_dir}"
        )
        mock_logger.info.assert_any_call(
            "Retrieved NL2Bash dataset: 2 NL descriptions, 2 commands"
        )


class TestNL2BashRetrieveIntegration:
    """Integration tests that test the retriever with minimal mocking."""
    
    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Create temporary paths for testing."""
        repo_path = tmp_path / "repo" / "nl2bash"
        data_dir = tmp_path / "data"
        
        # Create directory structure
        repo_data_dir = repo_path / "data" / "bash"
        repo_data_dir.mkdir(parents=True)
        
        # Create test files
        (repo_data_dir / "all.nl").write_text("task 1\ntask 2\ntask 3\n")
        (repo_data_dir / "all.cm").write_text("cmd 1\ncmd 2\ncmd 3\n")
        
        return repo_path, data_dir
    
    def test_retrieve_integration(self, temp_paths):
        """Test retrieval with real file operations."""
        repo_path, data_dir = temp_paths
        
        # Patch config to use temp paths
        with patch.object(config, 'NL2BASH_REPO_PATH', repo_path):
            with patch.object(config, 'DATA_DIR', data_dir):
                retriever = NL2BashRetrieve()
                
                # Mock the repository check since we created it manually
                with patch.object(retriever, '_ensure_repository'):
                    result = retriever.retrieve()
                
                # Verify files were copied
                assert len(result) == 2
                
                output_nl = data_dir / "raw" / "nl2bash" / "all.nl"
                output_cm = data_dir / "raw" / "nl2bash" / "all.cm"
                
                assert output_nl.exists()
                assert output_cm.exists()
                assert output_nl.read_text() == "task 1\ntask 2\ntask 3\n"
                assert output_cm.read_text() == "cmd 1\ncmd 2\ncmd 3\n" 