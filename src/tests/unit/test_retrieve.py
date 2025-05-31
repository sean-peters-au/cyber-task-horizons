"""
Unit tests for the Retrieve base class.

Tests the new simplified retriever interface.
"""

import pytest
from pathlib import Path
from typing import Optional

from human_ttc_eval.core.retrieve import Retrieve
from human_ttc_eval import config

# Store original DATA_DIR and override it for tests
ORIGINAL_DATA_DIR = config.DATA_DIR

@pytest.fixture(autouse=True)
def mock_config_paths(tmp_path_factory):
    """Fixture to temporarily change config.DATA_DIR for tests."""
    temp_data_root = tmp_path_factory.mktemp("test_data_root")
    config.DATA_DIR = temp_data_root
    yield
    config.DATA_DIR = ORIGINAL_DATA_DIR

class MockRetrieve(Retrieve):
    """Mock implementation of Retrieve for testing base class functionality."""
    MOCK_DATASET_NAME = "mock_retrieve_dataset"

    def __init__(self, dataset_name: str = MOCK_DATASET_NAME, fail_retrieve: bool = False):
        super().__init__(dataset_name) 
        self.fail_retrieve = fail_retrieve


    def retrieve(self) -> Optional[Path]:
        """Simulates data retrieval.

        Creates a dummy file in self.output_dir if successful.
        """
        if self.fail_retrieve:
            return None
        
        if not self.output_dir.exists():
            pytest.fail(f"MockRetrieve: self.output_dir {self.output_dir} does not exist before creating dummy file.")

        dummy_file = self.output_dir / f"{self.dataset_name}_raw_data.txt"
        try:
            with open(dummy_file, "w") as f:
                f.write("mock raw data")
            return dummy_file
        except Exception as e:
            pytest.fail(f"Failed to create dummy file in MockRetrieve: {e}")
            return None 

    def cleanup(self) -> None:
        """Simulates cleanup."""
        self.cleanup_called = True
        super().cleanup()


class TestRetrieve:
    """Tests for the Retrieve base class."""

    def test_initialization(self):
        """Test successful initialization and output_dir derivation."""
        dataset_name = "test_init_dataset"
        retriever = MockRetrieve(dataset_name=dataset_name)
        assert retriever.dataset_name == dataset_name
        expected_output_dir = config.DATA_DIR / "raw" / dataset_name
        assert retriever.output_dir == expected_output_dir
        assert retriever.output_dir.exists() 

    def test_initialization_failure_os_error(self, monkeypatch):
        """Test that __init__ raises OSError if directory creation fails and is re-raised."""
        dataset_name = "test_fail_oserror"
        
        def mock_mkdir_raises_os_error(*args, **kwargs):
            raise OSError("Test-induced OSError")

        monkeypatch.setattr(Path, "mkdir", mock_mkdir_raises_os_error)
        
        with pytest.raises(OSError, match="Test-induced OSError"):
            MockRetrieve(dataset_name=dataset_name)

    def test_output_dir_creation_implicit(self):
        """Test that output_dir is created by __init__."""
        dataset_name = "test_dir_creation"
        retriever = MockRetrieve(dataset_name=dataset_name)
        assert (config.DATA_DIR / "raw" / dataset_name).exists()

    def test_retrieve_method_success(self):
        """Test the retrieve method mock: successful retrieval."""
        dataset_name = "test_retrieve_success"
        retriever = MockRetrieve(dataset_name=dataset_name)
        result = retriever.retrieve()
        assert result is not None
        assert isinstance(result, Path)
        assert result.name == f"{dataset_name}_raw_data.txt"
        assert result.exists()
        assert result.parent == retriever.output_dir

    def test_retrieve_method_failure(self):
        """Test the retrieve method mock: failed retrieval (returns None)."""
        retriever = MockRetrieve(fail_retrieve=True)
        result = retriever.retrieve()
        assert result is None

    def test_cleanup_method_called_by_run(self):
        """Test that the cleanup method is called by run()."""
        retriever = MockRetrieve()
        retriever.cleanup_called = False 
        retriever.run()
        assert hasattr(retriever, 'cleanup_called') and retriever.cleanup_called

    def test_run_method_orchestration_success(self):
        """Test the run method orchestrates retrieve and cleanup on success."""
        dataset_name = "test_run_orchestration"
        retriever = MockRetrieve(dataset_name=dataset_name)
        retriever.cleanup_called = False

        result = retriever.run()
        assert result is not None
        assert isinstance(result, Path) 
        assert result.name == f"{dataset_name}_raw_data.txt"
        assert retriever.cleanup_called

    def test_run_method_orchestration_retrieve_returns_none(self):
        """Test run method when retrieve() returns None (simulated failure)."""
        retriever = MockRetrieve(fail_retrieve=True)
        retriever.cleanup_called = False
        
        # If retrieve() returns None, run() should call cleanup and return None.
        # It should not raise an exception in this specific scenario.
        result = retriever.run()
        assert result is None
        assert retriever.cleanup_called

    def test_run_method_orchestration_retrieve_raises_exception(self, monkeypatch):
        """Test run method when retrieve() itself raises an exception."""
        dataset_name = "test_run_retrieve_exception"
        retriever = MockRetrieve(dataset_name=dataset_name)
        retriever.cleanup_called = False

        def mock_retrieve_raises_exception(*args, **kwargs):
            raise ValueError("Simulated retrieve error")

        monkeypatch.setattr(retriever, "retrieve", mock_retrieve_raises_exception)

        with pytest.raises(ValueError, match="Simulated retrieve error"):
            retriever.run()
        
        # In this case, cleanup might not be called if retrieve fails before cleanup can run
        # As per current Retrieve.run(), if retrieve() raises, cleanup() is skipped.
        assert not retriever.cleanup_called 

    def test_abstract_methods_enforcement(self):
        """Test that abstract methods must be implemented. Only retrieve() is abstract."""
        with pytest.raises(TypeError, match="Can\'t instantiate abstract class BadRetrieve .* "):
            class BadRetrieve(Retrieve):
                def __init__(self):
                    super().__init__(dataset_name="bad_retrieve")
                # Missing retrieve() and cleanup()
                pass
            BadRetrieve()
