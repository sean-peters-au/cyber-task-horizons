"""
Unit tests for the Run dataclass.

Tests the METR Run schema implementation including validation,
serialization, and utility methods.
"""

import pytest
import json
import tempfile
from pathlib import Path

from human_ttc_eval.core.run import Run


class TestRunBasics:
    """Test basic Run instantiation and field behavior."""
    
    def test_minimal_run_creation(self):
        """Test creating a Run with minimal required fields."""
        run = Run(
            task_id="test_task/1",
            task_family="test_task",
            run_id="human_test_task_1",
            alias="Human Test",
            model="human",
            score_binarized=1,
            human_minutes=5.0,
            human_source="estimate",
            task_source="test_dataset"
        )
        
        assert run.task_id == "test_task/1"
        assert run.score_binarized == 1
        assert run.human_minutes == 5.0
        
    def test_post_init_calculations(self):
        """Test automatic calculations in __post_init__."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=1,
            human_minutes=10.0,
            human_source="estimate",
            task_source="test"
        )
        
        # Check human_cost calculation
        assert run.human_cost == 10.0 * 1.25  # 12.5
        
        # Check score_cont default
        assert run.score_cont == 1.0
        
        # Check human_score for human model
        assert run.human_score == 1.0
        
    def test_post_init_with_provided_values(self):
        """Test that provided values override automatic calculations."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=0,
            score_cont=0.75,  # Provided value
            human_minutes=10.0,
            human_cost=20.0,  # Provided value
            human_score=0.8,  # Provided value
            human_source="estimate",
            task_source="test"
        )
        
        # Provided values should not be overridden
        assert run.human_cost == 20.0
        assert run.score_cont == 0.75
        assert run.human_score == 0.8
        
    def test_ai_model_human_score(self):
        """Test that AI models don't get automatic human_score."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="ai_test_1",
            alias="GPT-4",
            model="openai/gpt-4",
            score_binarized=1,
            human_minutes=10.0,
            human_source="estimate",
            task_source="test"
        )
        
        # AI models shouldn't get automatic human_score
        assert run.human_score is None


class TestRunValidation:
    """Test Run validation logic."""
    
    def test_valid_run_passes_validation(self):
        """Test that a valid run passes validation."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=1,
            human_minutes=10.0,
            human_source="estimate",
            task_source="test"
        )
        
        # Should not raise
        run.validate()
        
    def test_missing_required_string_fields(self):
        """Test validation fails for missing required string fields."""
        run = Run(
            task_id="",  # Empty string
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=1,
            human_minutes=10.0,
            human_source="estimate",
            task_source="test"
        )
        
        with pytest.raises(ValueError, match="Field 'task_id' must be a non-empty string"):
            run.validate()
            
    def test_invalid_score_binarized(self):
        """Test validation fails for invalid score_binarized."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=2,  # Invalid
            human_minutes=10.0,
            human_source="estimate",
            task_source="test"
        )
        
        with pytest.raises(ValueError, match="score_binarized must be 0 or 1"):
            run.validate()
            
    def test_invalid_human_minutes(self):
        """Test validation fails for invalid human_minutes."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=1,
            human_minutes=-5.0,  # Negative
            human_source="estimate",
            task_source="test"
        )
        
        with pytest.raises(ValueError, match="human_minutes must be positive"):
            run.validate()
            
    def test_invalid_generation_cost(self):
        """Test validation fails for negative generation_cost."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="ai_test_1",
            alias="GPT-4",
            model="openai/gpt-4",
            score_binarized=1,
            human_minutes=10.0,
            generation_cost=-1.0,  # Negative
            human_source="estimate",
            task_source="test"
        )
        
        with pytest.raises(ValueError, match="generation_cost must be non-negative"):
            run.validate()
            
    def test_invalid_score_cont_range(self):
        """Test validation fails for score_cont outside [0,1]."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=1,
            score_cont=1.5,  # Out of range
            human_minutes=10.0,
            human_source="estimate",
            task_source="test"
        )
        
        with pytest.raises(ValueError, match="score_cont must be between 0 and 1"):
            run.validate()


class TestRunSerialization:
    """Test Run serialization methods."""
    
    def test_to_jsonl_dict_excludes_none(self):
        """Test that to_jsonl_dict excludes None values."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=1,
            human_minutes=10.0,
            human_source="estimate",
            task_source="test",
            time_limit=None,  # Should be excluded
            started_at=None   # Should be excluded
        )
        
        data = run.to_jsonl_dict()
        
        # Check required fields are present
        assert data["task_id"] == "test/1"
        assert data["human_minutes"] == 10.0
        
        # Check None fields are excluded
        assert "time_limit" not in data
        assert "started_at" not in data
        
        # Check HUMAN_COST_PER_MINUTE constant is excluded
        assert "HUMAN_COST_PER_MINUTE" not in data
        
    def test_to_jsonl_line(self):
        """Test JSONL line generation."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=1,
            human_minutes=10.0,
            human_source="estimate",
            task_source="test"
        )
        
        line = run.to_jsonl_line()
        
        # Should end with newline
        assert line.endswith('\n')
        
        # Should be valid JSON
        data = json.loads(line.strip())
        assert data["task_id"] == "test/1"
        
    def test_from_dict(self):
        """Test creating Run from dictionary."""
        data = {
            "task_id": "test/1",
            "task_family": "test",
            "run_id": "human_test_1",
            "alias": "Human",
            "model": "human",
            "score_binarized": 1,
            "human_minutes": 10.0,
            "human_source": "estimate",
            "task_source": "test",
            "extra_field": "ignored"  # Should be filtered out
        }
        
        run = Run.from_dict(data)
        
        assert run.task_id == "test/1"
        assert run.human_minutes == 10.0
        
        # Extra fields should be ignored, not cause errors
        assert not hasattr(run, "extra_field")


class TestRunUtilities:
    """Test Run utility methods."""
    
    def test_calculate_weights(self):
        """Test weight calculation and in-place application on Run objects."""
        runs = [
            Run(task_id="t1", task_family="fam1", run_id="r1", alias="A", model="m1", score_binarized=1, human_minutes=10),
            Run(task_id="t2", task_family="fam1", run_id="r2", alias="A", model="m1", score_binarized=1, human_minutes=20),
            Run(task_id="t3", task_family="fam2", run_id="r3", alias="A", model="m1", score_binarized=1, human_minutes=30),
            Run(task_id="t1", task_family="fam1", run_id="r4", alias="B", model="m2", score_binarized=0, human_minutes=10) # Duplicate task_id t1
        ]
        
        Run.calculate_weights(runs) # Should modify runs in-place
        
        # Total unique tasks = 3 (t1, t2, t3)
        expected_equal_weight = 1.0 / 3.0
        # Family counts for unique tasks: fam1 has t1, t2 (2 tasks); fam2 has t3 (1 task)
        expected_invsqrt_fam1 = 1.0 / (2**0.5)
        expected_invsqrt_fam2 = 1.0 / (1**0.5)
        
        for run in runs:
            assert run.equal_task_weight == pytest.approx(expected_equal_weight)
            if run.task_family == "fam1":
                assert run.invsqrt_task_weight == pytest.approx(expected_invsqrt_fam1)
            elif run.task_family == "fam2":
                assert run.invsqrt_task_weight == pytest.approx(expected_invsqrt_fam2)

    def test_calculate_weights_invalid(self):
        """Test weight calculation with empty list or list of runs with no task_family."""
        # Test with empty list
        empty_runs = []
        Run.calculate_weights(empty_runs) # Should not raise error and do nothing
        assert len(empty_runs) == 0

        # Test with runs lacking task_family (should result in invsqrt_weight of 0 or handle gracefully)
        runs_no_family = [
            Run(task_id="t1", task_family="", run_id="r1", alias="A", model="m1", score_binarized=1, human_minutes=10)
        ]
        Run.calculate_weights(runs_no_family)
        assert runs_no_family[0].equal_task_weight == pytest.approx(1.0)
        assert runs_no_family[0].invsqrt_task_weight == 0.0 # As per current logic for missing/empty family

    def test_filter_unique_tasks(self):
        """Test filtering unique task IDs."""
        runs = [
            Run(
                task_id="task/1",
                task_family="task",
                run_id=f"run_{i}",
                alias="Test",
                model="test",
                score_binarized=1,
                human_minutes=5.0,
                human_source="estimate",
                task_source="test"
            )
            for i in range(3)
        ]
        
        # Add runs for different tasks
        runs.append(Run(
            task_id="task/2",
            task_family="task",
            run_id="run_4",
            alias="Test",
            model="test",
            score_binarized=1,
            human_minutes=5.0,
            human_source="estimate",
            task_source="test"
        ))
        
        unique_tasks = Run.filter_unique_tasks(runs)
        
        assert len(unique_tasks) == 2
        assert "task/1" in unique_tasks
        assert "task/2" in unique_tasks
        
    def test_load_save_jsonl(self):
        """Test saving and loading JSONL files."""
        runs = [
            Run(
                task_id=f"test/{i}",
                task_family="test",
                run_id=f"human_test_{i}",
                alias="Human",
                model="human",
                score_binarized=i % 2,
                human_minutes=float(i * 5),
                human_source="estimate",
                task_source="test"
            )
            for i in range(3)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
            
        try:
            # Save runs
            Run.save_jsonl(runs, temp_path)
            
            # Load runs
            loaded_runs = Run.load_jsonl(temp_path)
            
            assert len(loaded_runs) == 3
            assert loaded_runs[0].task_id == "test/0"
            assert loaded_runs[1].human_minutes == 5.0
            assert loaded_runs[2].score_binarized == 0
            
            # Check that calculated fields were preserved
            assert loaded_runs[1].human_cost == 5.0 * 1.25
            
        finally:
            Path(temp_path).unlink()
            
    def test_load_jsonl_empty_lines(self):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"task_id": "test/1", "task_family": "test", "run_id": "r1", "alias": "A", "model": "m", "score_binarized": 1, "human_minutes": 5.0, "human_source": "e", "task_source": "t"}\n')
            f.write('\n')  # Empty line
            f.write('{"task_id": "test/2", "task_family": "test", "run_id": "r2", "alias": "A", "model": "m", "score_binarized": 0, "human_minutes": 10.0, "human_source": "e", "task_source": "t"}\n')
            temp_path = f.name
            
        try:
            runs = Run.load_jsonl(temp_path)
            assert len(runs) == 2  # Empty line should be skipped
            assert runs[0].task_id == "test/1"
            assert runs[1].task_id == "test/2"
        finally:
            Path(temp_path).unlink()


class TestRunEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_very_long_human_minutes(self):
        """Test handling of very long task times."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=1,
            human_minutes=1440.0,  # 24 hours
            human_source="estimate",
            task_source="test"
        )
        
        assert run.human_cost == 1440.0 * 1.25  # $1800
        run.validate()  # Should pass
        
    def test_zero_score_cont(self):
        """Test that zero score_cont is valid."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="human_test_1",
            alias="Human",
            model="human",
            score_binarized=0,
            score_cont=0.0,  # Explicitly zero
            human_minutes=10.0,
            human_source="estimate",
            task_source="test"
        )
        
        run.validate()  # Should pass
        assert run.score_cont == 0.0
        
    def test_fatal_error_run(self):
        """Test run with fatal error."""
        run = Run(
            task_id="test/1",
            task_family="test",
            run_id="ai_test_1",
            alias="GPT-4",
            model="openai/gpt-4",
            score_binarized=0,
            human_minutes=10.0,
            human_source="estimate",
            task_source="test",
            fatal_error_from="Connection timeout"
        )
        
        run.validate()  # Should pass despite error
        assert run.fatal_error_from == "Connection timeout"
        assert run.score_binarized == 0 