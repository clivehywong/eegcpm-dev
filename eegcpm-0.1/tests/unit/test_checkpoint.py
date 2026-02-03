"""Tests for checkpoint module."""

import json
import pickle
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from eegcpm.pipeline.base import ModuleResult
from eegcpm.pipeline.checkpoint import (
    CheckpointManager,
    generate_reproducibility_hash,
)


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_initialization(self):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir)

            assert manager.checkpoint_dir.exists()
            assert manager.checkpoint_dir == checkpoint_dir

    def test_initialization_creates_directory(self):
        """Test that initialization creates the checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "checkpoints"
            manager = CheckpointManager(nested_dir)

            assert nested_dir.exists()

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            result = ModuleResult(
                success=True,
                module_name="preprocessing",
                execution_time_seconds=10.5,
                output_files=[Path("/tmp/output.fif")],
                metadata={"n_channels": 64},
            )

            # Save checkpoint
            path = manager.save(
                subject_id="sub-001",
                module_name="preprocessing",
                result=result,
            )

            assert path.exists()

            # Load checkpoint
            loaded = manager.load_module_checkpoint("sub-001", "preprocessing")

            assert loaded is not None
            assert loaded["subject_id"] == "sub-001"
            assert loaded["module_name"] == "preprocessing"
            assert loaded["success"] is True
            assert loaded["execution_time"] == 10.5
            assert loaded["metadata"]["n_channels"] == 64

    def test_save_with_data(self):
        """Test saving checkpoint with pickled data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            result = ModuleResult(
                success=True,
                module_name="epochs",
                execution_time_seconds=5.0,
            )

            test_data = {"array": [1, 2, 3], "value": 42}

            path = manager.save(
                subject_id="sub-001",
                module_name="epochs",
                result=result,
                data=test_data,
            )

            # Load data
            loaded_data = manager.load_data("sub-001", "epochs")

            assert loaded_data is not None
            assert loaded_data["array"] == [1, 2, 3]
            assert loaded_data["value"] == 42

    def test_load_nonexistent_checkpoint(self):
        """Test loading a checkpoint that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            result = manager.load("sub-nonexistent")
            assert result is None

            result = manager.load_module_checkpoint("sub-001", "nonexistent")
            assert result is None

    def test_load_nonexistent_data(self):
        """Test loading data that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Save checkpoint without data
            result = ModuleResult(
                success=True,
                module_name="test",
                execution_time_seconds=1.0,
            )
            manager.save("sub-001", "test", result)

            # Try to load data
            data = manager.load_data("sub-001", "test")
            assert data is None

    def test_get_completed_modules(self):
        """Test getting list of completed modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # No modules completed initially
            completed = manager.get_completed_modules("sub-001")
            assert completed == []

            # Add some modules
            for module_name in ["preprocessing", "epochs", "source"]:
                result = ModuleResult(
                    success=True,
                    module_name=module_name,
                    execution_time_seconds=1.0,
                )
                manager.save("sub-001", module_name, result)

            completed = manager.get_completed_modules("sub-001")
            assert len(completed) == 3
            assert "preprocessing" in completed
            assert "epochs" in completed
            assert "source" in completed

    def test_clear_subject_checkpoints(self):
        """Test clearing checkpoints for a subject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Add some checkpoints
            for module_name in ["mod1", "mod2"]:
                result = ModuleResult(
                    success=True,
                    module_name=module_name,
                    execution_time_seconds=1.0,
                )
                manager.save("sub-001", module_name, result)

            # Verify they exist
            assert manager.get_completed_modules("sub-001") == ["mod1", "mod2"]

            # Clear
            manager.clear("sub-001")

            # Verify cleared
            assert manager.get_completed_modules("sub-001") == []
            assert not (manager.checkpoint_dir / "sub-001").exists()

    def test_clear_all_checkpoints(self):
        """Test clearing all checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Add checkpoints for multiple subjects
            for subject_id in ["sub-001", "sub-002", "sub-003"]:
                result = ModuleResult(
                    success=True,
                    module_name="test",
                    execution_time_seconds=1.0,
                )
                manager.save(subject_id, "test", result)

            # Verify they exist
            assert len(list(manager.checkpoint_dir.glob("sub-*"))) == 3

            # Clear all
            manager.clear_all()

            # Verify cleared (directory still exists but is empty)
            assert manager.checkpoint_dir.exists()
            assert len(list(manager.checkpoint_dir.glob("sub-*"))) == 0

    def test_manifest_update(self):
        """Test that manifest is updated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            result1 = ModuleResult(
                success=True,
                module_name="mod1",
                execution_time_seconds=1.0,
            )
            manager.save("sub-001", "mod1", result1)

            # Check manifest
            manifest = manager.load("sub-001")
            assert manifest is not None
            assert manifest["subject_id"] == "sub-001"
            assert manifest["completed_modules"] == ["mod1"]
            assert "last_updated" in manifest

            # Add another module
            result2 = ModuleResult(
                success=True,
                module_name="mod2",
                execution_time_seconds=2.0,
            )
            manager.save("sub-001", "mod2", result2)

            manifest = manager.load("sub-001")
            assert manifest["completed_modules"] == ["mod1", "mod2"]

    def test_no_duplicate_modules_in_manifest(self):
        """Test that saving the same module twice doesn't duplicate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            for _ in range(3):
                result = ModuleResult(
                    success=True,
                    module_name="test",
                    execution_time_seconds=1.0,
                )
                manager.save("sub-001", "test", result)

            completed = manager.get_completed_modules("sub-001")
            assert completed == ["test"]

    def test_multiple_subjects(self):
        """Test managing checkpoints for multiple subjects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            subjects = ["sub-001", "sub-002", "sub-003"]
            modules = ["preprocessing", "epochs"]

            for subject_id in subjects:
                for module_name in modules:
                    result = ModuleResult(
                        success=True,
                        module_name=module_name,
                        execution_time_seconds=1.0,
                    )
                    manager.save(subject_id, module_name, result)

            # Verify each subject has their own checkpoints
            for subject_id in subjects:
                completed = manager.get_completed_modules(subject_id)
                assert len(completed) == 2
                assert "preprocessing" in completed
                assert "epochs" in completed


class TestReproducibilityHash:
    """Test reproducibility hash generation."""

    def test_basic_hash(self):
        """Test basic hash generation."""
        config = {"param1": 1, "param2": "value"}
        hash1 = generate_reproducibility_hash(config, "1.0.0")

        assert hash1 is not None
        assert len(hash1) == 16  # Truncated to 16 chars

    def test_same_input_same_hash(self):
        """Test that same inputs produce same hash."""
        config = {"a": 1, "b": 2}

        hash1 = generate_reproducibility_hash(config, "1.0.0")
        hash2 = generate_reproducibility_hash(config, "1.0.0")

        assert hash1 == hash2

    def test_different_config_different_hash(self):
        """Test that different configs produce different hashes."""
        config1 = {"a": 1}
        config2 = {"a": 2}

        hash1 = generate_reproducibility_hash(config1, "1.0.0")
        hash2 = generate_reproducibility_hash(config2, "1.0.0")

        assert hash1 != hash2

    def test_different_version_different_hash(self):
        """Test that different versions produce different hashes."""
        config = {"a": 1}

        hash1 = generate_reproducibility_hash(config, "1.0.0")
        hash2 = generate_reproducibility_hash(config, "2.0.0")

        assert hash1 != hash2

    def test_with_data_checksums(self):
        """Test hash with data checksums."""
        config = {"param": 1}
        checksums = {"file1.fif": "abc123", "file2.fif": "def456"}

        hash1 = generate_reproducibility_hash(config, "1.0.0", checksums)
        hash2 = generate_reproducibility_hash(config, "1.0.0")  # No checksums

        assert hash1 != hash2

    def test_order_independence(self):
        """Test that key order doesn't affect hash."""
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}

        hash1 = generate_reproducibility_hash(config1, "1.0.0")
        hash2 = generate_reproducibility_hash(config2, "1.0.0")

        assert hash1 == hash2

    def test_nested_config(self):
        """Test hash with nested configuration."""
        config = {
            "preprocessing": {"l_freq": 0.5, "h_freq": 40.0},
            "epochs": {"tmin": -0.5, "tmax": 1.0},
        }

        hash1 = generate_reproducibility_hash(config, "1.0.0")
        assert len(hash1) == 16
