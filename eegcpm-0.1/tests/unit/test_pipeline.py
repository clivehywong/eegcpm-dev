"""Tests for pipeline module."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from eegcpm.pipeline.base import ModuleResult, BaseModule, RawDataModule, EpochsModule


class TestModuleResult:
    """Test ModuleResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful module result."""
        result = ModuleResult(
            success=True,
            module_name="test_module",
            execution_time_seconds=1.5,
            outputs={"data": "test"},
        )
        assert result.success
        assert result.module_name == "test_module"
        assert result.execution_time_seconds == 1.5
        assert result.outputs["data"] == "test"
        assert result.errors == []
        assert result.warnings == []

    def test_failed_result(self):
        """Test creating a failed module result."""
        result = ModuleResult(
            success=False,
            module_name="failing_module",
            execution_time_seconds=0.5,
            errors=["Something went wrong", "Another error"],
        )
        assert not result.success
        assert len(result.errors) == 2
        assert "Something went wrong" in result.errors

    def test_result_with_warnings(self):
        """Test result with warnings."""
        result = ModuleResult(
            success=True,
            module_name="warning_module",
            execution_time_seconds=2.0,
            warnings=["Low data quality", "Missing channels"],
        )
        assert result.success
        assert len(result.warnings) == 2

    def test_result_with_output_files(self):
        """Test result with output files."""
        files = [Path("/tmp/output1.fif"), Path("/tmp/output2.npz")]
        result = ModuleResult(
            success=True,
            module_name="output_module",
            execution_time_seconds=3.0,
            output_files=files,
        )
        assert len(result.output_files) == 2
        assert result.output_files[0] == Path("/tmp/output1.fif")

    def test_result_metadata(self):
        """Test result with metadata."""
        result = ModuleResult(
            success=True,
            module_name="meta_module",
            execution_time_seconds=1.0,
            metadata={
                "n_epochs": 100,
                "sampling_rate": 500,
                "channels_used": ["Fp1", "Fp2", "Fz"],
            },
        )
        assert result.metadata["n_epochs"] == 100
        assert len(result.metadata["channels_used"]) == 3

    def test_result_timestamp(self):
        """Test that timestamp is set automatically."""
        before = datetime.now()
        result = ModuleResult(
            success=True,
            module_name="time_module",
            execution_time_seconds=0.1,
        )
        after = datetime.now()

        assert before <= result.timestamp <= after

    def test_result_repr(self):
        """Test result string representation."""
        success_result = ModuleResult(
            success=True,
            module_name="test",
            execution_time_seconds=1.23,
        )
        assert "SUCCESS" in repr(success_result)
        assert "test" in repr(success_result)
        assert "1.23" in repr(success_result)

        fail_result = ModuleResult(
            success=False,
            module_name="test",
            execution_time_seconds=0.5,
        )
        assert "FAILED" in repr(fail_result)


class ConcreteModule(BaseModule):
    """Concrete implementation for testing BaseModule."""

    name = "concrete_module"
    version = "1.0.0"
    description = "A concrete test module"

    def validate_input(self, data: Any) -> bool:
        if data is None:
            raise ValueError("Data cannot be None")
        return True

    def process(self, data: Any, **kwargs) -> ModuleResult:
        return ModuleResult(
            success=True,
            module_name=self.name,
            execution_time_seconds=0.1,
            outputs={"data": data, "processed": True},
        )

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "data": "Processed data",
            "processed": "Boolean flag",
        }


class TestBaseModule:
    """Test BaseModule abstract class."""

    def test_module_initialization(self):
        """Test module initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = ConcreteModule(
                config={"param1": "value1"},
                output_dir=Path(tmpdir) / "output",
            )
            assert module.config["param1"] == "value1"
            assert module.output_dir.exists()

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output" / "dir"
            module = ConcreteModule(config={}, output_dir=output_dir)
            assert output_dir.exists()

    def test_validate_input(self):
        """Test input validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = ConcreteModule(config={}, output_dir=Path(tmpdir))

            # Valid input
            assert module.validate_input([1, 2, 3])

            # Invalid input
            with pytest.raises(ValueError, match="cannot be None"):
                module.validate_input(None)

    def test_process(self):
        """Test processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = ConcreteModule(config={}, output_dir=Path(tmpdir))
            result = module.process([1, 2, 3])

            assert result.success
            assert result.module_name == "concrete_module"
            assert result.outputs["data"] == [1, 2, 3]
            assert result.outputs["processed"]

    def test_get_output_spec(self):
        """Test output specification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = ConcreteModule(config={}, output_dir=Path(tmpdir))
            spec = module.get_output_spec()

            assert "data" in spec
            assert "processed" in spec

    def test_get_checkpoint_data(self):
        """Test checkpoint data generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = ConcreteModule(
                config={"key": "value"},
                output_dir=Path(tmpdir) / "output",
            )
            checkpoint = module.get_checkpoint_data()

            assert checkpoint["module_name"] == "concrete_module"
            assert checkpoint["version"] == "1.0.0"
            assert checkpoint["config"]["key"] == "value"
            assert "output" in checkpoint["output_dir"]

    def test_from_checkpoint(self):
        """Test restoring module from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = ConcreteModule(
                config={"param": 42},
                output_dir=Path(tmpdir) / "output",
            )
            checkpoint_data = original.get_checkpoint_data()

            restored = ConcreteModule.from_checkpoint(checkpoint_data)

            assert restored.config["param"] == 42
            assert str(restored.output_dir) == checkpoint_data["output_dir"]

    def test_cleanup(self):
        """Test cleanup method (default is no-op)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module = ConcreteModule(config={}, output_dir=Path(tmpdir))
            # Should not raise
            module.cleanup()

    def test_module_attributes(self):
        """Test module class attributes."""
        assert ConcreteModule.name == "concrete_module"
        assert ConcreteModule.version == "1.0.0"
        assert ConcreteModule.description == "A concrete test module"


class TestRawDataModule:
    """Test RawDataModule base class."""

    def test_validate_input_with_non_raw(self):
        """Test that non-Raw data raises ValueError."""
        # Create a simple concrete implementation
        class TestRawModule(RawDataModule):
            name = "test_raw"

            def process(self, data, **kwargs):
                return ModuleResult(
                    success=True,
                    module_name=self.name,
                    execution_time_seconds=0.1,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            module = TestRawModule(config={}, output_dir=Path(tmpdir))

            with pytest.raises(ValueError, match="Expected mne.io.Raw"):
                module.validate_input([1, 2, 3])

            with pytest.raises(ValueError, match="Expected mne.io.Raw"):
                module.validate_input(np.array([1, 2, 3]))


class TestEpochsModule:
    """Test EpochsModule base class."""

    def test_validate_input_with_non_epochs(self):
        """Test that non-Epochs data raises ValueError."""
        class TestEpochsModule(EpochsModule):
            name = "test_epochs"

            def process(self, data, **kwargs):
                return ModuleResult(
                    success=True,
                    module_name=self.name,
                    execution_time_seconds=0.1,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            module = TestEpochsModule(config={}, output_dir=Path(tmpdir))

            with pytest.raises(ValueError, match="Expected mne.Epochs"):
                module.validate_input([1, 2, 3])
