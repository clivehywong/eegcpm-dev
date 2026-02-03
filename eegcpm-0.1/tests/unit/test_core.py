"""Tests for core module."""

import pytest
from pathlib import Path
import tempfile

from eegcpm.core.models import Project, Subject, Session, Run, Event
from eegcpm.core.config import Config, PreprocessingConfig, RejectionConfig, EpochsConfig
from eegcpm.core.validation import validate_config


class TestModels:
    """Test data models."""

    def test_event_creation(self):
        """Test Event model."""
        event = Event(
            onset_seconds=1.5,
            duration_seconds=0.0,
            event_type="stimulus",
            event_value="target",
        )
        assert event.onset_seconds == 1.5
        assert event.event_type == "stimulus"

    def test_run_creation(self):
        """Test Run model."""
        with tempfile.NamedTemporaryFile(suffix=".fif") as f:
            run = Run(
                id="run-01",
                task_name="oddball",
                eeg_file=Path(f.name),
            )
            assert run.id == "run-01"
            assert run.task_name == "oddball"

    def test_session_creation(self):
        """Test Session model."""
        session = Session(
            id="ses-01",
            subject_id="sub-001",
        )
        assert session.id == "ses-01"
        assert len(session.runs) == 0

    def test_subject_creation(self):
        """Test Subject model."""
        subject = Subject(
            id="sub-001",
            behavioral_scores={"score": 42.0},
        )
        assert subject.id == "sub-001"
        assert subject.behavioral_scores["score"] == 42.0

    def test_project_creation(self):
        """Test Project model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(
                name="Test Project",
                root_path=Path(tmpdir),
                sampling_rate_hz=500.0,
            )
            assert project.name == "Test Project"
            assert project.sampling_rate_hz == 500.0

    def test_project_add_subject(self):
        """Test adding subjects to project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Project(
                name="Test",
                root_path=Path(tmpdir),
            )
            subject = Subject(id="sub-001")
            project.add_subject(subject)

            assert len(project.subjects) == 1
            assert project.get_subject("sub-001") is not None

            # Duplicate should raise
            with pytest.raises(ValueError):
                project.add_subject(subject)


class TestConfig:
    """Test configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.preprocessing.l_freq == 0.5
        assert config.preprocessing.h_freq == 40.0
        assert config.epochs.tmin == -0.5
        assert config.source.method == "sLORETA"

    def test_preprocessing_config(self):
        """Test preprocessing config."""
        config = PreprocessingConfig(
            l_freq=1.0,
            h_freq=30.0,
            ica_method="picard",
        )
        assert config.l_freq == 1.0
        assert config.ica_method == "picard"


class TestValidation:
    """Test validation functions."""

    def test_validate_valid_config(self):
        """Test validation of valid config."""
        config = Config()
        result = validate_config(config)
        assert result.is_valid

    def test_validate_invalid_filter(self):
        """Test validation catches invalid filter settings."""
        config = Config()
        config.preprocessing.l_freq = 50.0  # Higher than h_freq
        config.preprocessing.h_freq = 40.0

        result = validate_config(config)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_invalid_epochs(self):
        """Test validation catches invalid epoch settings."""
        config = Config()
        config.epochs.tmin = 1.0  # Greater than tmax
        config.epochs.tmax = 0.5

        result = validate_config(config)
        assert not result.is_valid


class TestRejectionConfig:
    """Test rejection configuration."""

    def test_default_rejection_config(self):
        """Test default rejection config."""
        rejection = RejectionConfig()
        assert rejection.reject is None
        assert rejection.flat is None
        assert rejection.reject_by_annotation is True
        assert rejection.use_autoreject is False
        assert rejection.strategy == "threshold"

    def test_rejection_with_thresholds(self):
        """Test rejection config with thresholds."""
        rejection = RejectionConfig(
            reject={"eeg": 150e-6, "eog": 250e-6},
            flat={"eeg": 1e-6},
        )
        assert rejection.reject["eeg"] == 150e-6
        assert rejection.flat["eeg"] == 1e-6

    def test_rejection_lenient_preset(self):
        """Test lenient rejection preset."""
        rejection = RejectionConfig.lenient()
        assert rejection.reject["eeg"] == 200e-6
        assert rejection.flat["eeg"] == 0.5e-6
        assert rejection.strategy == "threshold"

    def test_rejection_strict_preset(self):
        """Test strict rejection preset."""
        rejection = RejectionConfig.strict()
        assert rejection.reject["eeg"] == 100e-6
        assert rejection.flat["eeg"] == 1e-6

    def test_rejection_adaptive_preset(self):
        """Test adaptive rejection preset."""
        rejection = RejectionConfig.adaptive()
        assert rejection.use_autoreject is True
        assert rejection.strategy == "autoreject"

    def test_epochs_config_with_rejection(self):
        """Test epochs config includes rejection."""
        epochs = EpochsConfig(
            tmin=-0.2,
            tmax=0.8,
            rejection=RejectionConfig.strict(),
        )
        assert epochs.rejection.reject["eeg"] == 100e-6
        assert epochs.tmin == -0.2

    def test_rejection_time_window(self):
        """Test rejection time window config."""
        rejection = RejectionConfig(
            reject={"eeg": 150e-6},
            reject_tmin=0.0,
            reject_tmax=0.5,
        )
        assert rejection.reject_tmin == 0.0
        assert rejection.reject_tmax == 0.5
