"""Tests for project management module."""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from eegcpm.core.config import Config
from eegcpm.core.project import (
    create_project,
    save_project,
    load_project,
    scan_bids_directory,
    PipelineVersion,
    DerivativesManager,
    SubjectSplit,
    create_subject_split,
    AnalysisProject,
)


class TestCreateProject:
    """Test create_project function."""

    def test_create_project_basic(self):
        """Test basic project creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = create_project(
                name="Test Project",
                root_path=Path(tmpdir),
                description="A test project",
            )

            assert project.name == "Test Project"
            assert project.description == "A test project"
            assert project.root_path == Path(tmpdir)

    def test_create_project_with_sampling_rate(self):
        """Test project creation with sampling rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = create_project(
                name="Test",
                root_path=Path(tmpdir),
                sampling_rate_hz=500.0,
            )

            assert project.sampling_rate_hz == 500.0

    def test_create_project_creates_directories(self):
        """Test that project creation creates directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "my_project"
            project = create_project(name="Test", root_path=root)

            assert root.exists()
            assert (root / "raw").exists()
            assert (root / "derivatives").exists()
            assert (root / "derivatives" / "preprocessed").exists()
            assert (root / "derivatives" / "epochs").exists()
            assert (root / "derivatives" / "source").exists()
            assert (root / "derivatives" / "connectivity").exists()
            assert (root / "derivatives" / "features").exists()
            assert (root / "derivatives" / "predictions").exists()
            assert (root / "configs").exists()
            assert (root / "logs").exists()


class TestSaveLoadProject:
    """Test save_project and load_project functions."""

    def test_save_and_load_project(self):
        """Test saving and loading a project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = create_project(
                name="Test Project",
                root_path=Path(tmpdir),
                description="Test description",
                sampling_rate_hz=256.0,
            )

            # Save project
            save_path = save_project(project)
            assert save_path.exists()

            # Load project
            loaded = load_project(save_path)

            assert loaded.name == project.name
            assert loaded.description == project.description
            assert loaded.sampling_rate_hz == project.sampling_rate_hz

    def test_save_project_custom_path(self):
        """Test saving project to custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = create_project(name="Test", root_path=Path(tmpdir))

            custom_path = Path(tmpdir) / "custom" / "project.json"
            custom_path.parent.mkdir(parents=True)

            save_path = save_project(project, custom_path)

            assert save_path == custom_path
            assert custom_path.exists()


class TestScanBidsDirectory:
    """Test scan_bids_directory function."""

    def test_scan_empty_directory(self):
        """Test scanning empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = scan_bids_directory(Path(tmpdir))

            assert project.name == Path(tmpdir).name
            assert len(project.subjects) == 0

    def test_scan_bids_structure(self):
        """Test scanning BIDS-like directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create BIDS structure
            for sub_id in ["sub-001", "sub-002"]:
                for ses_id in ["ses-01", "ses-02"]:
                    eeg_dir = root / sub_id / ses_id / "eeg"
                    eeg_dir.mkdir(parents=True)

                    # Create dummy EEG file
                    eeg_file = eeg_dir / f"{sub_id}_{ses_id}_task-rest_eeg.fif"
                    eeg_file.touch()

            project = scan_bids_directory(root)

            assert len(project.subjects) == 2
            assert any(s.id == "sub-001" for s in project.subjects)
            assert any(s.id == "sub-002" for s in project.subjects)

            # Each subject should have 2 sessions
            for subject in project.subjects:
                assert len(subject.sessions) == 2

    def test_scan_bids_parses_task_and_run(self):
        """Test that scanning parses task and run from filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            eeg_dir = root / "sub-001" / "ses-01" / "eeg"
            eeg_dir.mkdir(parents=True)

            # Create file with task and run
            eeg_file = eeg_dir / "sub-001_ses-01_task-oddball_run-02_eeg.fif"
            eeg_file.touch()

            project = scan_bids_directory(root)

            run = project.subjects[0].sessions[0].runs[0]
            assert run.task_name == "oddball"
            assert run.id == "run-02"

    def test_scan_bids_no_session_level(self):
        """Test scanning when there's no session directory level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            eeg_dir = root / "sub-001" / "eeg"
            eeg_dir.mkdir(parents=True)

            eeg_file = eeg_dir / "sub-001_task-rest_eeg.fif"
            eeg_file.touch()

            project = scan_bids_directory(root)

            assert len(project.subjects) == 1
            assert len(project.subjects[0].sessions) == 1
            assert project.subjects[0].sessions[0].id == "ses-01"


class TestPipelineVersion:
    """Test PipelineVersion dataclass."""

    def test_pipeline_version_basic(self):
        """Test basic pipeline version creation."""
        version = PipelineVersion(
            name="baseline",
            description="Baseline preprocessing",
        )

        assert version.name == "baseline"
        assert version.description == "Baseline preprocessing"
        assert version.created_at != ""

    def test_pipeline_version_with_config(self):
        """Test pipeline version with config."""
        config = Config()
        version = PipelineVersion(
            name="strict",
            description="Strict filtering",
            config=config,
        )

        assert version.config is not None
        assert version.config_hash != ""
        assert len(version.config_hash) == 12

    def test_pipeline_version_hash_consistency(self):
        """Test that same config produces same hash."""
        config = Config()
        version1 = PipelineVersion(name="v1", config=config)
        version2 = PipelineVersion(name="v2", config=config)

        assert version1.config_hash == version2.config_hash

    def test_pipeline_version_to_dict(self):
        """Test conversion to dict."""
        version = PipelineVersion(
            name="test",
            description="Test version",
        )

        d = version.to_dict()

        assert d["name"] == "test"
        assert d["description"] == "Test version"
        assert "created_at" in d

    def test_pipeline_version_from_dict(self):
        """Test restoration from dict."""
        data = {
            "name": "restored",
            "description": "Restored version",
            "config_hash": "abc123",
            "created_at": "2024-01-01T00:00:00",
        }

        version = PipelineVersion.from_dict(data)

        assert version.name == "restored"
        assert version.description == "Restored version"
        assert version.config_hash == "abc123"


class TestDerivativesManager:
    """Test DerivativesManager class."""

    def test_derivatives_manager_init(self):
        """Test derivatives manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DerivativesManager(Path(tmpdir))

            assert manager.derivatives_dir.exists()
            assert len(manager.pipelines) == 0

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DerivativesManager(Path(tmpdir))
            config = Config()

            version = manager.create_pipeline(
                name="baseline",
                config=config,
                description="Baseline pipeline",
            )

            assert version.name == "baseline"
            assert "baseline" in manager.pipelines

            # Check directory structure
            pipeline_dir = manager.get_pipeline_dir("baseline")
            assert pipeline_dir.exists()
            assert (pipeline_dir / "config.yaml").exists()
            assert (pipeline_dir / "pipeline_info.json").exists()
            assert (pipeline_dir / "preprocessed").exists()
            assert (pipeline_dir / "epochs").exists()

    def test_create_duplicate_pipeline_raises(self):
        """Test that creating duplicate pipeline raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DerivativesManager(Path(tmpdir))
            config = Config()

            manager.create_pipeline("test", config)

            with pytest.raises(ValueError, match="already exists"):
                manager.create_pipeline("test", config)

    def test_list_pipelines(self):
        """Test listing pipelines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DerivativesManager(Path(tmpdir))
            config = Config()

            manager.create_pipeline("baseline", config)
            manager.create_pipeline("strict", config, description="Strict filtering")

            pipelines = manager.list_pipelines()

            assert len(pipelines) == 2
            names = [p["name"] for p in pipelines]
            assert "baseline" in names
            assert "strict" in names

    def test_get_subject_file(self):
        """Test getting subject file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DerivativesManager(Path(tmpdir))
            config = Config()
            manager.create_pipeline("baseline", config)

            path = manager.get_subject_file(
                "baseline",
                "preprocessed",
                "sub-001",
                "preprocessed_raw.fif",
            )

            expected = manager.get_pipeline_dir("baseline") / "preprocessed" / "sub-001_preprocessed_raw.fif"
            assert path == expected

    def test_delete_pipeline(self):
        """Test deleting a pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DerivativesManager(Path(tmpdir))
            config = Config()
            manager.create_pipeline("to_delete", config)

            assert "to_delete" in manager.pipelines

            manager.delete_pipeline("to_delete", confirm=True)

            assert "to_delete" not in manager.pipelines
            assert not manager.get_pipeline_dir("to_delete").exists()

    def test_delete_pipeline_requires_confirm(self):
        """Test that delete requires confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DerivativesManager(Path(tmpdir))
            config = Config()
            manager.create_pipeline("test", config)

            with pytest.raises(ValueError, match="confirm"):
                manager.delete_pipeline("test")


class TestSubjectSplit:
    """Test SubjectSplit dataclass."""

    def test_subject_split_basic(self):
        """Test basic subject split."""
        split = SubjectSplit(
            train=["sub-001", "sub-002", "sub-003"],
            validation=["sub-004"],
            test=["sub-005"],
        )

        assert len(split.train) == 3
        assert len(split.validation) == 1
        assert len(split.test) == 1
        assert split.n_total == 5

    def test_subject_split_summary(self):
        """Test split summary."""
        split = SubjectSplit(
            train=["sub-001", "sub-002", "sub-003", "sub-004"],
            validation=["sub-005"],
            test=["sub-006"],
            excluded=["sub-007"],
        )

        summary = split.summary()

        assert summary["n_train"] == 4
        assert summary["n_validation"] == 1
        assert summary["n_test"] == 1
        assert summary["n_excluded"] == 1
        assert summary["n_total"] == 7  # includes excluded
        assert abs(summary["train_ratio"] - 4/6) < 0.01

    def test_subject_split_save_load(self):
        """Test saving and loading split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split = SubjectSplit(
                train=["sub-001", "sub-002"],
                validation=["sub-003"],
                test=["sub-004"],
                random_state=42,
            )

            path = Path(tmpdir) / "split.json"
            split.save(path)

            loaded = SubjectSplit.load(path)

            assert loaded.train == split.train
            assert loaded.validation == split.validation
            assert loaded.test == split.test
            assert loaded.random_state == 42


class TestCreateSubjectSplit:
    """Test create_subject_split function."""

    def test_create_split_basic(self):
        """Test basic split creation."""
        subjects = [f"sub-{i:03d}" for i in range(100)]

        split = create_subject_split(
            subject_ids=subjects,
            test_ratio=0.2,
            val_ratio=0.1,
            random_state=42,
        )

        # Check rough proportions
        assert len(split.test) == 20
        assert len(split.validation) == 10
        assert len(split.train) == 70

        # Check no overlap
        all_subjects = set(split.train + split.validation + split.test)
        assert len(all_subjects) == 100

    def test_create_split_with_exclusions(self):
        """Test split with excluded subjects."""
        subjects = [f"sub-{i:03d}" for i in range(100)]
        excluded = [f"sub-{i:03d}" for i in range(10)]  # Exclude first 10

        split = create_subject_split(
            subject_ids=subjects,
            excluded_ids=excluded,
            test_ratio=0.2,
            val_ratio=0.1,
        )

        assert split.excluded == excluded
        assert len(split.train + split.validation + split.test) == 90

    def test_create_split_reproducibility(self):
        """Test that same random_state produces same split."""
        subjects = [f"sub-{i:03d}" for i in range(50)]

        split1 = create_subject_split(subjects, random_state=123)
        split2 = create_subject_split(subjects, random_state=123)

        assert split1.train == split2.train
        assert split1.validation == split2.validation
        assert split1.test == split2.test

    def test_create_split_different_seeds(self):
        """Test that different seeds produce different splits."""
        subjects = [f"sub-{i:03d}" for i in range(50)]

        split1 = create_subject_split(subjects, random_state=1)
        split2 = create_subject_split(subjects, random_state=2)

        # At least one partition should differ
        assert (split1.train != split2.train or
                split1.validation != split2.validation or
                split1.test != split2.test)


class TestAnalysisProject:
    """Test AnalysisProject class."""

    def test_analysis_project_init(self):
        """Test analysis project initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            analysis_dir = Path(tmpdir) / "analysis"
            source_dir.mkdir()

            project = AnalysisProject(
                source_dir=source_dir,
                analysis_dir=analysis_dir,
                project_name="Test Project",
            )

            assert project.project_name == "Test Project"
            assert project.analysis_dir.exists()
            assert project.derivatives is not None

    def test_discover_subjects(self):
        """Test subject discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            analysis_dir = Path(tmpdir) / "analysis"

            # Create some subject directories
            for i in range(5):
                (source_dir / f"sub-{i:03d}").mkdir(parents=True)

            project = AnalysisProject(source_dir, analysis_dir)
            subjects = project.discover_subjects()

            assert len(subjects) == 5
            assert "sub-000" in subjects

    def test_create_and_save_split(self):
        """Test creating and saving subject split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            analysis_dir = Path(tmpdir) / "analysis"

            for i in range(20):
                (source_dir / f"sub-{i:03d}").mkdir(parents=True)

            project = AnalysisProject(source_dir, analysis_dir)
            project.discover_subjects()

            split = project.create_split(random_state=42)

            assert split is not None
            assert len(split.train) + len(split.validation) + len(split.test) == 20

            # Verify it was saved
            loaded = project.get_subject_split()
            assert loaded is not None
            assert loaded.train == split.train

    def test_create_pipeline_via_project(self):
        """Test creating pipeline through analysis project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            analysis_dir = Path(tmpdir) / "analysis"
            source_dir.mkdir()

            project = AnalysisProject(source_dir, analysis_dir)
            config = Config()

            version = project.create_pipeline(
                "baseline",
                config,
                description="Baseline pipeline",
            )

            assert version.name == "baseline"
            assert "baseline" in project.derivatives.pipelines
