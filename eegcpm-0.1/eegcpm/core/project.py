"""
Project and derivatives management for EEGCPM.

Supports:
- BIDS-style data organization (read-only source)
- Multiple preprocessing pipelines with versioning
- Derivatives structure for processed data and metrics
- Train/validation/test splits
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import yaml

from eegcpm.core.models import Project, Subject, Session, Run
from eegcpm.core.config import Config


def create_project(
    name: str,
    root_path: Path,
    description: str = "",
    sampling_rate_hz: Optional[float] = None,
) -> Project:
    """
    Create a new EEGCPM project.

    Args:
        name: Project name
        root_path: Root directory for project data
        description: Project description
        sampling_rate_hz: Common sampling rate (optional)

    Returns:
        New Project instance
    """
    root_path = Path(root_path)

    # Create directory structure
    dirs = [
        root_path,
        root_path / "raw",
        root_path / "derivatives",
        root_path / "derivatives" / "preprocessed",
        root_path / "derivatives" / "epochs",
        root_path / "derivatives" / "source",
        root_path / "derivatives" / "connectivity",
        root_path / "derivatives" / "features",
        root_path / "derivatives" / "predictions",
        root_path / "configs",
        root_path / "logs",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    project = Project(
        name=name,
        description=description,
        root_path=root_path,
        sampling_rate_hz=sampling_rate_hz,
    )

    return project


def save_project(project: Project, path: Optional[Path] = None) -> Path:
    """
    Save project to JSON file.

    Args:
        project: Project to save
        path: Output path (defaults to project_root/project.json)

    Returns:
        Path to saved file
    """
    if path is None:
        path = project.root_path / "project.json"

    with open(path, "w") as f:
        json.dump(project.model_dump(mode="json"), f, indent=2, default=str)

    return path


def load_project(path: Path) -> Project:
    """
    Load project from JSON file.

    Args:
        path: Path to project.json

    Returns:
        Project instance
    """
    with open(path) as f:
        data = json.load(f)

    return Project(**data)


def scan_bids_directory(root_path: Path) -> Project:
    """
    Scan a BIDS-like directory structure and create Project.

    Expected structure:
        root/
            sub-001/
                ses-01/
                    eeg/
                        sub-001_ses-01_task-rest_eeg.fif
                        sub-001_ses-01_task-rest_events.tsv

    Args:
        root_path: Root of BIDS directory

    Returns:
        Project with discovered subjects/sessions/runs
    """
    root_path = Path(root_path)
    project = Project(
        name=root_path.name,
        root_path=root_path,
    )

    # Find subject directories
    for sub_dir in sorted(root_path.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        subject = Subject(id=sub_dir.name)

        # Find session directories (or assume single session)
        ses_dirs = list(sub_dir.glob("ses-*"))
        if not ses_dirs:
            ses_dirs = [sub_dir]  # No session level

        for ses_dir in sorted(ses_dirs):
            session_id = ses_dir.name if ses_dir.name.startswith("ses-") else "ses-01"
            session = Session(id=session_id, subject_id=subject.id)

            # Find EEG files
            eeg_dir = ses_dir / "eeg" if (ses_dir / "eeg").exists() else ses_dir

            for eeg_file in sorted(eeg_dir.glob("*.fif")) + \
                           sorted(eeg_dir.glob("*.edf")) + \
                           sorted(eeg_dir.glob("*.bdf")) + \
                           sorted(eeg_dir.glob("*.set")):

                # Parse filename for task and run
                parts = eeg_file.stem.split("_")
                task_name = "unknown"
                run_id = "run-01"

                for part in parts:
                    if part.startswith("task-"):
                        task_name = part.replace("task-", "")
                    elif part.startswith("run-"):
                        run_id = part

                run = Run(
                    id=run_id,
                    task_name=task_name,
                    eeg_file=eeg_file,
                )
                session.runs.append(run)

            if session.runs:
                subject.sessions.append(session)

        if subject.sessions:
            project.subjects.append(subject)

    return project


# =============================================================================
# Pipeline Versioning
# =============================================================================

@dataclass
class PipelineVersion:
    """Represents a specific preprocessing pipeline configuration."""

    name: str  # e.g., "baseline", "strict_filter", "ica_20"
    description: str = ""
    config: Optional[Config] = None
    config_hash: str = ""  # Hash of config for comparison
    created_at: str = ""
    parent_version: Optional[str] = None  # For tracking lineage

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.config and not self.config_hash:
            self.config_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of config for comparison."""
        if self.config is None:
            return ""
        config_str = json.dumps(self.config.model_dump(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[Config] = None) -> "PipelineVersion":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            config=config,
            config_hash=data.get("config_hash", ""),
            created_at=data.get("created_at", ""),
            parent_version=data.get("parent_version"),
        )


# =============================================================================
# Derivatives Manager
# =============================================================================

@dataclass
class DerivativesManager:
    """
    Manages derivatives (processed data) for a project.

    Structure:
    derivatives/
    ├── pipeline-baseline/
    │   ├── config.yaml
    │   ├── pipeline_info.json
    │   ├── qc/
    │   │   ├── sub-001_qc.json
    │   │   └── qc_summary.csv
    │   ├── preprocessed/
    │   │   ├── sub-001_preprocessed_raw.fif
    │   │   └── sub-001_ica.fif
    │   ├── epochs/
    │   │   └── sub-001_epo.fif
    │   ├── source/
    │   │   └── sub-001_stc-lh.stc
    │   ├── connectivity/
    │   │   └── sub-001_connectivity.npz
    │   └── features/
    │       └── sub-001_features.npz
    ├── pipeline-strict/
    │   └── ...
    └── models/
        ├── cpm_baseline_ridge/
        │   ├── model.pkl
        │   ├── train_subjects.json
        │   └── results.json
        └── ...
    """

    project_root: Path
    derivatives_dir: Path = field(init=False)
    pipelines: Dict[str, PipelineVersion] = field(default_factory=dict)

    def __post_init__(self):
        self.derivatives_dir = self.project_root / "derivatives"
        self.derivatives_dir.mkdir(parents=True, exist_ok=True)
        self._load_pipelines()

    def _load_pipelines(self):
        """Load existing pipeline versions from disk."""
        for pipeline_dir in self.derivatives_dir.glob("pipeline-*"):
            if pipeline_dir.is_dir():
                info_file = pipeline_dir / "pipeline_info.json"
                config_file = pipeline_dir / "config.yaml"

                if info_file.exists():
                    with open(info_file) as f:
                        info = json.load(f)

                    config = None
                    if config_file.exists():
                        config = Config.from_yaml(config_file)

                    self.pipelines[info["name"]] = PipelineVersion.from_dict(info, config)

    def create_pipeline(
        self,
        name: str,
        config: Config,
        description: str = "",
        parent: Optional[str] = None,
    ) -> PipelineVersion:
        """
        Create a new pipeline version.

        Args:
            name: Pipeline name (e.g., "baseline", "strict_filter")
            config: Configuration for this pipeline
            description: Human-readable description
            parent: Parent pipeline name (for tracking lineage)

        Returns:
            PipelineVersion object
        """
        if name in self.pipelines:
            raise ValueError(f"Pipeline '{name}' already exists")

        version = PipelineVersion(
            name=name,
            description=description,
            config=config,
            parent_version=parent,
        )

        # Create directory structure
        pipeline_dir = self.get_pipeline_dir(name)
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["qc", "preprocessed", "epochs", "source", "connectivity", "features"]:
            (pipeline_dir / subdir).mkdir(exist_ok=True)

        # Save config
        config.to_yaml(pipeline_dir / "config.yaml")

        # Save pipeline info
        with open(pipeline_dir / "pipeline_info.json", "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        self.pipelines[name] = version
        return version

    def get_pipeline_dir(self, name: str) -> Path:
        """Get directory for a pipeline."""
        return self.derivatives_dir / f"pipeline-{name}"

    def get_output_dir(self, pipeline_name: str, module_name: str) -> Path:
        """Get output directory for a specific module within a pipeline."""
        pipeline_dir = self.get_pipeline_dir(pipeline_name)
        return pipeline_dir / module_name

    def get_subject_file(
        self,
        pipeline_name: str,
        module_name: str,
        subject_id: str,
        suffix: str,
    ) -> Path:
        """Get path for a subject's output file."""
        output_dir = self.get_output_dir(pipeline_name, module_name)
        return output_dir / f"{subject_id}_{suffix}"

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipeline versions with summary info."""
        return [
            {
                **v.to_dict(),
                "directory": str(self.get_pipeline_dir(v.name)),
            }
            for v in self.pipelines.values()
        ]

    def compare_pipelines(
        self,
        pipeline_a: str,
        pipeline_b: str,
    ) -> Dict[str, Any]:
        """
        Compare two pipeline configurations.

        Returns dict with differences.
        """
        if pipeline_a not in self.pipelines or pipeline_b not in self.pipelines:
            raise ValueError("Pipeline not found")

        config_a = self.pipelines[pipeline_a].config
        config_b = self.pipelines[pipeline_b].config

        if config_a is None or config_b is None:
            return {"error": "Config not available"}

        dict_a = config_a.model_dump()
        dict_b = config_b.model_dump()

        def find_diffs(d1, d2, path=""):
            diffs = []
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in d1:
                    diffs.append({"path": new_path, "a": None, "b": d2[key]})
                elif key not in d2:
                    diffs.append({"path": new_path, "a": d1[key], "b": None})
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    diffs.extend(find_diffs(d1[key], d2[key], new_path))
                elif d1[key] != d2[key]:
                    diffs.append({"path": new_path, "a": d1[key], "b": d2[key]})
            return diffs

        return {
            "pipeline_a": pipeline_a,
            "pipeline_b": pipeline_b,
            "hash_a": self.pipelines[pipeline_a].config_hash,
            "hash_b": self.pipelines[pipeline_b].config_hash,
            "differences": find_diffs(dict_a, dict_b),
        }

    def delete_pipeline(self, name: str, confirm: bool = False):
        """Delete a pipeline and all its data."""
        if not confirm:
            raise ValueError("Must confirm deletion with confirm=True")

        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not found")

        pipeline_dir = self.get_pipeline_dir(name)
        if pipeline_dir.exists():
            shutil.rmtree(pipeline_dir)

        del self.pipelines[name]


# =============================================================================
# Subject Splits (Train/Validation/Test)
# =============================================================================

@dataclass
class SubjectSplit:
    """Train/validation/test split for subjects."""

    train: List[str] = field(default_factory=list)
    validation: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)  # QC failures
    random_state: int = 42
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def n_total(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)

    @property
    def n_usable(self) -> int:
        return self.n_total

    def summary(self) -> Dict[str, Any]:
        return {
            "n_train": len(self.train),
            "n_validation": len(self.validation),
            "n_test": len(self.test),
            "n_excluded": len(self.excluded),
            "n_total": self.n_total + len(self.excluded),
            "train_ratio": len(self.train) / self.n_total if self.n_total > 0 else 0,
            "val_ratio": len(self.validation) / self.n_total if self.n_total > 0 else 0,
            "test_ratio": len(self.test) / self.n_total if self.n_total > 0 else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "excluded": self.excluded,
            "random_state": self.random_state,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubjectSplit":
        return cls(
            train=data.get("train", []),
            validation=data.get("validation", []),
            test=data.get("test", []),
            excluded=data.get("excluded", []),
            random_state=data.get("random_state", 42),
            created_at=data.get("created_at", ""),
        )

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SubjectSplit":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def create_subject_split(
    subject_ids: List[str],
    excluded_ids: Optional[List[str]] = None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42,
    stratify_by: Optional[Dict[str, Any]] = None,
) -> SubjectSplit:
    """
    Create train/validation/test split.

    Args:
        subject_ids: All subject IDs
        excluded_ids: Subject IDs to exclude (e.g., QC failures)
        test_ratio: Proportion for test set (held out)
        val_ratio: Proportion for validation set
        random_state: Random seed for reproducibility
        stratify_by: Optional dict of subject_id -> group for stratification

    Returns:
        SubjectSplit object
    """
    excluded = excluded_ids or []
    usable = [s for s in subject_ids if s not in excluded]

    np.random.seed(random_state)
    indices = np.random.permutation(len(usable))

    n_test = int(len(usable) * test_ratio)
    n_val = int(len(usable) * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    return SubjectSplit(
        train=[usable[i] for i in train_idx],
        validation=[usable[i] for i in val_idx],
        test=[usable[i] for i in test_idx],
        excluded=excluded,
        random_state=random_state,
    )


# =============================================================================
# Analysis Project (High-level wrapper)
# =============================================================================

class AnalysisProject:
    """
    High-level project management combining BIDS source data with derivatives.

    Workflow:
        1. Create project pointing to BIDS source and analysis directory
        2. Run QC to identify bad subjects
        3. Create train/val/test split (excluding bad subjects)
        4. Create preprocessing pipelines with different configs
        5. Run pipelines on subjects
        6. Extract features and train models

    Usage:
        project = AnalysisProject(
            source_dir="/data/bids/my_study",
            analysis_dir="/analysis/my_study",
        )

        # Discover subjects from BIDS
        subjects = project.discover_subjects()

        # Create pipelines with different configs
        project.create_pipeline("baseline", baseline_config)
        project.create_pipeline("strict", strict_config, description="Stricter filtering")

        # After running QC, create split
        split = project.create_split(excluded_ids=qc_excluded)
        project.save_subject_split(split)

        # Compare pipeline results
        comparison = project.derivatives.compare_pipelines("baseline", "strict")
    """

    def __init__(
        self,
        source_dir: Path,
        analysis_dir: Path,
        project_name: Optional[str] = None,
    ):
        """
        Initialize analysis project.

        Args:
            source_dir: Path to BIDS source data (read-only)
            analysis_dir: Path for derivatives and analysis outputs
            project_name: Optional project name
        """
        self.source_dir = Path(source_dir)
        self.analysis_dir = Path(analysis_dir)
        self.project_name = project_name or self.source_dir.name

        # Create analysis directory
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Initialize derivatives manager
        self.derivatives = DerivativesManager(self.analysis_dir)

        # Load or create project metadata
        self.metadata_file = self.analysis_dir / "project.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load project metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "project_name": self.project_name,
                "source_dir": str(self.source_dir),
                "created_at": datetime.now().isoformat(),
                "subjects": [],
            }
            self._save_metadata()

    def _save_metadata(self):
        """Save project metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def discover_subjects(self) -> List[str]:
        """Discover subjects from BIDS source directory."""
        subjects = []
        for sub_dir in sorted(self.source_dir.glob("sub-*")):
            if sub_dir.is_dir():
                subjects.append(sub_dir.name)

        self.metadata["subjects"] = subjects
        self._save_metadata()
        return subjects

    def create_pipeline(
        self,
        name: str,
        config: Config,
        description: str = "",
    ) -> PipelineVersion:
        """Create a new preprocessing pipeline."""
        return self.derivatives.create_pipeline(name, config, description)

    def get_subject_split(self) -> Optional[SubjectSplit]:
        """Load saved subject split."""
        split_file = self.analysis_dir / "subject_split.json"
        if split_file.exists():
            return SubjectSplit.load(split_file)
        return None

    def save_subject_split(self, split: SubjectSplit):
        """Save subject split."""
        split.save(self.analysis_dir / "subject_split.json")

    def create_split(
        self,
        excluded_ids: Optional[List[str]] = None,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        random_state: int = 42,
    ) -> SubjectSplit:
        """Create and save a subject split."""
        subjects = self.metadata.get("subjects", [])
        if not subjects:
            subjects = self.discover_subjects()

        split = create_subject_split(
            subject_ids=subjects,
            excluded_ids=excluded_ids,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            random_state=random_state,
        )
        self.save_subject_split(split)
        return split
