"""
Configuration and data validation utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from eegcpm.core.config import Config
from eegcpm.core.models import Project


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


def validate_config(config: Config) -> ValidationResult:
    """
    Validate configuration before running pipeline.

    Checks:
    - Parameter ranges are valid
    - Dependencies between settings are satisfied
    - Required fields are present
    """
    result = ValidationResult()

    # Preprocessing validation
    if config.preprocessing.l_freq is not None and config.preprocessing.h_freq is not None:
        if config.preprocessing.l_freq >= config.preprocessing.h_freq:
            result.add_error(
                f"l_freq ({config.preprocessing.l_freq}) must be less than "
                f"h_freq ({config.preprocessing.h_freq})"
            )

    if config.preprocessing.ica_method not in ["infomax", "extended_infomax", "picard", "fastica"]:
        result.add_error(f"Unknown ICA method: {config.preprocessing.ica_method}")

    # Epochs validation
    if config.epochs.tmin >= config.epochs.tmax:
        result.add_error(
            f"epochs.tmin ({config.epochs.tmin}) must be less than "
            f"epochs.tmax ({config.epochs.tmax})"
        )

    if config.epochs.baseline is not None:
        bmin, bmax = config.epochs.baseline
        if bmin < config.epochs.tmin or bmax > config.epochs.tmax:
            result.add_warning(
                f"Baseline ({bmin}, {bmax}) extends beyond epoch window "
                f"({config.epochs.tmin}, {config.epochs.tmax})"
            )

    # Source validation
    valid_source_methods = ["dSPM", "sLORETA", "eLORETA", "MNE", "LCMV", "DICS"]
    if config.source.method not in valid_source_methods:
        result.add_error(f"Unknown source method: {config.source.method}")

    # Connectivity validation
    valid_conn_methods = [
        "correlation", "partial_correlation", "spearman",
        "plv", "pli", "wpli", "coherence", "imcoh",
        "mi", "transfer_entropy"
    ]
    for method in config.connectivity.methods:
        if method not in valid_conn_methods:
            result.add_warning(f"Unknown connectivity method: {method}")

    # Prediction validation
    valid_pred_types = ["within_subject", "between_subject", "between_group"]
    if config.prediction.prediction_type not in valid_pred_types:
        result.add_error(f"Unknown prediction type: {config.prediction.prediction_type}")

    valid_cv_strategies = ["kfold", "leave_one_out", "stratified_kfold", "group_kfold"]
    if config.prediction.cv_strategy not in valid_cv_strategies:
        result.add_error(f"Unknown CV strategy: {config.prediction.cv_strategy}")

    return result


def validate_project(project: Project) -> ValidationResult:
    """
    Validate project structure and data files.

    Checks:
    - Root path exists
    - EEG files exist and are readable
    - Subject/session structure is consistent
    """
    result = ValidationResult()

    # Check root path
    if not project.root_path.exists():
        result.add_error(f"Project root path does not exist: {project.root_path}")
        return result  # Can't continue without root

    # Check subjects
    if len(project.subjects) == 0:
        result.add_warning("Project has no subjects defined")

    for subject in project.subjects:
        if len(subject.sessions) == 0:
            result.add_warning(f"Subject {subject.id} has no sessions")

        for session in subject.sessions:
            for run in session.runs:
                if not run.eeg_file.exists():
                    result.add_error(
                        f"EEG file not found: {run.eeg_file} "
                        f"(subject: {subject.id}, session: {session.id}, run: {run.id})"
                    )

    return result


def validate_before_pipeline(project: Project, config: Config) -> ValidationResult:
    """
    Combined validation before running pipeline.
    """
    result = ValidationResult()

    # Validate config
    config_result = validate_config(config)
    result.errors.extend(config_result.errors)
    result.warnings.extend(config_result.warnings)

    # Validate project
    project_result = validate_project(project)
    result.errors.extend(project_result.errors)
    result.warnings.extend(project_result.warnings)

    return result
