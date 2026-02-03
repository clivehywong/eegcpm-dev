"""
Data models for EEGCPM using Pydantic v2.

BIDS-inspired structure for EEG project organization.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Event(BaseModel):
    """Single event/trigger in EEG recording."""

    onset_seconds: float = Field(..., description="Event onset time in seconds")
    duration_seconds: float = Field(default=0.0, description="Event duration")
    event_type: str = Field(..., description="Event type/category")
    event_value: Optional[str] = Field(default=None, description="Event value/code")

    class Config:
        json_schema_extra = {
            "example": {
                "onset_seconds": 1.5,
                "duration_seconds": 0.0,
                "event_type": "stimulus",
                "event_value": "target"
            }
        }


class Run(BaseModel):
    """Single recording run within a session."""

    id: str = Field(..., description="Run identifier (e.g., run-01)")
    task_name: str = Field(..., description="Task name (e.g., rest, oddball)")
    duration_seconds: Optional[float] = Field(default=None, description="Recording duration")
    eeg_file: Path = Field(..., description="Path to EEG data file")
    events_file: Optional[Path] = Field(default=None, description="Path to events file")
    events: List[Event] = Field(default_factory=list, description="List of events")
    processing_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("eeg_file", mode="before")
    @classmethod
    def validate_eeg_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class Session(BaseModel):
    """Recording session for a subject."""

    id: str = Field(..., description="Session identifier (e.g., ses-01)")
    subject_id: str = Field(..., description="Parent subject ID")
    date: Optional[datetime] = Field(default=None, description="Session date")
    runs: List[Run] = Field(default_factory=list, description="List of runs")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Subject(BaseModel):
    """Individual subject/participant."""

    id: str = Field(..., description="Subject identifier (e.g., sub-001)")
    sessions: List[Session] = Field(default_factory=list)
    demographics: Dict[str, Any] = Field(default_factory=dict)
    behavioral_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Behavioral/clinical scores for prediction targets"
    )


class Project(BaseModel):
    """Top-level project container."""

    name: str = Field(..., description="Project name")
    description: str = Field(default="", description="Project description")
    root_path: Path = Field(..., description="Root directory for project data")
    sampling_rate_hz: Optional[float] = Field(default=None, description="Common sampling rate")
    subjects: List[Subject] = Field(default_factory=list)

    # Processing configuration
    preprocessing_config: Dict[str, Any] = Field(default_factory=dict)
    epochs_config: Dict[str, Any] = Field(default_factory=dict)
    source_config: Dict[str, Any] = Field(default_factory=dict)
    connectivity_config: Dict[str, Any] = Field(default_factory=dict)
    prediction_config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("root_path", mode="before")
    @classmethod
    def validate_root_path(cls, v):
        return Path(v) if isinstance(v, str) else v

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        """Get subject by ID."""
        for subject in self.subjects:
            if subject.id == subject_id:
                return subject
        return None

    def add_subject(self, subject: Subject) -> None:
        """Add a subject to the project."""
        if self.get_subject(subject.id) is not None:
            raise ValueError(f"Subject {subject.id} already exists")
        self.subjects.append(subject)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "HBN_CPM_Study",
                "description": "Connectome predictive modeling on HBN dataset",
                "root_path": "/data/hbn/",
                "sampling_rate_hz": 500.0
            }
        }


# Connectivity-specific models

class ROI(BaseModel):
    """Region of Interest definition."""

    name: str
    network: str
    mni_coords: tuple[float, float, float]
    hemisphere: Optional[str] = None  # L, R, or None for midline


class ConnectivityMatrix(BaseModel):
    """Connectivity matrix with metadata."""

    method: str = Field(..., description="Connectivity method (plv, correlation, etc.)")
    frequency_band: Optional[str] = Field(default=None, description="Frequency band name")
    freq_range: Optional[tuple[float, float]] = Field(default=None)
    time_window: Optional[tuple[float, float]] = Field(default=None)
    roi_names: List[str] = Field(..., description="ROI labels in order")
    matrix_file: Path = Field(..., description="Path to saved matrix (.npy)")

    @property
    def n_rois(self) -> int:
        return len(self.roi_names)
