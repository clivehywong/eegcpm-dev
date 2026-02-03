"""
Base classes for Quality Control (QC) reporting.

Provides abstract base class and data structures for QC metrics and results,
with support for matplotlib figure embedding as base64 data URIs.
"""

import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


@dataclass
class QCMetric:
    """
    Single QC metric with value and status.

    Attributes:
        name: Metric name (e.g., "Mean Amplitude")
        value: Numeric value of the metric
        unit: Unit of measurement (e.g., "µV", "%")
        status: Overall status: "ok", "warning", or "bad"
        threshold_warning: Value above which status is "warning"
        threshold_bad: Value above which status is "bad"
    """

    name: str
    value: float
    unit: str = ""
    status: str = "ok"  # ok, warning, bad
    threshold_warning: Optional[float] = None
    threshold_bad: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "threshold_warning": self.threshold_warning,
            "threshold_bad": self.threshold_bad,
        }


@dataclass
class QCResult:
    """
    Collection of QC metrics for one subject/run.

    Attributes:
        subject_id: Unique identifier for the subject
        metrics: List of QCMetric objects
        figures: Dictionary mapping figure names to PNG bytes
        status: Overall status: "ok", "warning", or "bad"
        notes: List of notes/comments about the QC assessment
        metadata: Additional metadata (e.g., file paths, processing info)
    """

    subject_id: str
    metrics: List[QCMetric] = field(default_factory=list)
    figures: Dict[str, bytes] = field(default_factory=dict)  # name -> PNG bytes
    status: str = "ok"  # overall: ok, warning, bad
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, metric: QCMetric) -> "QCResult":
        """
        Add a QC metric to the result.

        Args:
            metric: QCMetric to add

        Returns:
            Self for method chaining
        """
        self.metrics.append(metric)
        return self

    def add_figure(self, name: str, fig_bytes: bytes) -> "QCResult":
        """
        Add a matplotlib figure as PNG bytes.

        Args:
            name: Name/identifier for the figure
            fig_bytes: PNG bytes of the figure

        Returns:
            Self for method chaining
        """
        self.figures[name] = fig_bytes
        return self

    def add_note(self, note: str) -> "QCResult":
        """
        Add a note to the QC result.

        Args:
            note: Note text

        Returns:
            Self for method chaining
        """
        self.notes.append(note)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation (excludes binary figure data).

        Returns:
            Dictionary with serializable data
        """
        return {
            "subject_id": self.subject_id,
            "metrics": [m.to_dict() for m in self.metrics],
            "figure_names": list(self.figures.keys()),
            "status": self.status,
            "notes": self.notes,
        }

    def save_json(self, path: Path) -> Path:
        """
        Save QC result as JSON (without figure data).

        Args:
            path: Path to save JSON file

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class BaseQC(ABC):
    """
    Abstract base class for QC modules.

    Subclasses must implement the compute() method to perform
    subject-specific quality control assessment.
    """

    def __init__(self, output_dir: Path, config: Optional[Dict] = None):
        """
        Initialize QC module.

        Args:
            output_dir: Directory for output files
            config: Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def compute(self, data: Any, subject_id: str, **kwargs) -> QCResult:
        """
        Compute QC metrics for given data.

        Must be implemented by subclasses.

        Args:
            data: Input data (format depends on subclass)
            subject_id: Subject identifier
            **kwargs: Additional arguments

        Returns:
            QCResult with metrics and figures
        """
        pass

    def save_report(self, result: QCResult, filename: str = None) -> Path:
        """
        Save QC result as JSON file.

        Args:
            result: QCResult to save
            filename: Optional filename (defaults to "{subject_id}_qc.json")

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{result.subject_id}_qc.json"
        return result.save_json(self.output_dir / filename)

    @staticmethod
    def fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
        """
        Convert matplotlib figure to base64-encoded PNG string.

        Args:
            fig: Matplotlib figure
            dpi: DPI for PNG rendering

        Returns:
            Base64-encoded PNG string (for embedding as data URI)
        """
        import base64
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        png_bytes = buffer.getvalue()
        buffer.close()
        # Convert to base64 string
        return base64.b64encode(png_bytes).decode('ascii')

    @staticmethod
    def bytes_to_data_uri(fig_bytes: bytes) -> str:
        """
        Convert PNG bytes to data URI string for HTML embedding.

        Args:
            fig_bytes: PNG bytes

        Returns:
            data:image/png;base64,... string
        """
        encoded = base64.b64encode(fig_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"
