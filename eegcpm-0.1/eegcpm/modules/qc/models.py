"""Data models for QC metrics.

Provides dataclass representations of QC metrics loaded from JSON files.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunQualityMetricsFromJSON:
    """Quality metrics loaded from QC metrics JSON file.

    This is a lightweight representation suitable for UI display,
    separate from the full RunQualityMetrics class used internally
    during preprocessing.
    """
    run: str
    subject_id: str
    session: str
    task: str
    pipeline: str
    quality_status: str  # 'excellent', 'good', 'acceptable', 'poor'
    pct_bad_channels: float
    n_bad_channels: int
    n_eeg_channels: int
    clustering_severity: str  # 'none', 'mild', 'moderate', 'severe'
    n_clustered_bad: int
    ica_success: bool
    n_ica_components: int
    n_components_rejected: int
    recommended_action: str  # 'accept', 'review', 'reject'
    qc_report_path: Optional[str] = None
    timestamp: Optional[str] = None

    @classmethod
    def from_json(cls, data: dict) -> "RunQualityMetricsFromJSON":
        """Create from JSON dictionary.

        Parameters
        ----------
        data : dict
            JSON data from QC metrics file

        Returns
        -------
        RunQualityMetricsFromJSON
            Parsed metrics object
        """
        quality_metrics = data.get('quality_metrics', {})

        return cls(
            run=data.get('run', 'unknown'),
            subject_id=data.get('subject_id', ''),
            session=data.get('session', '01'),
            task=data.get('task', ''),
            pipeline=data.get('pipeline', ''),
            quality_status=quality_metrics.get('quality_status', 'unknown'),
            pct_bad_channels=quality_metrics.get('pct_bad_channels', 0.0),
            n_bad_channels=quality_metrics.get('n_bad_channels', 0),
            n_eeg_channels=quality_metrics.get('n_eeg_channels', 0),
            clustering_severity=quality_metrics.get('clustering_severity', 'none'),
            n_clustered_bad=quality_metrics.get('n_clustered_bad', 0),
            ica_success=quality_metrics.get('ica_success', False),
            n_ica_components=quality_metrics.get('n_ica_components', 0),
            n_components_rejected=quality_metrics.get('n_components_rejected', 0),
            recommended_action=quality_metrics.get('recommended_action', 'review'),
            qc_report_path=data.get('qc_report_path'),
            timestamp=data.get('timestamp'),
        )
