"""
Artifact annotation step.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Optional, Tuple
import mne

from .base import ProcessingStep
from ..artifacts import ArtifactAnnotator


class ArtifactAnnotationStep(ProcessingStep):
    """
    Annotate artifact segments.

    Marks segments with high amplitude, gradient, flatline, or muscle artifacts
    as bad. These annotations can be used to exclude data during ICA fitting.

    Parameters
    ----------
    amplitude_threshold : float
        Peak-to-peak amplitude threshold in volts (default: 500e-6 = 500 µV)
    gradient_threshold : float or None
        Sample-to-sample gradient threshold in volts (default: None)
    flatline_duration : float
        Minimum flatline duration in seconds (default: 5.0)
    muscle_threshold : float or None
        Muscle artifact threshold (default: None)
    freq_muscle : tuple of float
        Frequency range for muscle detection (default: (110, 140))
    min_duration : float
        Minimum annotation duration in seconds (default: 0.1)

    Examples
    --------
    Conservative threshold:
    >>> step = ArtifactAnnotationStep(
    ...     amplitude_threshold=500e-6,  # 500 µV
    ...     flatline_duration=5.0
    ... )

    Aggressive with muscle detection:
    >>> step = ArtifactAnnotationStep(
    ...     amplitude_threshold=200e-6,  # 200 µV
    ...     muscle_threshold=5.0,
    ...     freq_muscle=(110, 140)
    ... )
    """

    name = "artifacts"
    version = "1.0"

    def __init__(
        self,
        amplitude_threshold: float = 500e-6,
        gradient_threshold: Optional[float] = None,
        flatline_duration: float = 5.0,
        muscle_threshold: Optional[float] = None,
        freq_muscle: Tuple[float, float] = (110, 140),
        min_duration: float = 0.1,
        enabled: bool = True,
    ):
        """Initialize artifact annotation step."""
        super().__init__(enabled=enabled)

        self.amplitude_threshold = amplitude_threshold
        self.gradient_threshold = gradient_threshold
        self.flatline_duration = flatline_duration
        self.muscle_threshold = muscle_threshold
        self.freq_muscle = freq_muscle
        self.min_duration = min_duration

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Annotate artifacts.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Data with artifact annotations added
        step_metadata : dict
            Annotation metadata
        """
        n_annotations_before = len(raw.annotations) if raw.annotations else 0

        # Create annotator
        annotator = ArtifactAnnotator(
            amplitude_threshold=self.amplitude_threshold,
            gradient_threshold=self.gradient_threshold,
            flatline_duration=self.flatline_duration,
            muscle_threshold=self.muscle_threshold,
            freq_muscle=self.freq_muscle,
            min_duration=self.min_duration,
        )

        # Annotate
        raw = annotator.annotate(raw)

        n_annotations_after = len(raw.annotations) if raw.annotations else 0
        n_new = n_annotations_after - n_annotations_before

        # Count by type
        annotation_counts = {}
        if raw.annotations:
            for desc in raw.annotations.description:
                if desc.startswith('BAD_'):
                    annotation_counts[desc] = annotation_counts.get(desc, 0) + 1

        step_metadata = {
            'applied': True,
            'n_annotations_added': n_new,
            'annotation_counts': annotation_counts,
            'amplitude_threshold': self.amplitude_threshold,
            'gradient_threshold': self.gradient_threshold,
            'flatline_duration': self.flatline_duration,
        }

        return raw, step_metadata

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'amplitude_threshold': self.amplitude_threshold,
            'flatline_duration': self.flatline_duration,
        })
        return config
