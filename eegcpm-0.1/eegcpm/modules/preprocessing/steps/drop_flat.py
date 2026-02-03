"""
Drop flat channels (zero or near-zero variance).

These channels cause numerical issues in bad channel detection (RANSAC, PREP)
and cannot be interpolated, so they must be removed early.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Tuple
import mne
import numpy as np

from .base import ProcessingStep


class DropFlatStep(ProcessingStep):
    """
    Drop flat channels with zero or near-zero variance.

    Flat channels are typically:
    - Reference electrodes (after re-referencing)
    - Disconnected/unplugged electrodes
    - Ground channels

    These channels cause division by zero in interpolation/detection algorithms,
    so they must be removed before bad channel detection.

    Parameters
    ----------
    variance_threshold : float, default=1e-15
        Channels with variance below this are considered flat
    enabled : bool, default=True
        Whether to apply this step

    Examples
    --------
    >>> step = DropFlatStep(variance_threshold=1e-15)
    >>> raw_out, metadata = step.process(raw, {})
    """

    name = "drop_flat"
    version = "1.0"

    def __init__(
        self,
        variance_threshold: float = 1e-15,
        enabled: bool = True,
    ):
        """Initialize drop flat step."""
        super().__init__(enabled=enabled)
        self.variance_threshold = variance_threshold

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Drop flat channels.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Data with flat channels removed
        step_metadata : dict
            Flat channel detection metadata
        """
        # Get EEG data
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

        if len(eeg_picks) == 0:
            return raw, {
                'applied': False,
                'reason': 'no_eeg_channels',
                'flat_channels': [],
                'n_dropped': 0,
            }

        # Compute variance for each channel
        data = raw.get_data(picks=eeg_picks)
        variances = np.var(data, axis=1)

        # Find flat channels
        flat_indices = np.where(variances < self.variance_threshold)[0]
        flat_channels = [raw.ch_names[eeg_picks[i]] for i in flat_indices]

        # Drop flat channels
        if flat_channels:
            print(f"[DROP_FLAT] Dropping {len(flat_channels)} flat channels: {flat_channels}")
            print(f"[DROP_FLAT] Variances: {[f'{variances[i]:.2e}' for i in flat_indices]}")
            raw.drop_channels(flat_channels)
            print(f"[DROP_FLAT] After drop: {len(raw.ch_names)} channels remain")

        step_metadata = {
            'applied': True,
            'flat_channels': flat_channels,
            'n_dropped': len(flat_channels),
            'variance_threshold': self.variance_threshold,
            'variances': {ch: float(variances[i]) for i, ch in enumerate(flat_channels)},
        }

        return raw, step_metadata

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'variance_threshold': self.variance_threshold,
        })
        return config
