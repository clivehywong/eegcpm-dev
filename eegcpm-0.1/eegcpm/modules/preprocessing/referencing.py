"""
EEG Re-referencing Module

Provides flexible re-referencing methods for EEG data including average reference,
mastoid reference, custom reference, and robust average (excluding bad channels).

Re-referencing is a critical preprocessing step that affects all downstream analyses
including connectivity and source localization.

Author: EEGCPM Development Team
Created: 2025-01
"""

from typing import List, Optional, Union
import mne
import numpy as np


class EEGReferencing:
    """
    EEG re-referencing with multiple reference schemes.

    Supports:
    - Average reference (most common for source localization)
    - Mastoid reference (single or linked)
    - Custom reference (specific channel or channels)
    - Robust average (excluding bad channels)
    - No re-referencing (keep original)

    Parameters
    ----------
    ref_type : str
        Reference type: 'average', 'mastoid', 'custom', 'robust_average', 'original'
    ref_channels : str or list of str, optional
        Channel name(s) for custom or mastoid reference.
        Examples:
        - 'M1' or 'TP9' for single mastoid
        - ['M1', 'M2'] or ['TP9', 'TP10'] for linked mastoids
        - 'Cz' for custom reference
    projection : bool, default=True
        If True, add reference as a projection (can be undone later).
        If False, re-reference data directly (cannot be undone).
    exclude_bads : bool, default=True
        If True, exclude bad channels from average/robust_average computation.

    Examples
    --------
    Average reference (most common):
    >>> ref = EEGReferencing(ref_type='average')
    >>> raw_reref = ref.apply(raw)

    Linked mastoid reference:
    >>> ref = EEGReferencing(ref_type='mastoid', ref_channels=['TP9', 'TP10'])
    >>> raw_reref = ref.apply(raw)

    Custom reference (e.g., Cz):
    >>> ref = EEGReferencing(ref_type='custom', ref_channels='Cz')
    >>> raw_reref = ref.apply(raw)

    Robust average (exclude bad channels):
    >>> ref = EEGReferencing(ref_type='robust_average', exclude_bads=True)
    >>> raw_reref = ref.apply(raw)
    """

    VALID_TYPES = ['average', 'mastoid', 'custom', 'robust_average', 'original']

    def __init__(
        self,
        ref_type: str = 'average',
        ref_channels: Optional[Union[str, List[str]]] = None,
        projection: bool = True,
        exclude_bads: bool = True,
    ):
        """
        Initialize EEG re-referencing.

        Parameters
        ----------
        ref_type : str
            Reference type (see class docstring)
        ref_channels : str or list of str, optional
            Reference channel(s) for custom/mastoid
        projection : bool
            Use projection (reversible) vs direct re-reference
        exclude_bads : bool
            Exclude bad channels from average reference
        """
        if ref_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid ref_type '{ref_type}'. Must be one of {self.VALID_TYPES}"
            )

        self.ref_type = ref_type
        self.ref_channels = ref_channels
        self.projection = projection
        self.exclude_bads = exclude_bads

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.ref_type in ['mastoid', 'custom'] and self.ref_channels is None:
            raise ValueError(
                f"ref_channels must be provided for ref_type='{self.ref_type}'"
            )

        if self.ref_type == 'original' and self.ref_channels is not None:
            raise ValueError(
                "ref_channels should not be provided for ref_type='original'"
            )

    def apply(self, raw: mne.io.BaseRaw, copy: bool = True) -> mne.io.BaseRaw:
        """
        Apply re-referencing to raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG data to re-reference
        copy : bool, default=True
            If True, operate on a copy (recommended).
            If False, modify in-place.

        Returns
        -------
        mne.io.BaseRaw
            Re-referenced raw data

        Notes
        -----
        The original reference is stored in raw.info['custom_ref_applied']
        for provenance tracking.
        """
        if copy:
            raw = raw.copy()

        if self.ref_type == 'original':
            # No re-referencing
            return raw

        elif self.ref_type == 'average':
            raw = self._apply_average_reference(raw)

        elif self.ref_type == 'robust_average':
            raw = self._apply_robust_average_reference(raw)

        elif self.ref_type == 'mastoid':
            raw = self._apply_mastoid_reference(raw)

        elif self.ref_type == 'custom':
            raw = self._apply_custom_reference(raw)

        # Note: MNE stores reference info automatically in raw.info['custom_ref_applied']
        # via set_eeg_reference(), so we don't need to set it manually

        return raw

    def _apply_average_reference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply average reference across all EEG channels."""
        # MNE's set_eeg_reference with 'average' automatically excludes bad channels
        raw.set_eeg_reference(
            ref_channels='average',
            projection=self.projection,
            verbose=False
        )
        return raw

    def _apply_robust_average_reference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Apply robust average reference (explicitly excluding bad channels).

        This is similar to average reference but ensures bad channels are excluded.
        """
        # Get good EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads' if self.exclude_bads else [])

        if len(eeg_picks) == 0:
            raise ValueError("No good EEG channels found for robust average reference")

        # Compute average of good channels
        good_channels = [raw.ch_names[i] for i in eeg_picks]

        raw.set_eeg_reference(
            ref_channels=good_channels,
            projection=self.projection,
            verbose=False
        )

        return raw

    def _apply_mastoid_reference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Apply mastoid reference (single or linked).

        For linked mastoids, the average of both is used.
        """
        # Ensure ref_channels is a list
        if isinstance(self.ref_channels, str):
            ref_channels = [self.ref_channels]
        else:
            ref_channels = self.ref_channels

        # Validate channels exist
        missing = [ch for ch in ref_channels if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"Mastoid channel(s) not found in data: {missing}. "
                f"Available channels: {raw.ch_names}"
            )

        raw.set_eeg_reference(
            ref_channels=ref_channels,
            projection=self.projection,
            verbose=False
        )

        return raw

    def _apply_custom_reference(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Apply custom reference to specific channel(s).

        Can be single channel (e.g., 'Cz') or multiple channels (average).
        """
        # Ensure ref_channels is a list
        if isinstance(self.ref_channels, str):
            ref_channels = [self.ref_channels]
        else:
            ref_channels = self.ref_channels

        # Validate channels exist
        missing = [ch for ch in ref_channels if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"Reference channel(s) not found in data: {missing}. "
                f"Available channels: {raw.ch_names}"
            )

        raw.set_eeg_reference(
            ref_channels=ref_channels,
            projection=self.projection,
            verbose=False
        )

        return raw

    def get_config(self) -> dict:
        """
        Get current configuration as dictionary.

        Returns
        -------
        dict
            Configuration dictionary compatible with YAML/JSON
        """
        return {
            'type': self.ref_type,
            'channels': self.ref_channels,
            'projection': self.projection,
            'exclude_bads': self.exclude_bads,
        }


def apply_reference_from_config(
    raw: mne.io.BaseRaw,
    config: dict,
    copy: bool = True
) -> mne.io.BaseRaw:
    """
    Apply re-referencing from configuration dictionary.

    This is a convenience function for pipeline integration.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    config : dict
        Configuration dictionary with keys:
        - 'type': Reference type
        - 'channels': Reference channels (optional)
        - 'projection': Use projection (optional, default=True)
        - 'exclude_bads': Exclude bad channels (optional, default=True)
    copy : bool
        Operate on copy

    Returns
    -------
    mne.io.BaseRaw
        Re-referenced raw data

    Examples
    --------
    >>> config = {
    ...     'type': 'average',
    ...     'projection': True,
    ...     'exclude_bads': True
    ... }
    >>> raw_reref = apply_reference_from_config(raw, config)

    >>> config = {
    ...     'type': 'mastoid',
    ...     'channels': ['TP9', 'TP10'],
    ...     'projection': False
    ... }
    >>> raw_reref = apply_reference_from_config(raw, config)
    """
    ref = EEGReferencing(
        ref_type=config.get('type', 'average'),
        ref_channels=config.get('channels'),
        projection=config.get('projection', True),
        exclude_bads=config.get('exclude_bads', True),
    )

    return ref.apply(raw, copy=copy)
