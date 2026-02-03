"""
Re-referencing step.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import mne

from .base import ProcessingStep


class ReferenceStep(ProcessingStep):
    """
    Re-referencing step.

    Changes the EEG reference to average reference, specific channels,
    or REST (Reference Electrode Standardization Technique).

    Parameters
    ----------
    type : str
        Reference type: 'average', 'channels', or 'rest'
    channels : str or list of str
        Reference channel(s) for 'channels' type
    projection : bool
        Use reference projection (default: True)
    exclude_bads : bool
        Exclude bad channels from reference (default: True)

    Examples
    --------
    Average reference:
    >>> step = ReferenceStep(type='average')

    Specific channel reference:
    >>> step = ReferenceStep(type='channels', channels=['Cz'])

    Linked mastoids:
    >>> step = ReferenceStep(type='channels', channels=['M1', 'M2'])
    """

    name = "reference"
    version = "1.0"

    def __init__(
        self,
        type: str = 'average',
        channels: Union[str, List[str], None] = None,
        projection: bool = True,
        exclude_bads: bool = True,
        enabled: bool = True,
    ):
        """Initialize reference step."""
        super().__init__(enabled=enabled)

        self.type = type
        self.channels = channels
        self.projection = projection
        self.exclude_bads = exclude_bads

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply re-referencing.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Re-referenced data
        step_metadata : dict
            Reference metadata
        """
        if self.type == 'average':
            raw.set_eeg_reference(
                ref_channels='average',
                projection=self.projection,
                verbose=False
            )
            ref_info = {'type': 'average'}

        elif self.type == 'channels':
            if not self.channels:
                raise ValueError("Must specify channels for 'channels' reference type")

            channels = self.channels if isinstance(self.channels, list) else [self.channels]
            raw.set_eeg_reference(
                ref_channels=channels,
                projection=self.projection,
                verbose=False
            )
            ref_info = {'type': 'channels', 'channels': channels}

        elif self.type == 'rest':
            # REST requires forward model - more complex
            # For now, fallback to average
            raw.set_eeg_reference(
                ref_channels='average',
                projection=self.projection,
                verbose=False
            )
            ref_info = {'type': 'rest_fallback_average', 'note': 'REST not yet implemented'}

        else:
            raise ValueError(f"Unknown reference type: {self.type}")

        step_metadata = {
            'applied': True,
            **ref_info,
            'projection': self.projection,
        }

        return raw, step_metadata

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'type': self.type,
            'channels': self.channels,
            'projection': self.projection,
        })
        return config
