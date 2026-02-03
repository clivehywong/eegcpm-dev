"""
Lowpass filtering step (separate from highpass for optimal pipeline ordering).

Allows Zapline/line noise removal to occur between highpass and lowpass filtering,
preventing spectral leakage from line noise into the passband.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Optional, Tuple
import mne

from .base import ProcessingStep


class LowpassStep(ProcessingStep):
    """
    Lowpass filtering step.

    Separate lowpass step allows optimal pipeline ordering:
    Highpass → Bad Channels → Zapline → Lowpass → ASR → ICA

    This prevents spectral leakage from line noise (50/60 Hz) into the
    passband when using low-frequency cutoffs (e.g., 40 Hz for beta).

    Parameters
    ----------
    h_freq : float
        High cut-off frequency in Hz (lowpass)
    method : str
        Filtering method: 'fir' (default) or 'iir'
    iir_params : dict or None
        IIR filter parameters (if method='iir')
        Example: {'order': 4, 'ftype': 'butter'}
    fir_design : str
        FIR filter design: 'firwin' (default) or 'firwin2'
    fir_window : str
        FIR window type: 'hamming' (default), 'hann', 'blackman'
    phase : str
        Phase behavior: 'zero' (default), 'zero-double', 'minimum'
    filter_length : str or int
        FIR filter length: 'auto' (default) or integer (samples)
    h_trans_bandwidth : str or float
        Highpass transition bandwidth: 'auto' (default) or float (Hz)
    picks : str, list, or None
        Channels to filter (default: 'eeg')
    verbose : bool
        Whether to print filter info

    Examples
    --------
    Standard 40 Hz lowpass:
    >>> step = LowpassStep(h_freq=40.0)

    With custom FIR parameters:
    >>> step = LowpassStep(
    ...     h_freq=30.0,
    ...     fir_window='blackman',
    ...     filter_length=1000
    ... )

    IIR Butterworth:
    >>> step = LowpassStep(
    ...     h_freq=40.0,
    ...     method='iir',
    ...     iir_params={'order': 4, 'ftype': 'butter'}
    ... )
    """

    name = "lowpass"
    version = "1.0"

    def __init__(
        self,
        h_freq: float,

        # Filter method
        method: str = 'fir',
        iir_params: Optional[Dict] = None,

        # FIR parameters
        fir_design: str = 'firwin',
        fir_window: str = 'hamming',
        phase: str = 'zero',
        filter_length: str = 'auto',
        h_trans_bandwidth: str = 'auto',

        # Channel selection
        picks: str = 'eeg',

        # Misc
        verbose: bool = False,
        enabled: bool = True,
    ):
        """Initialize lowpass filter step."""
        super().__init__(enabled=enabled)

        self.h_freq = h_freq
        self.method = method
        self.iir_params = iir_params

        self.fir_design = fir_design
        self.fir_window = fir_window
        self.phase = phase
        self.filter_length = filter_length
        self.h_trans_bandwidth = h_trans_bandwidth

        self.picks = picks
        self.verbose = verbose

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply lowpass filter.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Lowpass-filtered raw data (in-place modification)
        step_metadata : dict
            Filter metadata
        """
        # Build filter kwargs
        filter_kwargs = {
            'l_freq': None,  # No highpass
            'h_freq': self.h_freq,
            'picks': self.picks,
            'method': self.method,
            'verbose': self.verbose,
        }

        # Add method-specific parameters
        if self.method == 'fir':
            filter_kwargs.update({
                'fir_design': self.fir_design,
                'fir_window': self.fir_window,
                'phase': self.phase,
                'filter_length': self.filter_length,
                'h_trans_bandwidth': self.h_trans_bandwidth,
            })
        elif self.method == 'iir' and self.iir_params:
            filter_kwargs['iir_params'] = self.iir_params

        # Apply filter
        raw.filter(**filter_kwargs)

        # Build metadata
        step_metadata = {
            'h_freq': self.h_freq,
            'method': self.method,
            'applied': True,
        }

        if self.method == 'fir':
            step_metadata.update({
                'fir_window': self.fir_window,
                'fir_design': self.fir_design,
                'phase': self.phase,
            })

        return raw, step_metadata

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """Validate inputs."""
        # Check Nyquist frequency
        nyquist = raw.info['sfreq'] / 2
        if self.h_freq >= nyquist:
            raise ValueError(
                f"Lowpass frequency {self.h_freq} Hz must be less than "
                f"Nyquist frequency {nyquist} Hz"
            )

        return True

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'h_freq': self.h_freq,
            'method': self.method,
        })

        if self.method == 'fir':
            config.update({
                'fir_window': self.fir_window,
                'phase': self.phase,
            })

        return config
