"""
Temporal filtering step with full parameter control.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import mne

from .base import ProcessingStep


class FilterStep(ProcessingStep):
    """
    Temporal filtering step (bandpass, highpass, lowpass, notch).

    Exposes all MNE filter parameters for complete control over
    filter characteristics.

    Parameters
    ----------
    l_freq : float or None
        Low cut-off frequency in Hz (highpass). None for no highpass.
    h_freq : float or None
        High cut-off frequency in Hz (lowpass). None for no lowpass.
    notch_freq : float, list of float, or None
        Frequency/frequencies to notch filter (remove line noise)
    method : str
        Filtering method: 'fir' (default) or 'iir'
    iir_params : dict or None
        IIR filter parameters (if method='iir')
    fir_design : str
        FIR filter design: 'firwin' (default) or 'firwin2'
    fir_window : str
        FIR window type: 'hamming' (default), 'hann', 'blackman', etc.
    phase : str
        Phase behavior: 'zero' (default), 'zero-double', 'minimum'
    filter_length : str or int
        FIR filter length: 'auto' (default) or integer (samples)
    l_trans_bandwidth : str or float
        Lowpass transition bandwidth: 'auto' (default) or float (Hz)
    h_trans_bandwidth : str or float
        Highpass transition bandwidth: 'auto' (default) or float (Hz)
    picks : str, list, or None
        Channels to filter (default: 'eeg')
    skip_by_annotation : list of str
        Annotations to skip during filtering
    pad : str
        Padding method: 'reflect_limited' (default)
    verbose : bool
        Whether to print filter info

    Examples
    --------
    Basic bandpass filter:
    >>> step = FilterStep(l_freq=1.0, h_freq=40.0)

    Highpass only with custom FIR parameters:
    >>> step = FilterStep(
    ...     l_freq=0.5,
    ...     h_freq=None,
    ...     method='fir',
    ...     fir_window='blackman',
    ...     filter_length=1000
    ... )

    IIR Butterworth filter:
    >>> step = FilterStep(
    ...     l_freq=1.0,
    ...     h_freq=40.0,
    ...     method='iir',
    ...     iir_params={'order': 4, 'ftype': 'butter'}
    ... )

    Notch filter for line noise:
    >>> step = FilterStep(
    ...     l_freq=None,
    ...     h_freq=None,
    ...     notch_freq=[50, 100, 150]  # 50 Hz + harmonics
    ... )
    """

    name = "filter"
    version = "1.0"

    def __init__(
        self,
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        notch_freq: Optional[Union[float, List[float]]] = None,

        # Filter method
        method: str = 'fir',
        iir_params: Optional[Dict] = None,

        # FIR parameters
        fir_design: str = 'firwin',
        fir_window: str = 'hamming',
        phase: str = 'zero',
        filter_length: Union[str, int] = 'auto',
        l_trans_bandwidth: Union[str, float] = 'auto',
        h_trans_bandwidth: Union[str, float] = 'auto',

        # Processing options
        picks: Union[str, List[str], None] = 'eeg',
        skip_by_annotation: Optional[List[str]] = None,
        pad: str = 'reflect_limited',
        verbose: bool = False,

        # Base class
        enabled: bool = True,
    ):
        """Initialize filter step."""
        super().__init__(enabled=enabled)

        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.method = method
        self.iir_params = iir_params
        self.fir_design = fir_design
        self.fir_window = fir_window
        self.phase = phase
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.picks = picks
        self.skip_by_annotation = skip_by_annotation or []
        self.pad = pad
        self.verbose = verbose

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply temporal filtering.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Filtered raw data
        step_metadata : dict
            Filtering metadata
        """
        # Bandpass/highpass/lowpass filter
        if self.l_freq is not None or self.h_freq is not None:
            raw.filter(
                l_freq=self.l_freq,
                h_freq=self.h_freq,
                picks=self.picks,
                method=self.method,
                iir_params=self.iir_params,
                fir_design=self.fir_design,
                fir_window=self.fir_window,
                phase=self.phase,
                filter_length=self.filter_length,
                l_trans_bandwidth=self.l_trans_bandwidth,
                h_trans_bandwidth=self.h_trans_bandwidth,
                skip_by_annotation=self.skip_by_annotation,
                pad=self.pad,
                verbose=self.verbose,
            )

        # Notch filter
        if self.notch_freq is not None:
            freqs = self.notch_freq if isinstance(self.notch_freq, list) else [self.notch_freq]
            raw.notch_filter(
                freqs=freqs,
                picks=self.picks,
                method=self.method,
                verbose=self.verbose,
            )

        # Build metadata
        step_metadata = {
            'l_freq': self.l_freq,
            'h_freq': self.h_freq,
            'notch_freq': self.notch_freq,
            'method': self.method,
            'applied': True,
        }

        # Add method-specific details
        if self.method == 'fir':
            step_metadata.update({
                'fir_window': self.fir_window,
                'fir_design': self.fir_design,
                'phase': self.phase,
            })
        elif self.method == 'iir':
            step_metadata['iir_params'] = self.iir_params

        return raw, step_metadata

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """
        Validate inputs.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input data

        Returns
        -------
        valid : bool
            True if valid

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check that at least one frequency parameter is specified
        if self.l_freq is None and self.h_freq is None and self.notch_freq is None:
            raise ValueError("Must specify at least one of: l_freq, h_freq, notch_freq")

        # Check frequency values
        nyquist = raw.info['sfreq'] / 2
        if self.h_freq is not None and self.h_freq >= nyquist:
            raise ValueError(
                f"h_freq ({self.h_freq} Hz) must be less than Nyquist frequency ({nyquist} Hz)"
            )

        return True

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'l_freq': self.l_freq,
            'h_freq': self.h_freq,
            'notch_freq': self.notch_freq,
            'method': self.method,
            'fir_window': self.fir_window,
            'phase': self.phase,
        })
        return config
