"""
Notch filter step for line noise removal.

Alternative to Zapline for removing line noise (50/60 Hz) and harmonics
using traditional notch filtering.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import mne

from .base import ProcessingStep


class NotchStep(ProcessingStep):
    """
    Notch filter for line noise removal.

    Removes line noise (50/60 Hz) and harmonics using narrow-band rejection.
    Simpler than Zapline but creates spectral notches that may affect nearby
    brain signals.

    **When to use:**
    - Simple datasets with minimal line noise
    - When Zapline fails or is unavailable
    - Legacy pipeline compatibility

    **When NOT to use:**
    - Heavy line noise contamination → use Zapline
    - Signals of interest near line frequency → use Zapline
    - High-quality published research → use Zapline

    Parameters
    ----------
    freqs : float, list of float
        Frequency or frequencies to notch filter (Hz)
        Examples:
        - 60 (US fundamental only)
        - [60, 120, 180] (US + 2 harmonics)
        - [50, 100, 150] (EU + 2 harmonics)
    notch_widths : float, list of float, or None
        Width of notch in Hz (default: None = auto)
        Wider notches = more aggressive but more brain signal loss
    method : str
        Filter method: 'fir' (default) or 'iir'
    phase : str
        Phase behavior: 'zero' (default), 'zero-double', 'minimum'
    fir_design : str
        FIR design method: 'firwin' (default)
    fir_window : str
        FIR window: 'hamming' (default), 'hann', 'blackman'
    picks : str or list
        Channels to filter (default: 'eeg')
    verbose : bool
        Print filter information

    Examples
    --------
    US 60 Hz + 2 harmonics:
    >>> step = NotchStep(freqs=[60, 120, 180])

    EU 50 Hz fundamental only:
    >>> step = NotchStep(freqs=50)

    Custom notch width:
    >>> step = NotchStep(freqs=[60, 120], notch_widths=2.0)

    Notes
    -----
    Zapline is generally preferred over notch filtering because:
    - Adaptive to data characteristics
    - No fixed spectral notches
    - Better preserves brain signals near line frequency
    - Handles non-stationary line noise

    However, notch filtering is simpler and works well for clean data.
    """

    name = "notch"
    version = "1.0"

    def __init__(
        self,
        freqs: Union[float, List[float]],
        notch_widths: Optional[Union[float, List[float]]] = None,
        method: str = 'fir',
        phase: str = 'zero',
        fir_design: str = 'firwin',
        fir_window: str = 'hamming',
        picks: str = 'eeg',
        verbose: bool = False,
        enabled: bool = True,
    ):
        """Initialize notch filter step."""
        super().__init__(enabled=enabled)

        # Convert single freq to list
        if isinstance(freqs, (int, float)):
            self.freqs = [float(freqs)]
        else:
            self.freqs = [float(f) for f in freqs]

        self.notch_widths = notch_widths
        self.method = method
        self.phase = phase
        self.fir_design = fir_design
        self.fir_window = fir_window
        self.picks = picks
        self.verbose = verbose

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply notch filter.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Notch-filtered raw data (in-place modification)
        step_metadata : dict
            Notch filter metadata
        """
        if self.verbose:
            print(f"Notch filter: Removing {self.freqs} Hz")

        # Apply notch filter
        raw.notch_filter(
            freqs=self.freqs,
            notch_widths=self.notch_widths,
            picks=self.picks,
            method=self.method,
            phase=self.phase,
            fir_design=self.fir_design,
            fir_window=self.fir_window,
            verbose=self.verbose,
        )

        # Build metadata
        step_metadata = {
            'applied': True,
            'freqs': self.freqs,
            'notch_widths': self.notch_widths,
            'method': self.method,
            'n_freqs_removed': len(self.freqs),
        }

        if self.verbose:
            print(f"✓ Notch filter applied: {len(self.freqs)} frequenc{'y' if len(self.freqs) == 1 else 'ies'} removed")

        return raw, step_metadata

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """Validate inputs."""
        nyquist = raw.info['sfreq'] / 2

        for freq in self.freqs:
            if freq >= nyquist:
                raise ValueError(
                    f"Notch frequency {freq} Hz must be less than "
                    f"Nyquist frequency {nyquist} Hz"
                )

        return True

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'freqs': self.freqs,
            'notch_widths': self.notch_widths,
            'method': self.method,
        })
        return config
