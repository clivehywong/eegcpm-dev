"""Time-frequency analysis module."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import mne
import numpy as np

from eegcpm.pipeline.base import BaseModule, ModuleResult, EpochsModule


class TimeFrequencyModule(EpochsModule):
    """
    Time-frequency analysis of EEG epochs.

    Methods:
    - Morlet wavelets
    - Multitaper
    - Hilbert transform
    """

    name = "time_frequency"
    version = "0.1.0"
    description = "Time-frequency decomposition"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)
        self.method = config.get("method", "morlet")
        self.freqs = config.get("freqs", np.arange(4, 40, 1))
        self.n_cycles = config.get("n_cycles", 7)
        self.output_type = config.get("output", "power")  # power, phase, complex

    def process(
        self,
        data: mne.Epochs,
        subject: Optional[Any] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Compute time-frequency representation.

        Args:
            data: Epochs
            subject: Subject info

        Returns:
            ModuleResult with TFR
        """
        start_time = time.time()
        epochs = data
        output_files = []
        warnings = []

        try:
            freqs = np.array(self.freqs) if not isinstance(self.freqs, np.ndarray) else self.freqs

            if self.method == "morlet":
                tfr = mne.time_frequency.tfr_morlet(
                    epochs,
                    freqs=freqs,
                    n_cycles=self.n_cycles,
                    return_itc=False,
                    average=True,
                    verbose=False,
                )
            elif self.method == "multitaper":
                tfr = mne.time_frequency.tfr_multitaper(
                    epochs,
                    freqs=freqs,
                    n_cycles=self.n_cycles,
                    return_itc=False,
                    average=True,
                    verbose=False,
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Save TFR
            subject_id = subject.id if subject else "unknown"
            tfr_path = self.output_dir / f"{subject_id}_tfr.h5"
            tfr.save(tfr_path, overwrite=True, verbose=False)
            output_files.append(tfr_path)

            # Extract band power
            band_power = self._extract_band_power(tfr)

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "data": tfr,
                    "tfr": tfr,
                    "band_power": band_power,
                },
                output_files=output_files,
                warnings=warnings,
                metadata={
                    "method": self.method,
                    "freqs": freqs.tolist(),
                    "n_channels": len(tfr.ch_names),
                    "times": tfr.times.tolist(),
                },
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    def _extract_band_power(
        self,
        tfr,
        bands: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract power in frequency bands."""
        if bands is None:
            bands = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 45),
            }

        freqs = tfr.freqs
        power_data = tfr.data  # (n_channels, n_freqs, n_times)

        band_power = {}
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            if freq_mask.any():
                band_power[band_name] = np.mean(power_data[:, freq_mask, :], axis=1)

        return band_power

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "tfr": "MNE TFR object",
            "band_power": "Dict of band -> power array",
        }
