"""
Data loading utilities using MNE-Python.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import mne


def load_raw(
    file_path: Union[str, Path],
    preload: bool = True,
    verbose: bool = False,
) -> mne.io.Raw:
    """
    Load raw EEG data from various formats.

    Supports: .fif, .edf, .bdf, .set, .vhdr, .cnt

    Args:
        file_path: Path to EEG file
        preload: Whether to preload data into memory
        verbose: MNE verbosity

    Returns:
        MNE Raw object
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    readers = {
        ".fif": mne.io.read_raw_fif,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".vhdr": mne.io.read_raw_brainvision,
        ".cnt": mne.io.read_raw_cnt,
    }

    if suffix not in readers:
        raise ValueError(f"Unsupported file format: {suffix}")

    raw = readers[suffix](file_path, preload=preload, verbose=verbose)
    return raw


def load_epochs(
    file_path: Union[str, Path],
    preload: bool = True,
    verbose: bool = False,
) -> mne.Epochs:
    """
    Load epoched EEG data.

    Args:
        file_path: Path to epochs file (.fif or -epo.fif)
        preload: Whether to preload data into memory
        verbose: MNE verbosity

    Returns:
        MNE Epochs object
    """
    file_path = Path(file_path)
    return mne.read_epochs(file_path, preload=preload, verbose=verbose)


def load_events(
    file_path: Union[str, Path],
    raw: Optional[mne.io.Raw] = None,
) -> tuple:
    """
    Load or extract events from file or raw data.

    Args:
        file_path: Path to events file (.tsv, .txt) or raw EEG file
        raw: Optional Raw object to extract events from

    Returns:
        Tuple of (events array, event_id dict)
    """
    file_path = Path(file_path)

    if file_path.suffix == ".tsv":
        # BIDS-style events file
        import pandas as pd
        df = pd.read_csv(file_path, sep="\t")
        # Convert to MNE format
        # Assumes columns: onset, duration, trial_type (or value)
        # This is a simplified version
        events = []
        event_id = {}
        for i, row in df.iterrows():
            onset_sample = int(row["onset"] * raw.info["sfreq"]) if raw else int(row["onset"])
            trial_type = row.get("trial_type", row.get("value", str(i)))
            if trial_type not in event_id:
                event_id[trial_type] = len(event_id) + 1
            events.append([onset_sample, 0, event_id[trial_type]])
        import numpy as np
        return np.array(events), event_id

    elif raw is not None:
        # Extract from raw stim channel
        events = mne.find_events(raw, verbose=False)
        event_id = {str(e): e for e in set(events[:, 2])}
        return events, event_id

    else:
        raise ValueError("Either provide events file or raw object")


def get_montage(
    kind: str = "standard_1020",
) -> mne.channels.DigMontage:
    """
    Get standard EEG montage.

    Args:
        kind: Montage name (standard_1020, standard_1005, biosemi64, etc.)

    Returns:
        MNE DigMontage
    """
    return mne.channels.make_standard_montage(kind)
