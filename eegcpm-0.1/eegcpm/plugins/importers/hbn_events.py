"""
BIDS-compliant events.tsv writer for HBN data.

Generates events.tsv files with behavioral data columns and
corresponding events.json sidecar files with column descriptions.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eegcpm.plugins.importers.hbn_behavioral import BehavioralResult, TrialData

logger = logging.getLogger(__name__)


@dataclass
class BIDSEvent:
    """BIDS-compliant event representation."""

    onset: float  # Required: seconds from recording start
    duration: float  # Required: seconds (0 for instantaneous)
    trial_type: str  # Required: event category

    # Optional behavioral columns
    response_time: Optional[float] = None
    response: Optional[str] = None
    correct: Optional[int] = None  # 0/1 for BIDS compatibility
    value: Optional[str] = None  # Original trigger value

    # Stimulus info
    stim_side: Optional[str] = None

    # Trial metadata
    trial_number: Optional[int] = None

    # Task-specific extras
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_extras: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for TSV export."""
        d = {
            "onset": f"{self.onset:.6f}",
            "duration": f"{self.duration:.6f}" if self.duration > 0 else "0",
            "trial_type": self.trial_type,
        }

        if self.value is not None:
            d["value"] = self.value
        if self.response_time is not None:
            d["response_time"] = f"{self.response_time:.6f}"
        if self.response is not None:
            d["response"] = self.response
        if self.correct is not None:
            d["correct"] = str(self.correct)
        if self.stim_side is not None:
            d["stim_side"] = self.stim_side
        if self.trial_number is not None:
            d["trial_number"] = str(self.trial_number)

        if include_extras and self.extras:
            for k, v in self.extras.items():
                if v is not None:
                    d[k] = str(v) if not isinstance(v, str) else v

        return d


# Column descriptions for events.json sidecar
BIDS_EVENTS_COLUMNS = {
    "onset": {
        "LongName": "Event onset time",
        "Description": "Onset time of the event in seconds relative to the recording start",
        "Units": "s",
    },
    "duration": {
        "LongName": "Event duration",
        "Description": "Duration of the event in seconds (0 for instantaneous events)",
        "Units": "s",
    },
    "trial_type": {
        "LongName": "Trial type",
        "Description": "Categorical descriptor of the event type",
        "Levels": {
            "stimulus": "Stimulus presentation",
            "response": "Participant response",
            "trigger": "EEG trigger marker",
        },
    },
    "value": {
        "LongName": "Event value",
        "Description": "Original trigger code or event identifier from the recording",
    },
    "response_time": {
        "LongName": "Response time",
        "Description": "Reaction time from stimulus onset to response",
        "Units": "s",
    },
    "response": {
        "LongName": "Response",
        "Description": "Participant's response",
        "Levels": {
            "left": "Left button press",
            "right": "Right button press",
            "none": "No response (missed trial)",
        },
    },
    "correct": {
        "LongName": "Response correctness",
        "Description": "Whether the response was correct (1) or incorrect (0)",
        "Levels": {
            "1": "Correct response",
            "0": "Incorrect response",
        },
    },
    "stim_side": {
        "LongName": "Stimulus side",
        "Description": "Side of the correct target stimulus",
        "Levels": {
            "left": "Target on left",
            "right": "Target on right",
        },
    },
    "trial_number": {
        "LongName": "Trial number",
        "Description": "Sequential trial number within the block",
    },
}

# HBN trigger code mappings for each paradigm
HBN_TRIGGER_MAPPING = {
    "SurrSupp_Block": {
        93: "block_start_1",
        97: "block_start_2",
        5: "trial_start",
        8: "target_left",
        9: "target_right",
        12: "response_left",
        13: "response_right",
    },
    "SAIIT_2AFC": {
        94: "block_start_1",
        95: "block_start_2",
        96: "block_start_3",
        5: "trial_start",
        8: "target_left",
        9: "target_right",
        12: "response_left",
        13: "response_right",
    },
    "RestingState": {
        90: "rest_start",
    },
    "vis_learn": {
        91: "task_start",
    },
    "WISC_ProcSpeed": {
        92: "task_start",
    },
    "Video": {
        81: "video_start_1",
        82: "video_start_2",
        83: "video_start_3",
        84: "video_start_4",
    },
}

# Additional columns for specific paradigms
PARADIGM_COLUMNS = {
    "saiit2afc": {
        "iti": {
            "LongName": "Inter-trial interval",
            "Description": "Duration between trials",
            "Units": "s",
        },
    },
    "surrsuppblock": {
        "bg_contrast": {
            "LongName": "Background contrast",
            "Description": "Contrast level of the background stimulus",
        },
        "center_contrast": {
            "LongName": "Center contrast",
            "Description": "Contrast level of the center stimulus",
        },
        "stim_condition": {
            "LongName": "Stimulus condition",
            "Description": "Condition code for the stimulus configuration",
        },
    },
    "video": {
        "movie_id": {
            "LongName": "Movie ID",
            "Description": "Identifier of the video segment being presented",
        },
    },
    "wiscprocspeed": {
        "button": {
            "LongName": "Button pressed",
            "Description": "Button identifier that was pressed",
        },
    },
}


def trials_to_bids_events(
    trials: List[TrialData],
    paradigm: str,
    eeg_events: Optional[np.ndarray] = None,
    sfreq: float = 500.0,
) -> List[BIDSEvent]:
    """
    Convert behavioral trials to BIDS events.

    Args:
        trials: List of TrialData from behavioral parsing
        paradigm: Paradigm name
        eeg_events: Optional EEG events array [sample, 0, code] to merge
        sfreq: Sampling frequency for converting samples to seconds

    Returns:
        List of BIDSEvent objects
    """
    events = []

    # Get trigger mapping for this paradigm
    # Try exact match first, then case-insensitive with common variations
    trigger_map = {}
    if paradigm in HBN_TRIGGER_MAPPING:
        trigger_map = HBN_TRIGGER_MAPPING[paradigm]
    else:
        # Try case-insensitive matching
        paradigm_lower = paradigm.lower().replace('-', '').replace('_', '')
        for key in HBN_TRIGGER_MAPPING:
            key_lower = key.lower().replace('-', '').replace('_', '')
            if paradigm_lower == key_lower:
                trigger_map = HBN_TRIGGER_MAPPING[key]
                break

    # Process EEG trigger events
    if eeg_events is not None and len(eeg_events) > 0:
        for evt in eeg_events:
            sample, _, code = evt
            onset = sample / sfreq

            # Map trigger code to meaningful event type
            if code in trigger_map:
                event_type = trigger_map[code]
            else:
                # Unknown trigger - preserve with generic name
                event_type = f"trigger_{code}"

            trigger_event = BIDSEvent(
                onset=onset,
                duration=0,
                trial_type=event_type,
                value=str(code),
            )
            events.append(trigger_event)

    # Add behavioral trials metadata if available
    # Match by trial index for tasks with target events (8=left, 9=right)
    if trials:
        # Extract target events (trial start markers: 8=left, 9=right for SAIIT/SurrSupp)
        target_events = [e for e in events if e.trial_type in ['target_left', 'target_right']]

        # Match behavioral trials to target events by index
        for i, trial in enumerate(trials):
            if i < len(target_events):
                target_event = target_events[i]

                # Add trial metadata to the trigger event
                target_event.stim_side = trial.stimulus_side
                target_event.trial_number = trial.trial_number
                if trial.extras:
                    target_event.extras.update(trial.extras)

                # Add response information if available
                if trial.response is not None and trial.response != "none":
                    target_event.response = trial.response
                    target_event.response_time = trial.response_time
                    target_event.correct = 1 if trial.correct else 0

    # Sort by onset time
    events.sort(key=lambda e: e.onset)

    return events


def write_events_tsv(
    events: List[BIDSEvent],
    output_path: Path,
    include_extras: bool = True,
) -> Path:
    """
    Write events to BIDS-compliant TSV file.

    Args:
        events: List of BIDSEvent objects
        output_path: Output TSV file path
        include_extras: Include paradigm-specific extra columns

    Returns:
        Path to written file
    """
    if not events:
        logger.warning(f"No events to write to {output_path}")
        return output_path

    # Collect all unique columns
    all_dicts = [e.to_dict(include_extras=include_extras) for e in events]

    # Get column order (required columns first)
    required_cols = ["onset", "duration", "trial_type"]
    all_cols = set()
    for d in all_dicts:
        all_cols.update(d.keys())

    # Order: required, then optional standard, then extras
    standard_optional = ["value", "response_time", "response", "correct", "stim_side", "trial_number"]
    extra_cols = sorted(all_cols - set(required_cols) - set(standard_optional))
    col_order = required_cols + [c for c in standard_optional if c in all_cols] + extra_cols

    # Write TSV
    with open(output_path, "w") as f:
        f.write("\t".join(col_order) + "\n")
        for d in all_dicts:
            row = [d.get(col, "n/a") for col in col_order]
            f.write("\t".join(row) + "\n")

    logger.debug(f"Wrote {len(events)} events to {output_path}")
    return output_path


def write_events_json(
    output_path: Path,
    paradigm: str,
    columns_used: Optional[List[str]] = None,
) -> Path:
    """
    Write events.json sidecar file with column descriptions.

    Args:
        output_path: Output JSON file path
        paradigm: Paradigm name for paradigm-specific columns
        columns_used: List of columns actually used (optional filter)

    Returns:
        Path to written file
    """
    # Start with standard BIDS columns
    sidecar = dict(BIDS_EVENTS_COLUMNS)

    # Add paradigm-specific columns
    paradigm_lower = paradigm.lower()
    if paradigm_lower in PARADIGM_COLUMNS:
        sidecar.update(PARADIGM_COLUMNS[paradigm_lower])

    # Filter to only used columns if specified
    if columns_used:
        sidecar = {k: v for k, v in sidecar.items() if k in columns_used}

    with open(output_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.debug(f"Wrote events sidecar to {output_path}")
    return output_path


def write_trials_tsv(
    trials: List[TrialData],
    output_path: Path,
) -> Path:
    """
    Write full trial-by-trial table to TSV.

    This is a more detailed table than events.tsv,
    containing all behavioral variables per trial.

    Args:
        trials: List of TrialData objects
        output_path: Output TSV file path

    Returns:
        Path to written file
    """
    if not trials:
        logger.warning(f"No trials to write to {output_path}")
        return output_path

    # Convert to dictionaries
    all_dicts = [t.to_dict() for t in trials]

    # Get all columns
    all_cols = set()
    for d in all_dicts:
        all_cols.update(d.keys())

    # Column order: standard first, then extras alphabetically
    standard_cols = [
        "trial_number", "onset", "duration",
        "stimulus_type", "stimulus_side",
        "response", "response_time", "correct",
    ]
    extra_cols = sorted(all_cols - set(standard_cols))
    col_order = [c for c in standard_cols if c in all_cols] + extra_cols

    # Write TSV
    with open(output_path, "w") as f:
        f.write("\t".join(col_order) + "\n")
        for d in all_dicts:
            row = []
            for col in col_order:
                val = d.get(col, "")
                if val is None:
                    val = "n/a"
                elif isinstance(val, float):
                    val = f"{val:.6f}"
                else:
                    val = str(val)
                row.append(val)
            f.write("\t".join(row) + "\n")

    logger.debug(f"Wrote {len(trials)} trials to {output_path}")
    return output_path


class BIDSEventsWriter:
    """
    Writer for BIDS-compliant events and behavioral data.

    Handles conversion of behavioral results to:
    - events.tsv: BIDS events file
    - events.json: Column descriptions sidecar
    - trials.tsv: Full trial-by-trial table
    """

    def __init__(self, output_dir: Path):
        """
        Initialize writer.

        Args:
            output_dir: Base output directory (BIDS root or subject/session/eeg)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_behavioral_data(
        self,
        behavioral_result: BehavioralResult,
        bids_prefix: str,
        eeg_events: Optional[np.ndarray] = None,
        sfreq: float = 500.0,
    ) -> Dict[str, Path]:
        """
        Write all behavioral data files.

        Args:
            behavioral_result: Parsed behavioral data
            bids_prefix: BIDS filename prefix (e.g., "sub-NDAR*_ses-01_task-*")
            eeg_events: Optional EEG events to merge
            sfreq: Sampling frequency

        Returns:
            Dictionary of output file paths
        """
        outputs = {}

        if not behavioral_result.success or not behavioral_result.trials:
            logger.warning(f"No behavioral data to write for {bids_prefix}")
            return outputs

        paradigm = behavioral_result.paradigm

        # Convert to BIDS events
        bids_events = trials_to_bids_events(
            behavioral_result.trials,
            paradigm,
            eeg_events,
            sfreq,
        )

        # Write events.tsv
        events_path = self.output_dir / f"{bids_prefix}_events.tsv"
        write_events_tsv(bids_events, events_path)
        outputs["events_tsv"] = events_path

        # Get columns used for sidecar
        if bids_events:
            columns_used = list(bids_events[0].to_dict().keys())
        else:
            columns_used = ["onset", "duration", "trial_type"]

        # Write events.json sidecar
        json_path = self.output_dir / f"{bids_prefix}_events.json"
        write_events_json(json_path, paradigm, columns_used)
        outputs["events_json"] = json_path

        # Write detailed trials table
        trials_path = self.output_dir / f"{bids_prefix}_trials.tsv"
        write_trials_tsv(behavioral_result.trials, trials_path)
        outputs["trials_tsv"] = trials_path

        return outputs

    def write_from_eeg_events_only(
        self,
        events: np.ndarray,
        bids_prefix: str,
        sfreq: float = 500.0,
    ) -> Dict[str, Path]:
        """
        Write events.tsv from EEG events only (no behavioral data).

        Args:
            events: EEG events array [sample, 0, code]
            bids_prefix: BIDS filename prefix
            sfreq: Sampling frequency

        Returns:
            Dictionary of output file paths
        """
        outputs = {}

        if len(events) == 0:
            return outputs

        bids_events = []
        for evt in events:
            sample, _, code = evt
            onset = sample / sfreq
            bids_events.append(BIDSEvent(
                onset=onset,
                duration=0,
                trial_type="trigger",
                value=str(code),
            ))

        # Write events.tsv
        events_path = self.output_dir / f"{bids_prefix}_events.tsv"
        write_events_tsv(bids_events, events_path, include_extras=False)
        outputs["events_tsv"] = events_path

        # Write basic events.json
        json_path = self.output_dir / f"{bids_prefix}_events.json"
        write_events_json(json_path, "trigger", ["onset", "duration", "trial_type", "value"])
        outputs["events_json"] = json_path

        return outputs
