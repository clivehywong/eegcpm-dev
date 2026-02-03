"""
Healthy Brain Network (HBN) data importer.

Converts HBN EEG data (MAT/CSV format) to BIDS-compatible structure.

HBN data structure:
    raw/
    └── NDAR*/
        ├── Behavioral/
        ├── EEG/raw/
        │   ├── csv_format/
        │   │   ├── RestingState_data.csv
        │   │   └── RestingState_event.csv
        │   └── mat_format/
        │       └── RestingState.mat
        └── Eyetracking/

Output BIDS structure:
    bids/
    ├── dataset_description.json
    ├── participants.tsv
    └── sub-NDAR*/
        └── ses-01/
            └── eeg/
                ├── sub-NDAR*_ses-01_task-rest_eeg.fif
                ├── sub-NDAR*_ses-01_task-rest_eeg.json
                └── sub-NDAR*_ses-01_task-rest_events.tsv
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eegcpm.plugins.importers.base import BaseImporter, ImportResult, ImportSummary
from eegcpm.plugins.importers.hbn_behavioral import (
    discover_behavioral_files,
    parse_behavioral_file,
    BehavioralResult,
)
from eegcpm.plugins.importers.hbn_events import BIDSEventsWriter

logger = logging.getLogger(__name__)

# HBN paradigm mapping to BIDS task names
HBN_PARADIGMS = {
    "RestingState": {"task": "rest", "trigger": 90},
    "vis_learn": {"task": "vislearn", "trigger": 91},
    "WISC_ProcSpeed": {"task": "wiscprocspeed", "trigger": 92},
    "SurrSupp_Block": {"task": "surroundsuppress", "triggers": [93, 97]},
    "SAIIT_2AFC": {"task": "contrastdetection", "triggers": [94, 95, 96]},
    "Video": {"task": "video", "triggers": [81, 82, 83, 84]},
}

# Video variant mapping (trigger code -> abbreviation)
VIDEO_VARIANTS = {
    81: "wk",   # Wimpy Kid
    82: "ff",   # Fun with Fractals
    83: "dm",   # Despicable Me
    84: "tp",   # The Present
}

# HBN channel information
HBN_CHANNEL_INFO = {
    "n_channels": 129,  # 128 EEG + 1 reference
    "sfreq": 500.0,
    "montage": "GSN-HydroCel-128",  # EGI 128-channel cap
}


class HBNImporter(BaseImporter):
    """
    Import Healthy Brain Network EEG data to BIDS format.

    Supports:
    - MAT format (preferred, contains full structure)
    - CSV format (fallback)
    - Multiple paradigms per subject
    - Event extraction from triggers
    """

    name = "hbn"
    description = "Healthy Brain Network EEG data importer"
    supported_formats = ["mat", "csv"]

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        paradigms: Optional[List[str]] = None,
        prefer_mat: bool = True,
        skip_existing: bool = True,
        overwrite: bool = False,
        include_behavioral: bool = True,
    ):
        """
        Initialize HBN importer.

        Args:
            source_dir: HBN raw data directory (containing NDAR* folders)
            output_dir: Output BIDS directory
            paradigms: List of paradigms to import (default: all)
            prefer_mat: Prefer MAT format over CSV
            skip_existing: Skip already converted subjects
            overwrite: Overwrite existing files
            include_behavioral: Include behavioral data in events.tsv
        """
        super().__init__(source_dir, output_dir, skip_existing, overwrite)
        self.paradigms = paradigms or list(HBN_PARADIGMS.keys())
        self.prefer_mat = prefer_mat
        self.include_behavioral = include_behavioral

    def discover_subjects(self) -> List[str]:
        """Discover all NDAR* subject directories."""
        subjects = []
        raw_dir = self.source_dir / "raw"

        if not raw_dir.exists():
            # Check if source_dir itself contains NDAR folders
            raw_dir = self.source_dir

        for subject_dir in sorted(raw_dir.glob("NDAR*")):
            if subject_dir.is_dir():
                # Check if EEG data exists
                eeg_dir = subject_dir / "EEG" / "raw"
                if eeg_dir.exists():
                    subjects.append(subject_dir.name)

        logger.info(f"Discovered {len(subjects)} subjects with EEG data")
        return subjects

    def _get_subject_source_dir(self, subject_id: str) -> Path:
        """Get source directory for a subject."""
        raw_dir = self.source_dir / "raw"
        if not raw_dir.exists():
            raw_dir = self.source_dir
        return raw_dir / subject_id

    def _find_paradigm_files(
        self,
        subject_id: str,
        paradigm: str,
    ) -> List[Tuple[Path, str, str]]:
        """
        Find data files for a paradigm.

        Handles variations like:
        - RestingState.mat (exact match)
        - SAIIT_2AFC_Block1.mat, SAIIT_2AFC_Block2.mat (multiple blocks)
        - Video-DM.mat, Video-FF.mat (multiple variants)
        - SurroundSupp_Block2.mat (alternate naming)

        Returns:
            List of (file_path, format, variant_name) tuples
        """
        subject_dir = self._get_subject_source_dir(subject_id)
        eeg_dir = subject_dir / "EEG" / "raw"
        results = []

        # Alternate naming patterns for paradigms
        paradigm_patterns = {
            "SurrSupp_Block": ["SurrSupp_Block", "SurroundSupp", "SurrSupp"],
            "SAIIT_2AFC": ["SAIIT_2AFC", "SAIIT"],
            "Video": ["Video"],
            "RestingState": ["RestingState", "Resting"],
            "vis_learn": ["vis_learn", "VisLearn", "vislearn"],
            "WISC_ProcSpeed": ["WISC_ProcSpeed", "WISC"],
        }

        patterns = paradigm_patterns.get(paradigm, [paradigm])

        def find_matching_files(directory: Path, suffix: str) -> List[Tuple[Path, str]]:
            """Find files matching any pattern variant."""
            matches = []
            if not directory.exists():
                return matches
            for pattern in patterns:
                # Exact match
                exact = directory / f"{pattern}{suffix}"
                if exact.exists():
                    matches.append((exact, pattern))
                # Pattern with suffix (Block1, -DM, etc.)
                for f in directory.glob(f"{pattern}*{suffix}"):
                    if f not in [m[0] for m in matches]:
                        # Extract variant name
                        variant = f.stem.replace(pattern, "").strip("_-")
                        matches.append((f, f"{pattern}_{variant}" if variant else pattern))
            return matches

        # Try MAT format first
        if self.prefer_mat:
            mat_dir = eeg_dir / "mat_format"
            for fpath, variant in find_matching_files(mat_dir, ".mat"):
                results.append((fpath, "mat", variant))

        # Try CSV format if no MAT found or not preferred
        if not results or not self.prefer_mat:
            csv_dir = eeg_dir / "csv_format"
            for fpath, variant in find_matching_files(csv_dir, "_data.csv"):
                if not any(r[2] == variant for r in results):  # Avoid duplicates
                    results.append((fpath, "csv", variant))

        # If still no results and MAT not preferred, try MAT as fallback
        if not results and not self.prefer_mat:
            mat_dir = eeg_dir / "mat_format"
            for fpath, variant in find_matching_files(mat_dir, ".mat"):
                results.append((fpath, "mat", variant))

        return results

    def _load_mat_data(self, mat_file: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Load EEG data from MAT file.

        Returns:
            Tuple of (data, events, sfreq)
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy required for MAT file loading")

        mat = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)

        # HBN MAT structure varies - try common keys
        if "EEG" in mat:
            eeg_struct = mat["EEG"]
            data = np.array(eeg_struct.data)
            sfreq = float(eeg_struct.srate)

            # Extract events
            events = []
            if hasattr(eeg_struct, "event"):
                for evt in np.atleast_1d(eeg_struct.event):
                    try:
                        # HBN uses 'sample' for latency, others might use 'latency'
                        if hasattr(evt, "sample"):
                            latency = int(evt.sample)
                        elif hasattr(evt, "latency"):
                            latency = int(evt.latency)
                        else:
                            continue

                        # Get event type - can be numeric or string
                        if hasattr(evt, "type"):
                            evt_type = evt.type
                            if isinstance(evt_type, str):
                                # Try to extract numeric code from string like "91  "
                                evt_type = evt_type.strip()
                                try:
                                    event_code = int(evt_type)
                                except ValueError:
                                    # Non-numeric event type, skip or use hash
                                    continue
                            else:
                                event_code = int(evt_type)
                        else:
                            continue

                        events.append([latency, 0, event_code])
                    except (ValueError, TypeError):
                        continue
            events = np.array(events) if events else np.zeros((0, 3), dtype=int)

        elif "data" in mat:
            data = np.array(mat["data"])
            sfreq = float(mat.get("srate", mat.get("sfreq", HBN_CHANNEL_INFO["sfreq"])))
            events = np.array(mat.get("events", [])).reshape(-1, 3) if "events" in mat else np.zeros((0, 3), dtype=int)
        else:
            raise ValueError(f"Unknown MAT structure in {mat_file}")

        # Ensure data is channels x samples
        if data.shape[0] > data.shape[1]:
            data = data.T

        return data, events, sfreq

    def _load_csv_data(
        self,
        data_file: Path,
        event_file: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Load EEG data from CSV files.

        Returns:
            Tuple of (data, events, sfreq)
        """
        import pandas as pd

        # Load data
        df = pd.read_csv(data_file)
        data = df.values.T  # Transpose to channels x samples
        sfreq = HBN_CHANNEL_INFO["sfreq"]

        # Load events if available
        events = np.zeros((0, 3), dtype=int)
        if event_file is None:
            event_file = data_file.parent / data_file.name.replace("_data.csv", "_event.csv")

        if event_file.exists():
            try:
                events_df = pd.read_csv(event_file)
                # Assume columns: latency, duration, type or similar
                if "latency" in events_df.columns and "type" in events_df.columns:
                    events = np.column_stack([
                        events_df["latency"].values.astype(int),
                        np.zeros(len(events_df), dtype=int),
                        events_df["type"].values.astype(int),
                    ])
            except Exception as e:
                logger.warning(f"Could not parse events from {event_file}: {e}")

        return data, events, sfreq

    def _create_mne_raw(
        self,
        data: np.ndarray,
        sfreq: float,
        events: np.ndarray,
    ):
        """Create MNE Raw object from data."""
        import mne

        n_channels = data.shape[0]

        # Create channel names
        ch_names = [f"E{i+1}" for i in range(n_channels)]

        # Exclude E129 if present (reference electrode with no physical location)
        # HydroCel 129 montage only has E1-E128 + Cz, not E129
        if n_channels == 129 and 'E129' in ch_names:
            # Drop the last channel (E129)
            data = data[:-1, :]
            ch_names = ch_names[:-1]
            n_channels = 128

        # Create info
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=["eeg"] * n_channels,
        )

        # HBN data is stored in microvolts (µV), MNE expects Volts
        # Scale by 1e-6 to convert µV to V
        data_volts = data * 1e-6

        # Create Raw object
        raw = mne.io.RawArray(data_volts, info, verbose=False)

        # Set montage if available
        try:
            montage = mne.channels.make_standard_montage(HBN_CHANNEL_INFO["montage"])
            # Map channel names if needed
            ch_mapping = {f"E{i+1}": montage.ch_names[i] for i in range(min(n_channels, len(montage.ch_names)))}
            raw.rename_channels(ch_mapping)
            raw.set_montage(montage, on_missing="warn")
        except Exception as e:
            logger.warning(f"Could not set montage: {e}")

        # Add events as annotations
        if len(events) > 0:
            onset = events[:, 0] / sfreq
            duration = np.zeros(len(events))
            description = [str(e) for e in events[:, 2]]
            annotations = mne.Annotations(onset, duration, description)
            raw.set_annotations(annotations)

        return raw, events

    def _find_and_parse_behavioral(
        self,
        subject_id: str,
        paradigm: str,
        variant_name: str,
        eeg_events: Optional[np.ndarray] = None,
        sfreq: float = 500.0,
    ) -> Optional[BehavioralResult]:
        """
        Find and parse behavioral data for a paradigm.

        Args:
            subject_id: Subject ID
            paradigm: Paradigm name
            variant_name: Variant name (e.g., "SAIIT_2AFC_Block1")
            eeg_events: EEG events array to find paradigm start trigger
            sfreq: Sampling frequency

        Returns:
            BehavioralResult or None if not found
        """
        subject_dir = self._get_subject_source_dir(subject_id)

        # Discover behavioral files
        beh_files = discover_behavioral_files(subject_dir, subject_id)

        # Find paradigm start trigger time from EEG events
        time_offset = 0.0
        if eeg_events is not None and len(eeg_events) > 0:
            paradigm_triggers = HBN_PARADIGMS.get(paradigm, {})
            start_triggers = paradigm_triggers.get("triggers", [])
            if "trigger" in paradigm_triggers:
                start_triggers = [paradigm_triggers["trigger"]] + start_triggers

            # Find first matching trigger
            for sample, _, code in eeg_events:
                if code in start_triggers:
                    time_offset = sample / sfreq
                    break

        # Try to match by paradigm and variant
        for mat_file, beh_paradigm, block in beh_files:
            # Check paradigm match
            if beh_paradigm != paradigm:
                continue

            # Check variant/block match
            if block:
                # e.g., variant_name="SAIIT_2AFC_Block1", block="Block1"
                if block.lower() not in variant_name.lower():
                    continue

            # Parse the behavioral file with time offset
            return self._parse_behavioral_with_offset(
                mat_file, subject_id, paradigm, block, time_offset
            )

        return None

    def _parse_behavioral_with_offset(
        self,
        mat_file: Path,
        subject_id: str,
        paradigm: str,
        block: Optional[str],
        time_offset: float,
    ) -> Optional[BehavioralResult]:
        """Parse behavioral file with time offset for EEG alignment."""
        from eegcpm.plugins.importers.hbn_behavioral import get_parser

        try:
            from scipy.io import loadmat
        except ImportError:
            return None

        parser = get_parser(paradigm, subject_id)
        if parser is None:
            return None

        try:
            mat_data = loadmat(str(mat_file), squeeze_me=True, struct_as_record=False)
            mat_data = {k: v for k, v in mat_data.items() if not k.startswith("_")}

            # Call parse with time_offset if supported
            if hasattr(parser.parse, "__code__") and "time_offset" in parser.parse.__code__.co_varnames:
                return parser.parse(mat_data, block, time_offset=time_offset)
            else:
                return parser.parse(mat_data, block)
        except Exception as e:
            logger.warning(f"Error parsing behavioral data: {e}")
            return None

    def import_subject(
        self,
        subject_id: str,
        **kwargs,
    ) -> ImportResult:
        """
        Import a single HBN subject.

        Args:
            subject_id: NDAR subject ID

        Returns:
            ImportResult with status and output files
        """
        import mne

        output_files = []
        errors = []
        warnings = []
        metadata = {"paradigms": {}}

        bids_id = self._to_bids_id(subject_id)
        subject_dir = self.output_dir / bids_id / "ses-01" / "eeg"
        subject_dir.mkdir(parents=True, exist_ok=True)

        paradigms_imported = 0

        for paradigm in self.paradigms:
            paradigm_files = self._find_paradigm_files(subject_id, paradigm)

            if not paradigm_files:
                continue

            # Process each variant/block for this paradigm
            for file_path, file_format, variant_name in paradigm_files:
                try:
                    # Load data based on format
                    if file_format == "mat":
                        data, events, sfreq = self._load_mat_data(file_path)
                    else:
                        data, events, sfreq = self._load_csv_data(file_path)

                    # Create MNE Raw object
                    raw, events = self._create_mne_raw(data, sfreq, events)

                    # Get BIDS task name - include variant if multiple blocks
                    task_info = HBN_PARADIGMS.get(paradigm, {"task": paradigm.lower()})
                    base_task_name = task_info["task"]

                    # Determine run number for tasks with multiple blocks/runs
                    run_number = None

                    if paradigm == "Video":
                        # For videos, use trigger code to determine variant
                        # Find first video trigger in events
                        for evt in events:
                            code = evt[2]
                            if code in VIDEO_VARIANTS:
                                variant_suffix = VIDEO_VARIANTS[code]
                                task_name = f"{base_task_name}{variant_suffix}"
                                break
                        else:
                            task_name = base_task_name
                    elif paradigm in ["SAIIT_2AFC", "SurrSupp_Block"]:
                        # Extract run number from variant name (Block1 -> run 1)
                        import re
                        match = re.search(r'[Bb]lock(\d+)', variant_name)
                        if match:
                            run_number = int(match.group(1))
                        task_name = base_task_name
                    else:
                        task_name = base_task_name

                    # Save as FIF with BIDS-compliant naming
                    if run_number:
                        fif_name = f"{bids_id}_ses-01_task-{task_name}_run-{run_number}_eeg.fif"
                    else:
                        fif_name = f"{bids_id}_ses-01_task-{task_name}_eeg.fif"
                    fif_path = subject_dir / fif_name
                    raw.save(fif_path, overwrite=self.overwrite, verbose=False)
                    output_files.append(fif_path)

                    # Create sidecar JSON
                    sidecar = {
                        "TaskName": task_name,
                        "TaskDescription": f"{paradigm} ({variant_name})" if variant_name != paradigm else paradigm,
                        "SamplingFrequency": sfreq,
                        "PowerLineFrequency": 60,  # US data
                        "EEGChannelCount": len(raw.ch_names),
                        "RecordingType": "continuous",
                        "RecordingDuration": raw.times[-1],
                        "SoftwareFilters": "n/a",
                        "Manufacturer": "EGI",
                        "ManufacturersModelName": "GES 400",
                        "CapManufacturer": "EGI",
                        "CapManufacturersModelName": "GSN-HydroCel-128",
                        "EEGReference": "vertex",
                    }
                    json_path = subject_dir / fif_name.replace(".fif", ".json")
                    with open(json_path, "w") as f:
                        json.dump(sidecar, f, indent=2)
                    output_files.append(json_path)

                    # Create events TSV (with behavioral data if available)
                    bids_prefix = fif_name.replace("_eeg.fif", "")
                    events_writer = BIDSEventsWriter(subject_dir)

                    behavioral_result = None
                    if self.include_behavioral:
                        # Find matching behavioral file and align with EEG events
                        behavioral_result = self._find_and_parse_behavioral(
                            subject_id, paradigm, variant_name,
                            eeg_events=events, sfreq=sfreq
                        )

                    if behavioral_result and behavioral_result.success:
                        # Write events with behavioral data
                        event_files = events_writer.write_behavioral_data(
                            behavioral_result,
                            bids_prefix,
                            eeg_events=events,
                            sfreq=sfreq,
                        )
                        output_files.extend(event_files.values())
                    elif len(events) > 0:
                        # Fall back to EEG events only
                        event_files = events_writer.write_from_eeg_events_only(
                            events, bids_prefix, sfreq
                        )
                        output_files.extend(event_files.values())

                    # Store metadata
                    paradigm_meta = {
                        "n_channels": len(raw.ch_names),
                        "sfreq": sfreq,
                        "duration": raw.times[-1],
                        "n_events": len(events),
                        "format": file_format,
                        "source_file": file_path.name,
                    }
                    if behavioral_result and behavioral_result.summary:
                        paradigm_meta["behavioral"] = behavioral_result.summary.to_dict()
                    metadata["paradigms"][variant_name] = paradigm_meta
                    paradigms_imported += 1
                    logger.debug(f"{subject_id}: Imported {variant_name} from {file_format}")

                except Exception as e:
                    errors.append(f"{variant_name}: {str(e)}")
                    logger.warning(f"{subject_id}/{variant_name}: {e}")

        if paradigms_imported == 0:
            return ImportResult(
                subject_id=subject_id,
                success=False,
                errors=errors or ["No paradigms found"],
                warnings=warnings,
            )

        metadata["n_paradigms"] = paradigms_imported
        return ImportResult(
            subject_id=subject_id,
            success=True,
            output_files=output_files,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def is_subject_complete(self, subject_id: str) -> bool:
        """Check if subject has already been converted."""
        bids_id = self._to_bids_id(subject_id)
        subject_dir = self.output_dir / bids_id / "ses-01" / "eeg"

        if not subject_dir.exists():
            return False

        # Check if at least one paradigm is converted
        fif_files = list(subject_dir.glob("*_eeg.fif"))
        return len(fif_files) > 0

    def create_dataset_description(self, **kwargs) -> Dict[str, Any]:
        """Create HBN-specific dataset description."""
        return {
            "Name": "Healthy Brain Network",
            "BIDSVersion": "1.8.0",
            "DatasetType": "raw",
            "License": "CC-BY-4.0",
            "Authors": [
                "Alexander, L.M.",
                "Escalera, J.",
                "Ai, L.",
                "et al.",
            ],
            "Acknowledgements": "Healthy Brain Network",
            "HowToAcknowledge": "Please cite: Alexander et al. (2017). An open resource for transdiagnostic research in pediatric mental health and learning disorders. Scientific Data.",
            "ReferencesAndLinks": [
                "http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/",
                "https://www.nature.com/articles/sdata2017181",
            ],
            "DatasetDOI": "10.1038/sdata.2017.181",
            "SourceDatasets": [
                {
                    "URL": "http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/",
                }
            ],
            "GeneratedBy": [
                {
                    "Name": "EEGCPM",
                    "Version": "0.1.0",
                    "Description": "Converted from HBN raw format using HBN importer",
                }
            ],
        }

    def create_participants_tsv(self) -> Path:
        """Create participants.tsv from converted subjects."""
        participants_file = self.output_dir / "participants.tsv"

        subjects = []
        for subject_dir in sorted(self.output_dir.glob("sub-*")):
            if subject_dir.is_dir():
                subjects.append(subject_dir.name)

        with open(participants_file, "w") as f:
            f.write("participant_id\n")
            for sub in subjects:
                f.write(f"{sub}\n")

        return participants_file


def convert_hbn_to_bids(
    source_dir: str,
    output_dir: str,
    paradigms: Optional[List[str]] = None,
    skip_existing: bool = True,
    max_subjects: Optional[int] = None,
    verbose: bool = True,
) -> ImportSummary:
    """
    Convenience function to convert HBN data to BIDS format.

    Args:
        source_dir: HBN raw data directory
        output_dir: Output BIDS directory
        paradigms: Paradigms to convert (default: all)
        skip_existing: Skip already converted subjects
        max_subjects: Maximum subjects to convert (for testing)
        verbose: Print progress

    Returns:
        ImportSummary with conversion results
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    importer = HBNImporter(
        source_dir=Path(source_dir),
        output_dir=Path(output_dir),
        paradigms=paradigms,
        skip_existing=skip_existing,
    )

    def progress(current, total, subject_id):
        if verbose:
            print(f"[{current}/{total}] Processing {subject_id}...")

    # Run import
    summary = importer.import_all(
        max_subjects=max_subjects,
        progress_callback=progress if verbose else None,
    )

    # Create BIDS metadata
    importer.save_dataset_description()
    importer.create_participants_tsv()

    # Save summary
    summary.save(Path(output_dir) / "import_summary.json")

    if verbose:
        print(f"\nConversion complete:")
        print(f"  Successful: {summary.successful}")
        print(f"  Failed: {summary.failed}")
        print(f"  Skipped: {summary.skipped}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HBN data to BIDS format")
    parser.add_argument("source_dir", help="HBN raw data directory")
    parser.add_argument("output_dir", help="Output BIDS directory")
    parser.add_argument("--paradigms", nargs="+", help="Paradigms to convert")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing subjects")
    parser.add_argument("--max-subjects", type=int, help="Maximum subjects to convert")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    convert_hbn_to_bids(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        paradigms=args.paradigms,
        skip_existing=not args.no_skip,
        max_subjects=args.max_subjects,
        verbose=not args.quiet,
    )
