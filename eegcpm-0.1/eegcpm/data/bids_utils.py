"""BIDS utilities for file discovery and naming conventions.

This module handles both legacy and BIDS-compliant naming:
- Legacy: sub-ID_ses-01_task-saiit2afcblock1_eeg.fif
- BIDS:   sub-ID_ses-01_task-saiit_run-01_eeg.fif
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import re


@dataclass
class BIDSFile:
    """BIDS file information."""
    path: Path
    subject: str
    session: str
    task: str
    run: Optional[str] = None
    suffix: str = "eeg"
    extension: str = ".fif"

    @property
    def is_legacy_naming(self) -> bool:
        """Check if this uses legacy naming (no run, task includes block/run info)."""
        return self.run is None and any(
            substr in self.task.lower()
            for substr in ['block', 'run', '2afc']
        )

    def to_bids_compliant(self) -> 'BIDSFile':
        """Convert legacy naming to BIDS-compliant naming."""
        if not self.is_legacy_naming:
            return self

        # Extract task and run from legacy task name
        task, run = parse_legacy_task_name(self.task)

        return BIDSFile(
            path=self.path,
            subject=self.subject,
            session=self.session,
            task=task,
            run=run,
            suffix=self.suffix,
            extension=self.extension
        )

    def get_bids_filename(self) -> str:
        """Get BIDS-compliant filename."""
        parts = [f"sub-{self.subject}"]

        if self.session:
            parts.append(f"ses-{self.session}")

        parts.append(f"task-{self.task}")

        if self.run:
            parts.append(f"run-{self.run}")

        parts.append(self.suffix)

        return "_".join(parts) + self.extension


def parse_legacy_task_name(task_name: str) -> Tuple[str, str]:
    """
    Parse legacy task name to extract task and run.

    Examples:
        saiit2afcblock1 -> (saiit, 01)
        saiit2afcblock2 -> (saiit, 02)
        surrsuppblockblock1 -> (surrsupp, 01)
        rest -> (rest, 01)

    Parameters
    ----------
    task_name : str
        Legacy task name

    Returns
    -------
    tuple
        (task, run) where run is zero-padded
    """
    # Pattern: task name + optional suffix + block/run number
    patterns = [
        # saiit2afcblock1 -> saiit, 1
        (r'^(saiit)2afcblock(\d+)$', r'\1', r'\2'),
        # surrsuppblockblock1 -> surrsupp, 1
        (r'^(surrsupp)blockblock(\d+)$', r'\1', r'\2'),
        # Generic: taskblock1 -> task, 1
        (r'^(.+)block(\d+)$', r'\1', r'\2'),
        # Generic: taskrun1 -> task, 1
        (r'^(.+)run(\d+)$', r'\1', r'\2'),
    ]

    for pattern, task_group, run_group in patterns:
        match = re.match(pattern, task_name, re.IGNORECASE)
        if match:
            task = match.group(1)
            run = match.group(2).zfill(2)  # Zero-pad to 2 digits
            return task, run

    # No pattern matched - assume single run
    return task_name, "01"


def parse_bids_filename(filename: str) -> Optional[BIDSFile]:
    """
    Parse BIDS filename to extract components.

    Handles both:
    - BIDS: sub-ID_ses-01_task-saiit_run-01_eeg.fif
    - Legacy: sub-ID_ses-01_task-saiit2afcblock1_eeg.fif

    Parameters
    ----------
    filename : str
        BIDS filename (with or without path)

    Returns
    -------
    BIDSFile or None
        Parsed file info, or None if not BIDS format
    """
    filename = Path(filename).name

    # Remove extension
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    ext = '.' + ext if ext else ''

    # Parse entities using regex
    entities = {}

    # Required: subject
    match = re.search(r'sub-([a-zA-Z0-9]+)', name)
    if not match:
        return None
    entities['subject'] = match.group(1)

    # Optional: session
    match = re.search(r'ses-([a-zA-Z0-9]+)', name)
    entities['session'] = match.group(1) if match else None

    # Required: task
    match = re.search(r'task-([a-zA-Z0-9]+)', name)
    if not match:
        return None
    entities['task'] = match.group(1)

    # Optional: run
    match = re.search(r'run-([a-zA-Z0-9]+)', name)
    entities['run'] = match.group(1) if match else None

    # Suffix (last entity before extension)
    parts = name.split('_')
    suffix = parts[-1] if parts else 'eeg'

    return BIDSFile(
        path=Path(filename),
        subject=entities['subject'],
        session=entities['session'],
        task=entities['task'],
        run=entities['run'],
        suffix=suffix,
        extension=ext
    )


def find_bids_files(
    bids_root: Path,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    suffix: str = "eeg",
    extension: str = ".fif"
) -> List[BIDSFile]:
    """
    Find BIDS files matching criteria.

    Handles both legacy and BIDS-compliant naming.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root
    subject : str, optional
        Subject ID (without 'sub-' prefix)
    session : str, optional
        Session ID (without 'ses-' prefix)
    task : str, optional
        Task name (without 'task-' prefix)
    run : str, optional
        Run number (without 'run-' prefix)
    suffix : str
        File suffix (default: 'eeg')
    extension : str
        File extension (default: '.fif')

    Returns
    -------
    List[BIDSFile]
        List of matching BIDS files
    """
    bids_root = Path(bids_root)
    results = []

    # Build search pattern
    pattern_parts = []

    if subject:
        pattern_parts.append(f"sub-{subject}")
        subject_pattern = f"sub-{subject}"
    else:
        subject_pattern = "sub-*"

    # Search in subject directories
    for subject_dir in sorted(bids_root.glob(subject_pattern)):
        if not subject_dir.is_dir():
            continue

        # Look for session directories or directly in subject
        search_dirs = []

        if session:
            ses_dir = subject_dir / f"ses-{session}"
            if ses_dir.exists():
                search_dirs.append(ses_dir)
        else:
            # Look for all session directories
            ses_dirs = list(subject_dir.glob("ses-*"))
            if ses_dirs:
                search_dirs.extend(ses_dirs)
            else:
                # No session structure
                search_dirs.append(subject_dir)

        for search_dir in search_dirs:
            # Look in eeg subdirectory
            eeg_dir = search_dir / "eeg"
            if not eeg_dir.exists():
                continue

            # Find all EEG files
            for file_path in eeg_dir.glob(f"*{suffix}{extension}"):
                bids_file = parse_bids_filename(file_path.name)
                if not bids_file:
                    continue

                bids_file.path = file_path

                # Apply filters
                if task and bids_file.task != task:
                    # Check if legacy naming matches
                    bids_compliant = bids_file.to_bids_compliant()
                    if bids_compliant.task != task:
                        continue

                if run and bids_file.run != run:
                    continue

                results.append(bids_file)

    return sorted(results, key=lambda x: (x.subject, x.session or '', x.task, x.run or ''))


def find_subject_runs(
    bids_root: Path,
    subject: str,
    task: str,
    session: Optional[str] = None
) -> List[BIDSFile]:
    """
    Find all runs for a specific subject and task.

    Handles legacy naming by converting to BIDS-compliant.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root
    subject : str
        Subject ID
    task : str
        Task name (BIDS-compliant, e.g., 'saiit' not 'saiit2afcblock1')
    session : str, optional
        Session ID

    Returns
    -------
    List[BIDSFile]
        List of runs, sorted by run number
    """
    # Find all files for this subject/task
    files = find_bids_files(
        bids_root,
        subject=subject,
        session=session,
        task=None  # Don't filter by task yet
    )

    # Convert to BIDS-compliant and filter
    runs = []
    for f in files:
        bids_file = f.to_bids_compliant()
        if bids_file.task == task:
            runs.append(bids_file)

    # Sort by run number
    return sorted(runs, key=lambda x: x.run or '01')


def get_task_runs_summary(bids_root: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Get summary of all tasks and runs in dataset.

    Returns
    -------
    dict
        Nested dict: {subject: {task: [runs]}}
        For single-run tasks (no run entity), run list will be ['01']
    """
    summary = {}

    all_files = find_bids_files(bids_root)

    for f in all_files:
        bids_file = f.to_bids_compliant()

        if bids_file.subject not in summary:
            summary[bids_file.subject] = {}

        if bids_file.task not in summary[bids_file.subject]:
            summary[bids_file.subject][bids_file.task] = []

        run = bids_file.run if bids_file.run else '01'  # Default to 01 for single-run tasks
        summary[bids_file.subject][bids_file.task].append(run)

    # Sort and deduplicate runs
    for subject in summary:
        for task in summary[subject]:
            summary[subject][task] = sorted(set(summary[subject][task]))

    return summary
