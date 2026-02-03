#!/usr/bin/env python3
"""Regenerate QC report for a subject without reprocessing."""

import sys
from pathlib import Path
import mne

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eegcpm.modules.qc.preprocessed_qc import PreprocessedQC

def regenerate_qc(
    subject_id: str,
    pipeline: str = "standard",
    session: str = "01",
    task: str = "contdet",
    run: str = "1"
):
    """Regenerate QC report for an existing preprocessed subject."""

    # Build paths
    project_root = Path("/Volumes/Work/data/hbn")
    preproc_dir = (
        project_root / "derivatives" / "preprocessing" / pipeline /
        f"sub-{subject_id}" / f"ses-{session}" / f"task-{task}" / f"run-{run}"
    )

    if not preproc_dir.exists():
        print(f"Error: Preprocessing directory not found: {preproc_dir}")
        return

    # Load preprocessed data
    raw_file = preproc_dir / f"sub-{subject_id}_ses-{session}_task-{task}_run-{run}_preprocessed_raw.fif"
    ica_file = preproc_dir / f"sub-{subject_id}_ses-{session}_task-{task}_run-{run}_ica.fif"

    if not raw_file.exists():
        print(f"Error: Raw file not found: {raw_file}")
        return

    print(f"Loading preprocessed data for {subject_id}...")
    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)

    ica = None
    if ica_file.exists():
        ica = mne.preprocessing.read_ica(ica_file, verbose=False)

    # Load original raw if available (for before/after comparison)
    bids_dir = project_root / "bids" / f"sub-{subject_id}" / f"ses-{session}" / "eeg"
    raw_before_file = bids_dir / f"sub-{subject_id}_ses-{session}_task-{task}_run-{run}_eeg.fif"

    raw_before = None
    if raw_before_file.exists():
        print("Loading original raw for comparison...")
        raw_before = mne.io.read_raw_fif(raw_before_file, preload=True, verbose=False)

    # Initialize QC
    qc = PreprocessedQC(output_dir=preproc_dir, dpi=100)

    # Generate QC report
    print("Regenerating QC report...")
    result = qc.compute(
        data=raw,
        subject_id=subject_id,
        ica=ica,
        raw_before=raw_before,
        session_id=session,
        task_name=task
    )

    # Save HTML
    html_path = preproc_dir / f"sub-{subject_id}_ses-{session}_task-{task}_run-{run}_preprocessed_qc.html"
    qc.generate_html_report(result, save_path=html_path)

    print(f"✓ QC report saved to: {html_path}")
    print(f"\nOpen in browser: file://{html_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regenerate_qc.py <subject_id> [pipeline] [session] [task] [run]")
        print("Example: python regenerate_qc.py NDARAA306NT2 standard 01 contdet 1")
        sys.exit(1)

    subject_id = sys.argv[1]
    pipeline = sys.argv[2] if len(sys.argv) > 2 else "standard"
    session = sys.argv[3] if len(sys.argv) > 3 else "01"
    task = sys.argv[4] if len(sys.argv) > 4 else "contdet"
    run = sys.argv[5] if len(sys.argv) > 5 else "1"

    regenerate_qc(subject_id, pipeline, session, task, run)
