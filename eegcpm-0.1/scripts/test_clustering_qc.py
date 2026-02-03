#!/usr/bin/env python3
"""
Test bad channel clustering integration in QC reports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
from eegcpm.modules.qc.preprocessed_qc import PreprocessedQC

# Configuration
BIDS_ROOT = Path("/Volumes/Work/data/hbn/bids")
SUBJECT_ID = "sub-NDARAA306NT2"  # Subject with many bad channels
TASK = "task-saiit2afcblock1"
OUTPUT_DIR = Path("/tmp/test_clustering_qc")

def main():
    """Test clustering integration in QC."""

    print("=" * 80)
    print("Testing Bad Channel Clustering in QC Reports")
    print("=" * 80)

    # Load data
    eeg_file = BIDS_ROOT / SUBJECT_ID / "ses-01/eeg" / f"{SUBJECT_ID}_ses-01_{TASK}_eeg.fif"

    if not eeg_file.exists():
        print(f"ERROR: File not found: {eeg_file}")
        return

    print(f"\nLoading: {eeg_file.name}")
    raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    raw.drop_channels(['E129'])
    print(f"  Channels: {len(raw.ch_names)}, Duration: {raw.times[-1]:.1f}s")

    # Simulate bad channel detection
    from eegcpm.modules.preprocessing.bad_channels import detect_and_interpolate_bad_channels

    config = {
        'enabled': True,
        'detection_methods': [
            {'flatline': True},
            {'variance': True}
        ],
        'flatline_duration_s': 5.0,
        'variance_threshold': 3.0,
        'interpolation': {
            'enabled': False  # Don't interpolate for this test
        }
    }

    print("\nDetecting bad channels...")
    raw_before = raw.copy()
    raw_processed, bad_channels = detect_and_interpolate_bad_channels(raw, config, copy=True)
    print(f"Found {len(bad_channels)} bad channels")

    # Create QC report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qc = PreprocessedQC(OUTPUT_DIR, dpi=100)

    print("\nGenerating QC report with clustering analysis...")
    result = qc.compute(
        data=raw_processed,
        raw_before=raw_before,
        subject_id=SUBJECT_ID.replace('sub-', ''),
        removed_channels={ch: 'bad' for ch in bad_channels}
    )

    # Save HTML report
    html_output = qc.generate_html_report(result, save_path=OUTPUT_DIR / f"{SUBJECT_ID}_qc.html")

    print(f"\n{'='*60}")
    print("QC Report Generated Successfully!")
    print(f"{'='*60}")
    print(f"\nOutput: {OUTPUT_DIR / f'{SUBJECT_ID}_qc.html'}")

    # Display clustering metrics
    print("\nClustering Metrics:")
    for metric in result.metrics:
        if 'Cluster' in metric.name:
            print(f"  {metric.name}: {metric.value} {metric.unit} [{metric.status}]")

    # Display recommendations
    print("\nNotes:")
    for note in result.notes:
        if 'Cluster' in note or 'cluster' in note:
            print(f"  - {note}")

if __name__ == "__main__":
    main()
