#!/usr/bin/env python3
"""
Test clustering-aware bad channel preprocessing.

Tests different clustering actions on a subject with known bad channels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mne
from eegcpm.modules.preprocessing import PreprocessingModule

# Configuration
BIDS_ROOT = Path("/Volumes/Work/data/hbn/bids")
SUBJECT_ID = "sub-NDARAB678VYW"  # Subject with moderate bad channels
TASK = "task-saiit2afcblock1"
OUTPUT_DIR = Path("/tmp/test_clustering_preprocessing")


def test_clustering_action(action: str, description: str):
    """Test a specific clustering action."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"Action: {action}")
    print(f"{'='*80}")

    # Load data
    eeg_file = BIDS_ROOT / SUBJECT_ID / "ses-01/eeg" / f"{SUBJECT_ID}_ses-01_{TASK}_eeg.fif"
    raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    raw.drop_channels(['E129'])

    print(f"Original: {len(raw.ch_names)} channels")

    # Create preprocessing config with clustering action
    config = {
        'montage': {
            'enabled': True,
            'type': 'GSN-HydroCel-128'
        },
        'bad_channels': {
            'auto_detect': True,
            'methods': ['variance', 'correlation'],
            'interpolate': True,
            'check_clustering': True,
            'clustering_action': action,
            'clustering_thresholds': {
                'moderate': 20.0,
                'severe': 50.0
            },
            'max_bad_channel_percent': 30.0  # Higher threshold for testing
        },
        'reference': {'type': 'average'},
        'filter': {'l_freq': 0.5, 'h_freq': 40.0},
        'ica': {'enabled': False}  # Disable ICA for faster testing
    }

    # Create output directory
    action_dir = OUTPUT_DIR / action
    action_dir.mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    module = PreprocessingModule(config, action_dir)
    result = module.process(raw, subject_id=SUBJECT_ID.replace('sub-', ''))

    if result.success:
        print(f"\n✓ SUCCESS")
        print(f"  Final channels: {len(result.outputs['data'].ch_names)}")

        # Display bad channel info
        bad_ch_info = result.metadata.get('bad_channels', {})
        print(f"\n  Bad Channel Summary:")
        print(f"    Detected: {bad_ch_info.get('n_bad', 0)}")
        print(f"    Dropped (clustered): {bad_ch_info.get('n_dropped_clustered', 0)}")
        print(f"    Interpolated: {bad_ch_info.get('n_interpolated', 0)}")
        print(f"    Quality: {bad_ch_info.get('quality_status', 'unknown')}")

        # Display clustering info
        clustering = bad_ch_info.get('clustering')
        if clustering:
            print(f"\n  Clustering Analysis:")
            print(f"    Severity: {clustering.get('severity', 'unknown')}")
            print(f"    Clustered: {clustering.get('n_clustered_channels', 0)}/{clustering.get('n_bad_channels', 0)} "
                  f"({clustering.get('pct_clustered', 0):.1f}%)")

        print(f"\n  {bad_ch_info.get('quality_message', '')}")

    else:
        print(f"\n✗ FAILED")
        print(f"  Errors: {result.errors}")

    return result


def main():
    """Test all clustering actions."""

    print("="*80)
    print("Clustering-Aware Preprocessing Tests")
    print("="*80)

    # Test 1: Adaptive (default) - should drop clustered on moderate/severe
    result1 = test_clustering_action('adaptive', 'Adaptive: mild=interp all, moderate/severe=drop clustered')

    # Test 2: Interpolate all - ignore clustering
    result2 = test_clustering_action('interpolate_all', 'Interpolate All: ignore clustering')

    # Test 3: Drop clustered - always drop clustered channels
    result3 = test_clustering_action('drop_clustered', 'Drop Clustered: always drop clustered, interp isolated')

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    if result1.success and result2.success and result3.success:
        n_ch1 = len(result1.outputs['data'].ch_names)
        n_ch2 = len(result2.outputs['data'].ch_names)
        n_ch3 = len(result3.outputs['data'].ch_names)

        print(f"  Adaptive:        {n_ch1} channels remaining")
        print(f"  Interpolate All: {n_ch2} channels remaining")
        print(f"  Drop Clustered:  {n_ch3} channels remaining")

        print(f"\n  Difference from standard interpolation:")
        print(f"    Adaptive:       {n_ch2 - n_ch1:+d} channels")
        print(f"    Drop Clustered: {n_ch2 - n_ch3:+d} channels")


if __name__ == "__main__":
    main()
