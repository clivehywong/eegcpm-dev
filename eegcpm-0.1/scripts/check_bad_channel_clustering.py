#!/usr/bin/env python3
"""
Check for spatial clustering of bad channels.

Clustered bad channels are problematic for:
1. Interpolation - needs nearby good channels for spline estimation
2. Source reconstruction - creates coverage gaps
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

BIDS_ROOT = Path("/Volumes/Work/data/hbn/bids")
SUBJECT_ID = "sub-NDARAA306NT2"  # Failed subject
TASK = "task-saiit2afcblock1"


def compute_bad_channel_clustering(raw, bad_channels, n_neighbors=6):
    """
    Compute spatial clustering metrics for bad channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with montage
    bad_channels : list
        List of bad channel names
    n_neighbors : int
        Number of nearest neighbors to consider

    Returns
    -------
    dict with clustering metrics
    """
    montage = raw.get_montage()
    if montage is None:
        return {'error': 'No montage available'}

    pos = montage.get_positions()
    ch_pos = pos['ch_pos']

    # Get 3D positions for all EEG channels
    all_channels = [ch for ch in raw.ch_names if ch in ch_pos]
    positions = np.array([ch_pos[ch] for ch in all_channels])

    # Compute distance matrix
    dist_matrix = cdist(positions, positions, metric='euclidean')

    # For each bad channel, find its nearest neighbors
    clustering_metrics = []

    for bad_ch in bad_channels:
        if bad_ch not in all_channels:
            continue

        bad_idx = all_channels.index(bad_ch)

        # Get distances to all other channels
        distances = dist_matrix[bad_idx]

        # Find K nearest neighbors (excluding self)
        nearest_idx = np.argsort(distances)[1:n_neighbors+1]
        nearest_channels = [all_channels[i] for i in nearest_idx]

        # Count how many neighbors are also bad
        bad_neighbors = [ch for ch in nearest_channels if ch in bad_channels]
        pct_bad_neighbors = 100 * len(bad_neighbors) / n_neighbors

        clustering_metrics.append({
            'channel': bad_ch,
            'n_bad_neighbors': len(bad_neighbors),
            'pct_bad_neighbors': pct_bad_neighbors,
            'nearest_channels': nearest_channels,
            'bad_neighbors': bad_neighbors,
            'is_clustered': pct_bad_neighbors > 50  # >50% neighbors are bad
        })

    # Overall clustering statistics
    if clustering_metrics:
        pct_bad_neighbors_all = [m['pct_bad_neighbors'] for m in clustering_metrics]
        n_clustered = sum(1 for m in clustering_metrics if m['is_clustered'])

        summary = {
            'n_bad_channels': len(bad_channels),
            'n_clustered_channels': n_clustered,
            'pct_clustered': 100 * n_clustered / len(bad_channels),
            'mean_pct_bad_neighbors': np.mean(pct_bad_neighbors_all),
            'max_pct_bad_neighbors': np.max(pct_bad_neighbors_all),
            'details': clustering_metrics
        }
    else:
        summary = {'error': 'No bad channels with valid positions'}

    return summary


def visualize_bad_channel_clustering(raw, bad_channels):
    """Create visualization of bad channel spatial distribution."""
    montage = raw.get_montage()
    if montage is None:
        print("No montage available for visualization")
        return

    pos = montage.get_positions()
    ch_pos = pos['ch_pos']

    # Create 2D projection (top view)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: All channels with bad marked
    for ch_name in raw.ch_names:
        if ch_name not in ch_pos:
            continue
        xyz = ch_pos[ch_name]
        x, y = xyz[0], xyz[1]

        if ch_name in bad_channels:
            ax1.plot(x, y, 'rx', markersize=10, markeredgewidth=2, label='Bad' if ch_name == bad_channels[0] else '')
        else:
            ax1.plot(x, y, 'go', markersize=6, alpha=0.6, label='Good' if ch_name == raw.ch_names[0] else '')

    ax1.set_aspect('equal')
    ax1.set_title('Channel Layout - Bad Channels Marked', fontweight='bold')
    ax1.set_xlabel('X (anterior-posterior)')
    ax1.set_ylabel('Y (left-right)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Clustering visualization
    # Compute clustering
    clustering = compute_bad_channel_clustering(raw, bad_channels)

    if 'details' in clustering:
        for ch_name in raw.ch_names:
            if ch_name not in ch_pos:
                continue
            xyz = ch_pos[ch_name]
            x, y = xyz[0], xyz[1]

            # Find clustering info for this channel
            ch_info = next((m for m in clustering['details'] if m['channel'] == ch_name), None)

            if ch_info:
                if ch_info['is_clustered']:
                    # Clustered bad channel - RED with X marker
                    ax2.plot(x, y, 'rX', markersize=14, markeredgewidth=3,
                            markeredgecolor='darkred', markerfacecolor='red')
                else:
                    # Isolated bad channel - ORANGE with diamond marker
                    ax2.plot(x, y, 'D', color='darkorange', markersize=10,
                            markeredgewidth=2, markeredgecolor='orangered')
            else:
                # Good channel - green circle
                ax2.plot(x, y, 'o', color='limegreen', markersize=7, alpha=0.7,
                        markeredgewidth=1, markeredgecolor='darkgreen')

        ax2.set_aspect('equal')
        ax2.set_title(
            f'Clustering Analysis\n'
            f'{clustering["n_clustered_channels"]}/{clustering["n_bad_channels"]} '
            f'({clustering["pct_clustered"]:.1f}%) are clustered',
            fontweight='bold'
        )
        ax2.set_xlabel('X (anterior-posterior)')
        ax2.set_ylabel('Y (left-right)')

        # Create custom legend with proper markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
                   markeredgecolor='darkred', markersize=12, markeredgewidth=2,
                   label='Clustered bad (>50% bad neighbors)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='darkorange',
                   markeredgecolor='orangered', markersize=10, markeredgewidth=2,
                   label='Isolated bad (<50% bad neighbors)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                   markeredgecolor='darkgreen', markersize=8, markeredgewidth=1,
                   label='Good channel')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Check bad channel clustering for problematic subject."""

    print("=" * 80)
    print("Bad Channel Clustering Analysis")
    print("=" * 80)
    print(f"\nSubject: {SUBJECT_ID}")

    # Load data
    eeg_file = BIDS_ROOT / SUBJECT_ID / "ses-01/eeg" / f"{SUBJECT_ID}_ses-01_{TASK}_eeg.fif"

    if not eeg_file.exists():
        print(f"ERROR: File not found: {eeg_file}")
        return

    print(f"Loading: {eeg_file.name}")
    raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    raw.drop_channels(['E129'])

    print(f"Channels: {len(raw.ch_names)}")

    # Simulate bad channel detection (we need to actually run it)
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
            'enabled': True,
            'method': 'spline'
        }
    }

    # Detect bad channels
    print("\nDetecting bad channels...")
    raw_processed, bad_channels = detect_and_interpolate_bad_channels(raw, config, copy=True)

    print(f"\nFound {len(bad_channels)} bad channels:")
    print(bad_channels[:20], '...' if len(bad_channels) > 20 else '')

    # Analyze clustering
    print("\nAnalyzing spatial clustering...")
    clustering = compute_bad_channel_clustering(raw, bad_channels, n_neighbors=6)

    if 'error' in clustering:
        print(f"Error: {clustering['error']}")
        return

    print(f"\nClustering Summary:")
    print(f"  Total bad channels: {clustering['n_bad_channels']}")
    print(f"  Clustered bad channels: {clustering['n_clustered_channels']} ({clustering['pct_clustered']:.1f}%)")
    print(f"  Mean % bad neighbors: {clustering['mean_pct_bad_neighbors']:.1f}%")
    print(f"  Max % bad neighbors: {clustering['max_pct_bad_neighbors']:.1f}%")

    # Show most clustered channels
    print(f"\nMost clustered bad channels:")
    sorted_details = sorted(clustering['details'], key=lambda x: x['pct_bad_neighbors'], reverse=True)
    for detail in sorted_details[:10]:
        print(f"  {detail['channel']}: {detail['n_bad_neighbors']}/{6} bad neighbors ({detail['pct_bad_neighbors']:.0f}%)")

    # Visualize
    print("\nGenerating visualization...")
    fig = visualize_bad_channel_clustering(raw, bad_channels)

    output_path = Path("/tmp/bad_channel_clustering.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
