"""
Bad Channel Clustering Analysis

Detects spatial clustering of bad channels, which is important for:
1. Interpolation quality - clustered bad channels lead to poor interpolation
2. Source reconstruction - creates coverage gaps
3. Data quality assessment - indicates systematic issues

Author: EEGCPM Development Team
Date: 2025-12-02
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist
import mne


def compute_bad_channel_clustering(
    raw: mne.io.Raw,
    bad_channels: List[str],
    n_neighbors: int = 6,
    cluster_threshold: float = 0.5
) -> Dict:
    """
    Compute spatial clustering metrics for bad channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with montage set
    bad_channels : list of str
        List of bad channel names
    n_neighbors : int, default=6
        Number of nearest neighbors to consider
    cluster_threshold : float, default=0.5
        Proportion of bad neighbors to consider a channel "clustered"
        (0.5 = 50% or more neighbors are bad)

    Returns
    -------
    dict
        Clustering analysis results with keys:
        - n_bad_channels: Total number of bad channels
        - n_clustered_channels: Number of clustered bad channels
        - pct_clustered: Percentage of bad channels that are clustered
        - mean_pct_bad_neighbors: Mean % of bad neighbors across all bad channels
        - max_pct_bad_neighbors: Maximum % of bad neighbors
        - severity: 'none', 'mild', 'moderate', 'severe'
        - details: List of per-channel clustering info
        - warning: Warning message if clustering is problematic
    """
    if len(bad_channels) == 0:
        return {
            'n_bad_channels': 0,
            'n_clustered_channels': 0,
            'pct_clustered': 0.0,
            'mean_pct_bad_neighbors': 0.0,
            'max_pct_bad_neighbors': 0.0,
            'severity': 'none',
            'details': [],
            'clustered_channels': []
        }

    montage = raw.get_montage()
    if montage is None:
        return {'error': 'No montage available - cannot assess spatial clustering'}

    pos = montage.get_positions()
    ch_pos = pos['ch_pos']

    # Get 3D positions for all EEG channels
    all_channels = [ch for ch in raw.ch_names if ch in ch_pos]
    if len(all_channels) == 0:
        return {'error': 'No channels with position information'}

    positions = np.array([ch_pos[ch] for ch in all_channels])

    # Compute pairwise distance matrix
    dist_matrix = cdist(positions, positions, metric='euclidean')

    # Analyze each bad channel
    clustering_details = []

    for bad_ch in bad_channels:
        if bad_ch not in all_channels:
            continue

        bad_idx = all_channels.index(bad_ch)

        # Get distances to all other channels
        distances = dist_matrix[bad_idx]

        # Find K nearest neighbors (excluding self at index 0)
        nearest_idx = np.argsort(distances)[1:n_neighbors+1]
        nearest_channels = [all_channels[i] for i in nearest_idx]

        # Count how many neighbors are also bad
        bad_neighbors = [ch for ch in nearest_channels if ch in bad_channels]
        pct_bad_neighbors = 100 * len(bad_neighbors) / n_neighbors

        # Determine if this channel is in a cluster
        is_clustered = (len(bad_neighbors) / n_neighbors) >= cluster_threshold

        clustering_details.append({
            'channel': bad_ch,
            'n_bad_neighbors': len(bad_neighbors),
            'pct_bad_neighbors': pct_bad_neighbors,
            'nearest_channels': nearest_channels,
            'bad_neighbors': bad_neighbors,
            'is_clustered': is_clustered,
            'mean_distance_to_neighbors': np.mean(distances[nearest_idx])
        })

    # Compute summary statistics
    pct_bad_neighbors_all = [d['pct_bad_neighbors'] for d in clustering_details]
    clustered_channels = [d['channel'] for d in clustering_details if d['is_clustered']]
    n_clustered = len(clustered_channels)
    pct_clustered = 100 * n_clustered / len(bad_channels) if bad_channels else 0

    # Determine severity
    if pct_clustered == 0:
        severity = 'none'
    elif pct_clustered < 20:
        severity = 'mild'
    elif pct_clustered < 50:
        severity = 'moderate'
    else:
        severity = 'severe'

    # Generate warning message
    warning = None
    if severity == 'severe':
        warning = (
            f"Severe bad channel clustering detected: {n_clustered}/{len(bad_channels)} "
            f"({pct_clustered:.1f}%) bad channels are clustered. "
            "Interpolation will be unreliable in affected regions. "
            "Consider dropping clustered channels or excluding this subject."
        )
    elif severity == 'moderate':
        warning = (
            f"Moderate bad channel clustering: {n_clustered}/{len(bad_channels)} "
            f"({pct_clustered:.1f}%) bad channels are clustered. "
            "Interpolation quality may be reduced in clustered regions."
        )

    summary = {
        'n_bad_channels': len(bad_channels),
        'n_clustered_channels': n_clustered,
        'pct_clustered': pct_clustered,
        'mean_pct_bad_neighbors': np.mean(pct_bad_neighbors_all) if pct_bad_neighbors_all else 0,
        'max_pct_bad_neighbors': np.max(pct_bad_neighbors_all) if pct_bad_neighbors_all else 0,
        'severity': severity,
        'details': clustering_details,
        'clustered_channels': clustered_channels,
        'warning': warning
    }

    return summary


def visualize_channel_clustering(
    raw: mne.io.Raw,
    bad_channels: List[str],
    clustering_result: Optional[Dict] = None,
    n_neighbors: int = 6
) -> 'matplotlib.figure.Figure':
    """
    Create visualization of bad channel spatial distribution and clustering.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with montage
    bad_channels : list of str
        List of bad channel names
    clustering_result : dict, optional
        Pre-computed clustering result. If None, will compute it.
    n_neighbors : int, default=6
        Number of neighbors for clustering analysis

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with two subplots:
        - Left: All channels with bad marked
        - Right: Clustering visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    montage = raw.get_montage()
    if montage is None:
        raise ValueError("No montage available for visualization")

    pos = montage.get_positions()
    ch_pos = pos['ch_pos']

    # Compute clustering if not provided
    if clustering_result is None:
        clustering_result = compute_bad_channel_clustering(
            raw, bad_channels, n_neighbors=n_neighbors
        )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: All channels with bad marked
    for ch_name in raw.ch_names:
        if ch_name not in ch_pos:
            continue
        xyz = ch_pos[ch_name]
        x, y = xyz[0], xyz[1]

        if ch_name in bad_channels:
            ax1.plot(x, y, 'rx', markersize=10, markeredgewidth=2)
        else:
            ax1.plot(x, y, 'go', markersize=6, alpha=0.6)

    ax1.set_aspect('equal')
    ax1.set_title(
        f'Channel Layout - Bad Channels Marked\n'
        f'{len(bad_channels)}/{len(raw.ch_names)} channels marked bad',
        fontweight='bold', fontsize=11
    )
    ax1.set_xlabel('X (anterior-posterior)')
    ax1.set_ylabel('Y (left-right)')

    kept_patch = mpatches.Patch(color='green', label='Good channels')
    bad_patch = mpatches.Patch(color='red', label='Bad channels')
    ax1.legend(handles=[kept_patch, bad_patch], loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Clustering visualization
    if 'details' in clustering_result:
        for ch_name in raw.ch_names:
            if ch_name not in ch_pos:
                continue
            xyz = ch_pos[ch_name]
            x, y = xyz[0], xyz[1]

            # Find clustering info
            ch_info = next(
                (d for d in clustering_result['details'] if d['channel'] == ch_name),
                None
            )

            if ch_info:
                if ch_info['is_clustered']:
                    # Clustered bad - RED X
                    ax2.plot(x, y, 'rX', markersize=14, markeredgewidth=3,
                            markeredgecolor='darkred', markerfacecolor='red')
                else:
                    # Isolated bad - ORANGE diamond
                    ax2.plot(x, y, 'D', color='darkorange', markersize=10,
                            markeredgewidth=2, markeredgecolor='orangered')
            else:
                # Good channel - green circle
                ax2.plot(x, y, 'o', color='limegreen', markersize=7, alpha=0.7,
                        markeredgewidth=1, markeredgecolor='darkgreen')

        ax2.set_aspect('equal')

        severity_color = {
            'none': 'green',
            'mild': 'orange',
            'moderate': 'darkorange',
            'severe': 'red'
        }

        title_color = severity_color.get(clustering_result['severity'], 'black')

        ax2.set_title(
            f'Clustering Analysis - Severity: {clustering_result["severity"].upper()}\n'
            f'{clustering_result["n_clustered_channels"]}/{clustering_result["n_bad_channels"]} '
            f'({clustering_result["pct_clustered"]:.1f}%) channels are clustered',
            fontweight='bold', fontsize=11, color=title_color
        )
        ax2.set_xlabel('X (anterior-posterior)')
        ax2.set_ylabel('Y (left-right)')

        # Custom legend
        legend_elements = [
            Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
                   markeredgecolor='darkred', markersize=12, markeredgewidth=2,
                   label=f'Clustered bad (≥{int(0.5 * n_neighbors)}/{n_neighbors} bad neighbors)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='darkorange',
                   markeredgecolor='orangered', markersize=10, markeredgewidth=2,
                   label=f'Isolated bad (<{int(0.5 * n_neighbors)}/{n_neighbors} bad neighbors)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                   markeredgecolor='darkgreen', markersize=8, markeredgewidth=1,
                   label='Good channel')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def get_clustering_recommendation(clustering_result: Dict) -> str:
    """
    Get processing recommendation based on clustering severity.

    Parameters
    ----------
    clustering_result : dict
        Result from compute_bad_channel_clustering()

    Returns
    -------
    str
        Recommendation for handling the bad channels
    """
    severity = clustering_result.get('severity', 'none')
    n_clustered = clustering_result.get('n_clustered_channels', 0)
    clustered_channels = clustering_result.get('clustered_channels', [])

    if severity == 'none':
        return "No clustering detected. Proceed with standard interpolation."

    elif severity == 'mild':
        return (
            f"Mild clustering detected ({n_clustered} clustered channels). "
            "Standard interpolation should work, but monitor interpolation quality."
        )

    elif severity == 'moderate':
        return (
            f"Moderate clustering detected ({n_clustered} clustered channels). "
            f"Consider dropping these clustered channels instead of interpolating: "
            f"{', '.join(clustered_channels[:10])}"
            f"{'...' if len(clustered_channels) > 10 else ''}"
        )

    elif severity == 'severe':
        return (
            f"Severe clustering detected ({n_clustered} clustered channels). "
            f"STRONGLY RECOMMEND: Drop all clustered channels instead of interpolation. "
            f"Interpolation in these regions will be unreliable and may introduce artifacts. "
            f"Consider excluding this subject if too many channels remain after dropping."
        )

    return "Unknown severity level."
