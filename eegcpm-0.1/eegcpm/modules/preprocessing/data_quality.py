"""
Data Quality Detection for EEG Preprocessing

Detects and handles common data quality issues:
1. Flatline channels (zero or near-zero variance)
2. Bridged channels (near-perfect correlation)
3. Rank deficiency (linear dependencies)

These issues must be detected BEFORE ICA to prevent LAPACK errors.

Author: EEGCPM Development Team
Created: 2025-01-27
"""

from typing import Dict, List, Tuple
import numpy as np
import mne
from scipy.stats import pearsonr


def detect_flatline_channels(
    raw: mne.io.BaseRaw,
    variance_threshold: float = 1e-12,
    duration_threshold: float = 5.0,
) -> List[str]:
    """
    Detect channels with zero or near-zero variance (flatline/dead channels).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    variance_threshold : float
        Minimum variance threshold (default: 1e-12 V^2)
        Channels below this are considered flatline
    duration_threshold : float
        Minimum duration (seconds) of flatline to flag channel
        If entire recording is flatline, always flag

    Returns
    -------
    flatline_channels : list of str
        Names of flatline channels

    Notes
    -----
    Flatline channels can occur due to:
    - Disconnected electrodes
    - Poor electrode contact
    - Hardware failure
    - Recording errors

    These channels cause:
    - Rank deficiency (reduce effective rank)
    - Division by zero in normalization
    - LAPACK errors in ICA decomposition

    Examples
    --------
    >>> flatline = detect_flatline_channels(raw)
    >>> if flatline:
    ...     print(f"Dropping {len(flatline)} flatline channels: {flatline}")
    ...     raw.drop_channels(flatline)
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data = raw.get_data(picks=eeg_picks)
    ch_names = [raw.ch_names[i] for i in eeg_picks]

    flatline = []

    for idx, ch_name in enumerate(ch_names):
        ch_data = data[idx]

        # Compute variance
        var = np.var(ch_data)

        if var < variance_threshold:
            # Entire channel is flatline
            flatline.append(ch_name)
        else:
            # Check for extended flatline periods
            # Compute variance in sliding windows
            sfreq = raw.info['sfreq']
            window_samples = int(duration_threshold * sfreq)

            if len(ch_data) > window_samples:
                # Slide window and check variance
                n_windows = len(ch_data) - window_samples + 1
                has_flatline_period = False

                for i in range(0, n_windows, window_samples // 2):  # 50% overlap
                    window_var = np.var(ch_data[i:i+window_samples])
                    if window_var < variance_threshold:
                        has_flatline_period = True
                        break

                if has_flatline_period:
                    flatline.append(ch_name)

    return flatline


def detect_bridged_channels(
    raw: mne.io.BaseRaw,
    correlation_threshold: float = 0.98,
    min_good_channels: int = 10,
) -> List[str]:
    """
    Detect bridged channels (near-perfect correlation between channels).

    Channel bridging occurs when electrode gel creates conductive path
    between electrodes, causing them to record identical signals.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    correlation_threshold : float
        Correlation threshold (default: 0.98)
        Pairs above this are considered bridged
    min_good_channels : int
        Minimum number of channels to keep
        Will not drop channels if it reduces count below this

    Returns
    -------
    bridged_channels : list of str
        Names of bridged channels (second in each pair)

    Notes
    -----
    Bridged channels cause:
    - Rank deficiency (linear dependencies)
    - Inflated connectivity estimates
    - Biased ICA decomposition

    Detection strategy:
    - Compute pairwise Pearson correlations
    - Flag pairs with |r| > threshold
    - Drop second channel in each pair
    - Preserve channels with most unique information

    Examples
    --------
    >>> bridged = detect_bridged_channels(raw, correlation_threshold=0.98)
    >>> if bridged:
    ...     print(f"Dropping {len(bridged)} bridged channels: {bridged}")
    ...     raw.drop_channels(bridged)
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data = raw.get_data(picks=eeg_picks)
    ch_names = [raw.ch_names[i] for i in eeg_picks]

    n_channels = len(ch_names)

    if n_channels <= min_good_channels:
        # Too few channels, don't drop any
        return []

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data)

    # Find bridged pairs
    bridged_pairs = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if abs(corr_matrix[i, j]) > correlation_threshold:
                bridged_pairs.append((i, j, corr_matrix[i, j]))

    if not bridged_pairs:
        return []

    # Decide which channels to drop
    # Strategy: Drop channel involved in most bridges
    bridge_count = np.zeros(n_channels)
    for i, j, r in bridged_pairs:
        bridge_count[i] += 1
        bridge_count[j] += 1

    to_drop_indices = set()
    remaining_pairs = list(bridged_pairs)

    while remaining_pairs and (n_channels - len(to_drop_indices)) > min_good_channels:
        # Find channel involved in most remaining bridges
        current_counts = np.zeros(n_channels)
        for i, j, r in remaining_pairs:
            if i not in to_drop_indices:
                current_counts[i] += 1
            if j not in to_drop_indices:
                current_counts[j] += 1

        if current_counts.max() == 0:
            break

        # Drop channel with most bridges
        drop_idx = np.argmax(current_counts)
        to_drop_indices.add(drop_idx)

        # Remove pairs involving this channel
        remaining_pairs = [(i, j, r) for i, j, r in remaining_pairs
                          if i != drop_idx and j != drop_idx]

    bridged_channels = [ch_names[idx] for idx in sorted(to_drop_indices)]

    return bridged_channels


def compute_data_rank(
    raw: mne.io.BaseRaw,
    tol: float = 1e-6,
    tol_kind: str = 'relative',
) -> Dict[str, int]:
    """
    Compute effective rank of EEG data.

    Rank indicates number of independent signal dimensions.
    Rank < n_channels indicates linear dependencies (redundant channels).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    tol : float
        Tolerance for rank estimation (default: 1e-6)
    tol_kind : str
        'relative' or 'absolute'

    Returns
    -------
    rank_dict : dict
        Dictionary with channel type ranks
        e.g., {'eeg': 120, 'eog': 2}

    Notes
    -----
    Rank deficiency can be caused by:
    - Flatline channels (zero variance)
    - Bridged channels (perfect correlation)
    - Reference dependencies
    - Insufficient data samples

    For ICA:
    - Need rank ≥ n_components + 1
    - Recommended: rank ≥ 2 * n_components
    - If rank < 10, ICA not recommended

    Examples
    --------
    >>> rank_dict = compute_data_rank(raw)
    >>> eeg_rank = rank_dict['eeg']
    >>> print(f"EEG rank: {eeg_rank} (n_channels: {len(raw.ch_names)})")
    >>> if eeg_rank < 20:
    ...     print("Warning: Low rank, reduce ICA components")
    """
    rank_dict = mne.compute_rank(raw, tol=tol, tol_kind=tol_kind)
    return rank_dict


def detect_high_variance_channels(
    raw: mne.io.BaseRaw,
    deviation_threshold: float = 5.0,
) -> List[str]:
    """
    Detect channels with abnormally high or low amplitude using robust statistics.

    Uses the PREP pipeline approach (Bigdely-Shamlo et al., 2015) with IQR-based
    robust z-scores to identify channels that deviate significantly from the
    median channel amplitude.

    High amplitude channels typically show:
    - Line noise (60 Hz sine waves)
    - Movement artifacts
    - Electrical interference
    - Poor electrode contact with high impedance

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    deviation_threshold : float
        Z-score threshold for flagging channels (default: 5.0)
        Channels with |z| > threshold are considered bad

    Returns
    -------
    high_variance_channels : list of str
        Names of channels with abnormal amplitude

    Notes
    -----
    This method uses robust statistics (IQR-based) that are resistant to outliers:
    - Channel amplitude = IQR of channel data * 0.7413 (scaled to SD units)
    - Robust z-score = (amplitude - median) / robust_SD
    - robust_SD = IQR of amplitudes * 0.7413

    The 0.7413 factor converts IQR to SD assuming normal distribution.
    This approach is used by PREP pipeline and pyprep.

    High variance channels can corrupt:
    - RANSAC bad channel detection (skews correlation estimates)
    - ICA decomposition (dominates components)
    - Source localization (inflates activity estimates)

    These should be removed BEFORE RANSAC to prevent:
    - False negatives (noisy channels mask truly bad channels)
    - Unreliable spatial interpolation

    References
    ----------
    Bigdely-Shamlo, N., et al. (2015). The PREP pipeline: standardized
    preprocessing for large-scale EEG analysis. Frontiers in Neuroinformatics.

    Examples
    --------
    >>> high_var = detect_high_variance_channels(raw, deviation_threshold=5.0)
    >>> if high_var:
    ...     print(f"Dropping {len(high_var)} high variance channels: {high_var}")
    ...     raw.drop_channels(high_var)
    """
    from scipy.stats import iqr

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data = raw.get_data(picks=eeg_picks)
    ch_names = [raw.ch_names[i] for i in eeg_picks]

    if len(ch_names) == 0:
        return []

    # PREP pipeline approach: IQR-based robust amplitude estimation
    # IQR to SD conversion factor (assumes normal distribution)
    IQR_TO_SD = 0.7413

    # Compute channel amplitudes using IQR (robust to outliers)
    chan_amplitudes = iqr(data, axis=1) * IQR_TO_SD

    # Compute robust statistics of amplitudes
    amp_median = np.median(chan_amplitudes)
    amp_sd = iqr(chan_amplitudes) * IQR_TO_SD

    # Handle edge case where all channels have same amplitude
    if amp_sd < 1e-10:
        return []

    # Compute robust z-scores
    z_scores = (chan_amplitudes - amp_median) / amp_sd

    # Flag channels with |z| > threshold
    high_var_indices = np.where(np.abs(z_scores) > deviation_threshold)[0]
    high_var_channels = [ch_names[idx] for idx in high_var_indices]

    return high_var_channels


def detect_all_quality_issues(
    raw: mne.io.BaseRaw,
    flatline_variance_threshold: float = 1e-12,
    bridged_correlation_threshold: float = 0.98,
    deviation_threshold: float = 5.0,
    min_good_channels: int = 10,
) -> Dict[str, any]:
    """
    Comprehensive data quality assessment.

    Detects:
    1. Flatline channels
    2. Bridged channels
    3. High variance channels
    4. Data rank
    5. Recommended ICA component count
    6. Overall data quality status

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    flatline_variance_threshold : float
        Variance threshold for flatline detection
    bridged_correlation_threshold : float
        Correlation threshold for bridge detection
    deviation_threshold : float
        Z-score threshold for high variance detection (PREP pipeline method)
    min_good_channels : int
        Minimum channels to preserve

    Returns
    -------
    quality_report : dict
        {
            'flatline_channels': list,
            'bridged_channels': list,
            'high_variance_channels': list,
            'rank': dict,
            'n_channels_original': int,
            'n_channels_after_qc': int,
            'recommended_ica_components': int or None,
            'ica_feasible': bool,
            'issues': list of str (warnings),
        }

    Examples
    --------
    >>> quality = detect_all_quality_issues(raw)
    >>> print(f"Flatline: {len(quality['flatline_channels'])}")
    >>> print(f"Bridged: {len(quality['bridged_channels'])}")
    >>> print(f"High variance: {len(quality['high_variance_channels'])}")
    >>> print(f"Rank: {quality['rank']['eeg']}")
    >>> if not quality['ica_feasible']:
    ...     print("ICA not recommended for this data")
    """
    issues = []

    # Original channel count
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    n_original = len(eeg_picks)

    # Detect flatline channels
    flatline = detect_flatline_channels(
        raw,
        variance_threshold=flatline_variance_threshold
    )
    if flatline:
        issues.append(f"Found {len(flatline)} flatline channels")

    # Detect bridged channels
    bridged = detect_bridged_channels(
        raw,
        correlation_threshold=bridged_correlation_threshold,
        min_good_channels=min_good_channels,
    )
    if bridged:
        issues.append(f"Found {len(bridged)} bridged channels")

    # Detect high variance channels (using PREP pipeline robust method)
    high_variance = detect_high_variance_channels(
        raw,
        deviation_threshold=deviation_threshold,
    )
    if high_variance:
        issues.append(f"Found {len(high_variance)} high variance channels")

    # Compute rank
    rank_dict = compute_data_rank(raw)
    eeg_rank = rank_dict.get('eeg', n_original)

    # Estimate rank after dropping problematic channels
    n_after_qc = n_original - len(flatline) - len(bridged) - len(high_variance)
    expected_rank = min(eeg_rank, n_after_qc)

    # Determine if ICA is feasible
    min_ica_rank = 10  # Minimum rank for useful ICA
    ica_feasible = expected_rank >= min_ica_rank

    if ica_feasible:
        # Recommend components: rank - 1, capped at 20
        recommended_components = min(expected_rank - 1, 20)
    else:
        recommended_components = None
        issues.append(f"Rank ({expected_rank}) too low for ICA (need ≥ {min_ica_rank})")

    return {
        'flatline_channels': flatline,
        'bridged_channels': bridged,
        'high_variance_channels': high_variance,
        'rank': rank_dict,
        'n_channels_original': n_original,
        'n_channels_after_qc': n_after_qc,
        'recommended_ica_components': recommended_components,
        'ica_feasible': ica_feasible,
        'issues': issues,
    }


__all__ = [
    'detect_flatline_channels',
    'detect_bridged_channels',
    'detect_high_variance_channels',
    'compute_data_rank',
    'detect_all_quality_issues',
]
