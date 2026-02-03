# EEGCPM Preprocessing Exclusion Criteria

**Date**: 2025-11-28
**Purpose**: Document all exclusion criteria applied at each preprocessing stage
**Status**: COMPLETE

---

## Executive Summary

This document provides comprehensive documentation of all exclusion criteria applied during EEGCPM preprocessing, ensuring methodological transparency and reproducibility. Each preprocessing stage has specific criteria that determine which channels, segments, or components are excluded from analysis.

**Critical Finding**: Current approach removes bad channels entirely, leading to **variable channel counts across subjects (17-127 channels)**. This creates serious issues for source localization and CPM prediction. **Recommendation**: Switch to channel interpolation instead of removal.

---

## Overview: Preprocessing Pipeline

The EEGCPM preprocessing pipeline consists of 8 sequential stages:

```
1. Set Montage          → Spatial information for topographic plots
2. Data Quality Check   → Flatline & bridged channel detection
3. Bad Channel Detection → RANSAC, variance, correlation methods
4. Artifact Annotation  → Mark bad segments (not removed)
5. Re-reference         → Average reference
6. Resample (optional)  → Downsample if enabled
7. Filter               → Bandpass + notch
8. ICA                  → Independent component analysis
```

**Note**: ASR (Artifact Subspace Reconstruction) is disabled by default in current configuration.

---

## Stage 1: Montage Setting

**Location**: `eegcpm/modules/preprocessing/__init__.py:396-422`

### Exclusion Criteria

**None** - No channels are excluded at this stage.

### Purpose

Sets channel positions for spatial interpolation and topographic visualization. Required for:
- Bad channel interpolation (RANSAC method)
- Source reconstruction
- Topographic plots in QC reports

### Configuration

```yaml
montage:
  enabled: true
  type: "standard_1020"
  file: "/path/to/custom/montage.sfp"  # Optional custom montage
```

### Failure Handling

- If montage cannot be set, processing continues with warning
- RANSAC bad channel detection will be skipped (requires montage)
- Other methods (variance, correlation) remain available

---

## Stage 2: Data Quality Assessment

**Location**: `eegcpm/modules/preprocessing/data_quality.py`

### 2.1 Flatline Channel Detection

**Function**: `detect_flatline_channels()` (lines 21-100)

#### Exclusion Criteria

Channels are **DROPPED** (not interpolated) if:

1. **Global flatline**: Entire channel variance < `1e-12 V²` (essentially zero)

2. **Extended flatline period**: Any sliding window (default 5 seconds) has variance < `1e-12 V²`
   - Window size: `duration_threshold * sfreq` samples
   - Overlap: 50%

#### Configuration

```python
# In detect_flatline_channels() call
flatline_variance_threshold: 1e-12  # V² (variance threshold)
duration_threshold: 5.0             # seconds
```

#### Rationale

Flatline channels indicate:
- Disconnected electrodes
- Poor electrode contact
- Hardware failure
- Recording errors

These channels cause:
- **Rank deficiency** (reduce effective rank)
- **Division by zero** in normalization
- **LAPACK errors** in ICA decomposition

**Cannot be interpolated** - must be dropped.

---

### 2.2 Bridged Channel Detection

**Function**: `detect_bridged_channels()` (lines 103-205)

#### Exclusion Criteria

Channels are **DROPPED** if:

1. **Pairwise correlation** > `correlation_threshold` (default: 0.98)
   - Uses Pearson correlation
   - Absolute value checked: |r| > 0.98

2. **Greedy dropping strategy**:
   - For each bridged pair, drop the channel involved in **most bridges**
   - Iteratively remove channels until no bridges remain or minimum channel count reached

#### Configuration

```python
bridged_correlation_threshold: 0.98  # Correlation threshold
min_good_channels: 10                # Minimum channels to preserve
```

#### Rationale

Channel bridging occurs when electrode gel creates conductive path between electrodes, causing identical signals.

Bridged channels cause:
- **Rank deficiency** (linear dependencies)
- **Inflated connectivity estimates**
- **Biased ICA decomposition**

**Cannot be reliably interpolated** - must be dropped.

---

### 2.3 Rank Computation

**Function**: `compute_data_rank()` (lines 208-256)

#### No Exclusion Applied

Rank computation only **assesses** data quality:

```python
rank_dict = mne.compute_rank(raw, tol=1e-6, tol_kind='relative')
# Example: {'eeg': 120, 'eog': 2}
```

#### ICA Feasibility Check

**Function**: `detect_all_quality_issues()` (lines 259-359)

```python
min_ica_rank = 10  # Minimum rank for useful ICA

# ICA not feasible if:
ica_feasible = (rank >= 10)

# Recommended components if feasible:
recommended_components = min(rank - 1, 20)
```

#### Configuration

```python
min_ica_rank: 10  # Minimum rank threshold
```

#### Impact

If `rank < 10`:
- ICA is **skipped** entirely
- Warning added: "Rank ({rank}) too low for ICA"
- Preprocessing continues without ICA

---

## Stage 3: Bad Channel Detection

**Location**: `eegcpm/modules/preprocessing/bad_channels.py`

### Configuration (MNE-BIDS equivalent)

```yaml
bad_channels:
  auto_detect: true
  methods: ["ransac"]                    # RANSAC only (MNE-BIDS default)
  ransac_sample_prop: 0.25               # Use 25% of channels
  ransac_corr_threshold: 0.75            # Correlation threshold
  interpolate: true                      # Apply interpolation
  manual_bads: []                        # Optional manual list
```

### 3.1 RANSAC Method (Default)

**Function**: `_detect_by_ransac()` (lines 208-310)

#### Exclusion Criteria

Channels are flagged as bad if **consistently poorly predicted** by random channel subsets:

1. **Requires montage** - skipped if montage not set
2. **Requires ≥4 channels** - skipped if fewer
3. **RANSAC iterations**: 50 iterations sampling 25% of channels
4. **Prediction correlation** < `ransac_corr_threshold` (0.75) flagged as bad

#### Algorithm

```python
# Create 1-second overlapping epochs
epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, overlap=0.5)

# RANSAC parameters
ransac = Ransac(
    n_resample=50,                        # 50 iterations
    min_channels=0.25,                    # 25% of channels per sample
    min_corr=0.75,                        # Correlation threshold
    unbroken_time=True,
    n_jobs=1                              # Reproducibility
)

# Fit and detect
epochs_clean = ransac.fit_transform(epochs)
bad_channels = ransac.bad_chs_  # List of bad channel names
```

#### Configuration

```python
ransac_sample_prop: 0.25      # Proportion of channels to sample
ransac_corr_threshold: 0.75   # Minimum correlation for good channel
```

#### Rationale

RANSAC is **more robust** than variance/correlation methods for:
- Large channel arrays (64-128 channels)
- Datasets with multiple artifact types
- Spatial outlier detection

Reference: Jas et al. (2017), *NeuroImage*

---

### 3.2 Variance Method (Alternative)

**Function**: `_detect_by_variance()` (lines 162-184)

#### Exclusion Criteria

Channels flagged if **variance is outlier**:

1. Compute robust statistics:
   ```python
   median_var = np.median(variances)
   mad = np.median(np.abs(variances - median_var))
   robust_std = 1.4826 * mad  # MAD to SD conversion
   ```

2. Flag if **z-score** > `variance_threshold`:
   ```python
   z_scores = abs((variances - median_var) / robust_std)
   bad_if: z_scores > 5.0  # Default threshold
   ```

#### Configuration

```python
variance_threshold: 5.0  # Standard deviations from median
```

#### Rationale

Catches channels with:
- Extremely high variance (noise, movement)
- Extremely low variance (poor contact)

Uses MAD (Median Absolute Deviation) for robustness to outliers.

---

### 3.3 Correlation Method (Alternative)

**Function**: `_detect_by_correlation()` (lines 186-206)

#### Exclusion Criteria

Channels flagged if **poorly correlated with neighbors**:

1. Compute pairwise correlation matrix
2. For each channel, compute **median correlation** with all others
3. Flag if median correlation < `correlation_threshold`

```python
corr_matrix = np.corrcoef(data)
median_corrs = np.median(corr_matrix, axis=1)
bad_if: median_corrs < 0.4  # Default threshold
```

#### Configuration

```python
correlation_threshold: 0.4  # Minimum median correlation
```

#### Rationale

Poorly correlated channels indicate:
- Isolated noise
- Disconnected electrodes
- Different signal source (artifact)

---

### 3.4 Deviation Method (Alternative)

**Function**: `_detect_by_deviation()` (lines 312-337)

#### Exclusion Criteria

Channels flagged if **deviating from robust average**:

1. Compute robust average signal (median across channels at each timepoint)
2. Measure deviation of each channel from this average
3. Flag if z-score of deviation > `deviation_threshold`

```python
robust_avg = np.median(data, axis=0)  # Median at each timepoint
deviations = np.mean((data - robust_avg)**2, axis=1)
z_scores = abs((deviations - median_dev) / robust_std)
bad_if: z_scores > 5.0  # Default threshold
```

#### Configuration

```python
deviation_threshold: 5.0  # Standard deviations from robust mean
```

---

### 3.5 Bad Channel Handling

**Function**: `detect_and_interpolate_bad_channels()` (lines 351-477)

#### Current Behavior (CRITICAL ISSUE)

```python
if config.get('interpolate', True):
    raw.info['bads'] = bad_channels
    raw.interpolate_bads(reset_bads=True)  # Spherical spline interpolation
else:
    raw.drop_channels(bad_channels)        # REMOVES channels entirely
```

#### Zero-Variance Channel Handling

Channels with variance < `1e-20` **cannot be interpolated**:
- Always **dropped** regardless of `interpolate` setting
- Typically reference/ground channels or completely disconnected

#### Configuration

```yaml
bad_channels:
  interpolate: true   # CRITICAL: Set to true for source localization
```

#### Impact on Dataset

From current HBN batch processing (18 runs):

| Metric | Value |
|--------|-------|
| **Best case** | 127 channels (99% of montage) |
| **Worst case** | 17 channels (13% of montage) |
| **Mean** | ~88 channels (69% of montage) |
| **Runs with <50 channels** | 6/18 (33%) |

**This is CRITICAL** for:
- Source localization: Different forward models non-comparable
- CPM: Variable feature dimensions prevent model training

---

## Stage 4: Artifact Annotation

**Location**: `eegcpm/modules/preprocessing/artifacts.py`

### Configuration

```yaml
artifact_detection:
  enabled: true
  amplitude_threshold: 0.00015     # 150 µV
  gradient_threshold: 0.000075     # 75 µV/sample
  flatline_duration: 5.0           # seconds
  muscle_threshold: null           # Disabled for resting-state
  freq_muscle: [110, 140]          # Hz
  min_duration: 0.1                # seconds
```

### 4.1 Amplitude-Based Detection

**Function**: `_annotate_amplitude()` (lines 144-170)

#### Exclusion Criteria

Segments **annotated** (not removed) if **peak-to-peak amplitude** exceeds threshold:

```python
threshold = {'eeg': 150e-6}  # 150 µV
annot_amp, scores = mne.preprocessing.annotate_amplitude(
    raw, peak=threshold, min_duration=0.1
)
```

#### Configuration

```python
amplitude_threshold: 150e-6  # 150 µV (0.00015 V)
min_duration: 0.1            # Minimum segment duration (seconds)
```

#### Annotation Label

```
'BAD_amplitude'
```

#### Use

Annotations are **not removals** - they:
- Exclude segments from ICA fitting (`reject_by_annotation=True`)
- Can be used for epoch rejection in event-related analysis

---

### 4.2 Gradient-Based Detection

**Function**: `_annotate_gradient()` (lines 172-210)

#### Exclusion Criteria

Segments annotated if **voltage change per sample** exceeds threshold:

```python
gradient = np.abs(np.diff(data, axis=1))  # First derivative
bad_mask = np.any(gradient > 75e-6, axis=0)  # Any channel exceeds
```

#### Configuration

```python
gradient_threshold: 75e-6  # 75 µV/sample
min_duration: 0.1          # seconds
```

#### Annotation Label

```
'BAD_gradient'
```

#### Detects

- Muscle activity (rapid changes)
- Electrode pops
- Movement artifacts

---

### 4.3 Flatline Detection

**Function**: `_annotate_flatline()` (lines 212-259)

#### Exclusion Criteria

Segments annotated if **variance in sliding window** is near-zero:

```python
window_samples = int(5.0 * sfreq)  # 5-second windows
variances = np.var(window, axis=1)
bad_if: np.any(variances < 1e-20)  # Any channel flatline
```

#### Configuration

```python
flatline_duration: 5.0  # Window size (seconds)
min_duration: 0.1       # Minimum annotation duration
```

#### Annotation Label

```
'BAD_flatline'
```

#### Detects

- Disconnected electrodes
- Amplifier saturation
- Recording gaps

---

### 4.4 Muscle Artifact Detection

**Function**: `_annotate_muscle()` (lines 261-311)

#### Exclusion Criteria

Segments annotated if **high-frequency power** exceeds threshold:

```python
# Filter in muscle frequency range
raw_muscle = raw.filter(l_freq=110, h_freq=140)

# Compute envelope
muscle_power = np.mean(np.abs(raw_muscle.get_data()), axis=0)

# Z-score
z_scores = (muscle_power - mean) / std
bad_if: z_scores > 4.0  # Threshold
```

#### Configuration

```python
muscle_threshold: null        # Disabled by default (set to 4.0 to enable)
freq_muscle: [110, 140]       # Hz frequency range
min_duration: 0.1             # seconds
```

#### Annotation Label

```
'BAD_muscle'
```

#### Use Case

- Typically **disabled** for resting-state EEG
- Enable for task-based studies with movement

---

## Stage 5: Re-referencing

**Location**: `eegcpm/modules/preprocessing/__init__.py:476-493`

### Exclusion Criteria

**None** - No channels excluded.

### Configuration

```yaml
reference:
  type: "average"         # Average reference
  projection: true        # Use reversible projection
  exclude_bads: true      # Exclude bad channels from average
  channels: null          # Or specific channels for mastoid/custom
```

### Behavior

```python
if exclude_bads:
    # Bad channels already marked in raw.info['bads']
    # Average computed excluding these channels
    raw.set_eeg_reference(ref_channels='average', projection=True)
```

No channels are dropped - bad channels simply excluded from average computation.

---

## Stage 6: Resampling (Optional)

**Location**: `eegcpm/modules/preprocessing/__init__.py:317-324`

### Exclusion Criteria

**None** - No channels excluded.

### Configuration

```yaml
resample:
  enabled: false   # Disabled by default
  sfreq: 250       # Target sampling rate (Hz)
```

### Behavior

If enabled:
```python
raw.resample(250, verbose=False)  # Downsample to 250 Hz
```

Used to reduce computational cost for large datasets.

---

## Stage 7: Filtering

**Location**: `eegcpm/modules/preprocessing/__init__.py:495-524`

### Exclusion Criteria

**None** - No channels excluded.

### Configuration

```yaml
filter:
  l_freq: 1.0      # Highpass: 1.0 Hz (MNE-BIDS default)
  h_freq: 40.0     # Lowpass: 40.0 Hz
  notch_freq: null # Notch filter (e.g., [50, 60] for line noise)
```

### Behavior

```python
# Bandpass filter
raw.filter(l_freq=1.0, h_freq=40.0, picks='eeg')

# Notch filter (if specified)
if notch_freq:
    raw.notch_filter(freqs=notch_freq, picks='eeg')
```

Filters applied to all EEG channels - no exclusions.

---

## Stage 8: ICA (Independent Component Analysis)

**Location**: `eegcpm/modules/preprocessing/__init__.py:526-688`

### Configuration

```yaml
ica:
  enabled: true
  method: "extended-infomax"    # MNE-BIDS default
  n_components: null            # Auto: min(20, rank-1)
  auto_detect_artifacts: true   # EOG/ECG detection
  manual_exclude: []            # Manual component indices
  use_iclabel: false            # Requires mne-icalabel
  iclabel_threshold: 0.8
```

### 8.1 ICA Feasibility Check

**Lines**: 555-565

#### Exclusion Criteria

ICA **skipped entirely** if:

```python
if rank < 10:
    # ICA not feasible
    return raw, None, {'success': False, 'reason': 'insufficient_rank'}
```

From data quality report (Stage 2.3):
- Minimum rank required: 10
- If rank < 10, ICA cannot decompose signal reliably

---

### 8.2 Component Count Determination

**Lines**: 589-596

#### Criteria

```python
if quality_report['recommended_ica_components'] is not None:
    # Use rank-based recommendation
    n_components = min(rank - 1, 20)
elif n_components is None:
    # Default
    n_components = min(20, n_channels - 1)
```

**Rules**:
- Never exceed `rank - 1` (would be unstable)
- Cap at 20 components (sufficient for most artifacts)
- Ensure `rank ≥ 2 * n_components` for stability

---

### 8.3 Component Exclusion: Auto-Detection

**Lines**: 636-656

#### EOG Component Detection

```python
if len(eog_channels) > 0:
    eog_indices, scores = ica.find_bads_eog(raw)
    exclude_indices.extend(eog_indices)
```

**Method**: Cross-correlation between ICs and EOG channels

**Threshold**: Correlation > 0.9 (MNE default)

#### ECG Component Detection

```python
if len(ecg_channels) > 0:
    ecg_indices, scores = ica.find_bads_ecg(raw)
    exclude_indices.extend(ecg_indices)
```

**Method**: Cross-correlation between ICs and ECG channels or heart rate template

**Threshold**: Correlation > 0.25 (MNE default, lower because ECG harder to detect in EEG)

---

### 8.4 Component Exclusion: ICLabel (Optional)

**Lines**: 658-671

#### Exclusion Criteria

If `use_iclabel: true`:

```python
labels = label_components(raw, ica, method='iclabel')

artifact_labels = [
    'eye blink', 'heart beat', 'muscle artifact',
    'line noise', 'channel noise'
]

for idx, (label, prob) in enumerate(zip(labels['labels'], labels['y_pred_proba'])):
    if label in artifact_labels and prob > 0.8:
        exclude_indices.append(idx)
```

**Threshold**: 80% probability (configurable)

**Artifact Categories**:
- Eye blink
- Heart beat
- Muscle artifact
- Line noise
- Channel noise

**Brain category** preserved (not excluded)

---

### 8.5 Component Exclusion: Manual

**Lines**: 633

#### Criteria

Manually specified component indices always excluded:

```yaml
ica:
  manual_exclude: [0, 5, 12]  # Example: manually identified components
```

---

### 8.6 Final Component Removal

**Lines**: 674-677

```python
# Combine all exclusions (remove duplicates)
ica.exclude = sorted(list(set(exclude_indices)))

# Apply ICA (remove excluded components)
raw = ica.apply(raw)
```

**Total excluded components** = AUTO (EOG + ECG) + ICLabel + Manual

**Typical range**: 1-5 components excluded per subject

---

## Stage 9: ASR (Disabled)

**Location**: `eegcpm/modules/preprocessing/__init__.py:690-726`

### Configuration

```yaml
asr:
  enabled: false        # Disabled by default (not standard practice)
  cutoff: 20.0
  train_duration: 60
```

### Exclusion Criteria

**None** - ASR is disabled in standard configuration.

If enabled, ASR would **correct** (not exclude) artifact segments by subspace reconstruction.

---

## Summary of Exclusion Criteria

### Channels Excluded (Dropped)

| Stage | Method | Criteria | Count (Typical) |
|-------|--------|----------|-----------------|
| **Data Quality** | Flatline | Variance < 1e-12 V² | 1-3 |
| **Data Quality** | Bridged | Correlation > 0.98 | 0-100 (!!) |
| **Bad Channels** | RANSAC | Prediction correlation < 0.75 | 2-10 |
| **Bad Channels** | Variance | Z-score > 5.0 | 1-5 |
| **Bad Channels** | Correlation | Median r < 0.4 | 1-3 |
| **Bad Channels** | Zero-variance | Variance < 1e-20 | 0-2 |

**Total**: **17-127 channels dropped** (out of 128 in current HBN batch)

**CRITICAL ISSUE**: This creates variable channel counts preventing source localization and CPM.

---

### Segments Annotated (Not Removed)

| Stage | Method | Criteria | Typical % |
|-------|--------|----------|-----------|
| **Artifacts** | Amplitude | Peak > 150 µV | 5-15% |
| **Artifacts** | Gradient | Change > 75 µV/sample | 2-10% |
| **Artifacts** | Flatline | Variance < 1e-20 (5s window) | 1-5% |
| **Artifacts** | Muscle | Z-score > 4.0 (110-140 Hz) | 0% (disabled) |

**Use**: Excluded from ICA fitting and epoch creation

---

### ICA Components Excluded

| Stage | Method | Criteria | Typical Count |
|-------|--------|----------|---------------|
| **ICA** | Feasibility | Rank < 10 | ICA skipped |
| **ICA** | EOG detection | Correlation > 0.9 | 1-2 |
| **ICA** | ECG detection | Correlation > 0.25 | 0-1 |
| **ICA** | ICLabel | Prob(artifact) > 0.8 | 0-3 (if enabled) |
| **ICA** | Manual | User-specified | 0 (typically) |

**Total**: **1-5 components excluded**

**Typical ICA**: 20 components fitted, 2-3 excluded, 17-18 preserved

---

## Critical Issues & Recommendations

### Issue 1: Variable Channel Counts

**Problem**: Current implementation **drops** bad channels instead of interpolating.

**Impact**:
- Subject A: 127 channels
- Subject B: 17 channels (!)
- Cannot compare source estimates
- Cannot train CPM models (different feature dimensions)

**Solution**: **Always interpolate bad channels**

```yaml
bad_channels:
  interpolate: true  # CRITICAL: Set to true
```

---

### Issue 2: Excessive Bridging

**Problem**: Some subjects have 100+ bridged channels (block 2 in sub-NDARZX849GL3).

**Possible Causes**:
- Poor electrode preparation
- Excessive gel
- Net slippage during recording
- Recording artifacts

**Recommendations**:
1. Set **subject-level exclusion threshold**: Exclude subjects with >40 bad channels
2. Review bridged channel detection threshold (0.98 may be too sensitive)
3. Consider **interpolation** for bridged channels instead of dropping

```python
# Add to config
subject_exclusion:
  max_bad_channels: 40  # Exclude subjects exceeding this
```

---

### Issue 3: Merged ERP Epoch Rejection

**Problem**: 150 µV rejection threshold too strict → all epochs rejected.

**Current Solution**: Disabled rejection, using high-variance channel removal instead.

**Better Solution**: Use **Autoreject** for adaptive thresholds

```python
from autoreject import AutoReject
ar = AutoReject(n_interpolate=[1, 2, 4])
epochs_clean = ar.fit_transform(epochs)
```

---

### Issue 4: Documentation Gaps

**Problem**: No formal documentation of exclusion criteria until now.

**Solution**: This document! Update regularly as methods evolve.

**Action Items**:
1. Add exclusion criteria to module docstrings
2. Include in QC reports (show what was excluded and why)
3. Save exclusion metadata with preprocessed files

```python
# Add to ModuleResult metadata
metadata['exclusions'] = {
    'channels_dropped': bad_channels,
    'segments_annotated': len(annotations),
    'ica_components_excluded': ica.exclude,
    'epochs_rejected': n_rejected,
}
```

---

## References

### Methods

- **RANSAC**: Jas et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage*, 159, 417-429.

- **MAD**: Leys et al. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *Journal of Experimental Social Psychology*, 49(4), 764-766.

- **MNE-BIDS**: Appelhoff et al. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. *Journal of Open Source Software*, 4(44), 1896.

### MNE-Python Documentation

- Bad channel detection: https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html
- ICA artifact removal: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
- Artifact annotation: https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html

---

## Configuration Quick Reference

### Recommended Settings for Source Localization / CPM

```yaml
# CRITICAL: Always interpolate for consistent channel counts
bad_channels:
  auto_detect: true
  methods: ["ransac"]
  ransac_sample_prop: 0.25
  ransac_corr_threshold: 0.75
  interpolate: true  # CRITICAL

# Moderate artifact detection
artifact_detection:
  enabled: true
  amplitude_threshold: 150e-6
  gradient_threshold: 75e-6
  flatline_duration: 5.0
  muscle_threshold: null
  min_duration: 0.1

# Standard MNE-BIDS filtering
filter:
  l_freq: 1.0
  h_freq: 40.0
  notch_freq: null

# Extended-infomax ICA with auto-detection
ica:
  enabled: true
  method: "extended-infomax"
  n_components: null  # Auto
  auto_detect_artifacts: true
  use_iclabel: false
```

### Conservative Settings (Fewer Exclusions)

```yaml
bad_channels:
  methods: ["variance"]
  variance_threshold: 7.0  # Higher = fewer excluded
  interpolate: true

artifact_detection:
  amplitude_threshold: 200e-6  # Higher = fewer annotated
  gradient_threshold: 100e-6

ica:
  auto_detect_artifacts: false  # Manual review recommended
```

### Aggressive Settings (More Exclusions)

```yaml
bad_channels:
  methods: ["variance", "correlation", "deviation"]
  variance_threshold: 3.0  # Lower = more excluded
  correlation_threshold: 0.5
  interpolate: true

artifact_detection:
  amplitude_threshold: 100e-6  # Lower = more annotated
  gradient_threshold: 50e-6
  muscle_threshold: 3.0  # Enable muscle detection

ica:
  auto_detect_artifacts: true
  use_iclabel: true
  iclabel_threshold: 0.7  # Lower = more excluded
```

---

## Files Modified

This documentation references:
1. `eegcpm/modules/preprocessing/__init__.py` - Main preprocessing module
2. `eegcpm/modules/preprocessing/data_quality.py` - Flatline/bridged detection
3. `eegcpm/modules/preprocessing/bad_channels.py` - Bad channel detection methods
4. `eegcpm/modules/preprocessing/artifacts.py` - Artifact annotation
5. `config/preprocessing_mne_bids_equivalent.yaml` - Default configuration

---

## Changelog

- **2025-11-28**: Initial documentation created
  - Documented all 8 preprocessing stages
  - Identified critical issue with variable channel counts
  - Provided configuration recommendations

---

**Next Steps**:
1. Implement channel interpolation by default
2. Add subject-level exclusion criteria (>40 bad channels)
3. Integrate Autoreject for adaptive epoch rejection
4. Add exclusion metadata to QC reports
