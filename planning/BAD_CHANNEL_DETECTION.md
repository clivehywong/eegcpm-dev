# Bad Channel Detection & Clustering

**Status**: Implemented
**Location**: `eegcpm/modules/preprocessing/channel_clustering.py`

---

## Overview

Comprehensive bad channel detection and spatial clustering analysis system that identifies problematic channels and determines optimal handling strategies (interpolate vs drop) based on spatial proximity patterns.

### Why Clustering Matters

**Problems with Clustered Bad Channels:**

1. **Interpolation Quality**: Spline interpolation requires nearby good channels as references. When bad channels cluster together, there are insufficient good neighbors for reliable reconstruction.

2. **Source Reconstruction**: Clustered gaps in electrode coverage create blind spots in source localization, particularly affecting regions under the cluster.

3. **Data Quality Indicator**: High clustering suggests systematic issues (electrode cap fit, skin impedance, amplifier malfunction) rather than random channel failures.

---

## Detection Methods

### Tier 1: Automatic Removal (Pre-RANSAC)

**Flatline Channels**
- **Detection**: Variance < 1e-12 V²
- **Cause**: Disconnected electrodes, hardware failure
- **Action**: **DROP** immediately
- **Rationale**: Zero variance breaks ICA and source localization

**Bridged Channels**
- **Detection**: Correlation > 0.98 between channel pairs
- **Cause**: Electrode gel creating conductive bridge
- **Action**: **DROP** second channel in pair
- **Rationale**: Linear dependency reduces rank, biases ICA

**High Variance Channels**
- **Detection**: Variance > 5× the 99th percentile
- **Cause**: Line noise, movement artifacts, electrical interference
- **Action**: **DROP** before RANSAC
- **Rationale**: Corrupts RANSAC correlation estimates
- **Critical**: Must be removed BEFORE RANSAC

### Tier 2: RANSAC Detection

- **Method**: Random sample consensus with spatial correlation
- **Threshold**: Correlation < 0.75 with predicted signal
- **Action**: INTERPOLATE or DROP (depending on clustering)
- **When**: After Tier 1 removal (on clean data)

---

## Clustering Analysis

### Algorithm: K-Nearest Neighbors

For each bad channel:
1. Find K=6 nearest neighbors in 3D electrode space
2. Count how many neighbors are also bad
3. Calculate percentage of bad neighbors
4. Classify as **clustered** if ≥50% neighbors are bad

**Why K=6?**
- Balances local vs global structure
- Typical electrode spacing in dense arrays (128 ch)
- Matches spline interpolation kernel size

### Severity Levels

| Severity | % Clustered | Description | Recommendation |
|----------|------------|-------------|----------------|
| **None** | 0% | No clustering | Standard interpolation safe |
| **Mild** | <20% | Small isolated clusters | Monitor interpolation quality |
| **Moderate** | 20-50% | Larger clusters or multiple small | Consider dropping clustered channels |
| **Severe** | >50% | Very large clusters | Drop clustered channels or exclude subject |

### Clustering Result Structure

```python
{
    'n_bad_channels': int,                # Total bad channels
    'n_clustered_channels': int,          # How many are clustered
    'pct_clustered': float,               # Percentage clustered
    'mean_pct_bad_neighbors': float,      # Average across all bad channels
    'max_pct_bad_neighbors': float,       # Worst channel
    'severity': str,                      # 'none', 'mild', 'moderate', 'severe'
    'clustered_channels': list,           # Names of clustered channels
    'warning': str,                       # Warning message if moderate/severe
    'details': list                       # Per-channel neighbor analysis
}
```

---

## Adaptive Handling Strategies

### Configuration

```yaml
bad_channels:
  enabled: true
  auto_detect: true
  interpolate: true

  # Clustering analysis
  check_clustering: true
  clustering_action: 'adaptive'  # See options below

  # Thresholds
  clustering_thresholds:
    moderate: 20.0              # % clustered for moderate severity
    severe: 50.0                # % clustered for severe severity

  # Algorithm parameters
  clustering_k_neighbors: 6
  clustering_neighbor_threshold: 0.5  # 50% bad neighbors = clustered
```

### Clustering Actions

| Action | Behavior | Use Case |
|--------|----------|----------|
| **adaptive** (default) | Mild: interpolate all<br>Moderate/Severe: drop clustered, interpolate isolated | Recommended for most datasets |
| **interpolate_all** | Always interpolate, ignore clustering | When you trust interpolation or have dense arrays |
| **drop_clustered** | Always drop clustered channels | Conservative approach |
| **fail_on_severe** | Raise error if severe clustering | Strict QC for high-stakes analyses |

### Adaptive Logic

```python
if severity == 'none' or severity == 'mild':
    # Safe to interpolate all
    interpolate(all_bad_channels)
elif severity == 'moderate' or severity == 'severe':
    # Drop clustered, interpolate isolated
    drop(clustered_channels)
    interpolate(isolated_channels)

    # Safeguard: minimum 10 channels required
    if remaining_channels < 10:
        fallback_to_interpolation()
        add_warning("Too few channels, interpolating despite clustering")
```

---

## Integration with Preprocessing

### Pipeline Order

```
1. Load raw data
2. Set montage (needed for clustering)
3. Detect flatline channels → DROP
4. Detect bridged channels → DROP
5. Detect high variance channels → DROP
6. RANSAC bad channel detection
7. ↓ NEW: Clustering analysis ↓
8. Compute spatial clustering of RANSAC-detected channels
9. Based on severity + action, decide which to drop vs interpolate
10. Apply drops first
11. Apply interpolation to remaining bad channels
12. Continue with re-reference, filter, ICA
```

### Quality Status

Quality assessment now includes clustering:

```python
if pct_bad >= 20 or clustering_severity == 'severe':
    status = 'exclude'
elif pct_bad >= 10 or clustering_severity == 'moderate':
    status = 'warning'
else:
    status = 'ok'
```

---

## Visualization

### Two-Panel Clustering Visualization

**Left Panel**: All Channels
- Good channels: Green circles (●)
- Bad channels: Red X markers (✕)

**Right Panel**: Clustering Analysis
- Good channels: Lime green circles (●)
- **Isolated bad**: Dark orange diamonds (◆) - <50% bad neighbors
- **Clustered bad**: Red X markers (✕) - ≥50% bad neighbors

**Color Rationale:**
- Green = safe for interpolation
- Orange = caution, but interpolation should work
- Red = problematic, interpolation unreliable

### QC Report Integration

Clustering analysis automatically appears in `PreprocessedQC` reports when:
1. Raw data before preprocessing is available
2. Bad channels were detected
3. Montage (channel positions) is set

**Added to report:**
- **Metric**: "Clustered Bad Channels" with severity badge
- **Figure**: Two-panel visualization
- **Notes**: Warning message + processing recommendation

---

## API Reference

### Core Functions

```python
from eegcpm.modules.preprocessing.channel_clustering import (
    compute_bad_channel_clustering,
    visualize_channel_clustering,
    get_clustering_recommendation
)

# Compute clustering
clustering = compute_bad_channel_clustering(
    raw,                           # MNE Raw with montage set
    bad_channels=['E1', 'E5'],     # List of bad channel names
    n_neighbors=6,                 # K for KNN
    cluster_threshold=0.5          # 50% bad neighbors = clustered
)

# Get recommendation
recommendation = get_clustering_recommendation(clustering)
# Returns: "Proceed with standard interpolation" or
#          "Consider dropping clustered channels" etc.

# Visualize
fig = visualize_channel_clustering(raw, bad_channels, clustering)
```

### Preprocessing Integration

```python
from eegcpm.modules.preprocessing import PreprocessingModule

config = {
    'bad_channels': {
        'enabled': True,
        'check_clustering': True,
        'clustering_action': 'adaptive',
    }
}

module = PreprocessingModule(config, output_dir)
result = module.process(raw, subject_id='sub-001')

# Check clustering in result
clustering = result.metadata['bad_channels']['clustering']
print(f"Severity: {clustering['severity']}")
print(f"Clustered: {clustering['n_clustered_channels']}/{clustering['n_bad_channels']}")
```

---

## Batch Processing

### Multi-Pipeline Comparison

Test different clustering strategies on the same subjects:

```bash
python scripts/run_clustering_comparison_batch.py
```

**Pipelines compared**:
1. `standard_interpolate_all`: Ignore clustering (baseline)
2. `clustering_adaptive`: Adaptive strategy (recommended)
3. `clustering_drop_all`: Conservative (always drop)
4. `clustering_strict_qc`: Fail on severe clustering

**Outputs**:
- Individual QC reports per subject/pipeline
- Summary statistics table
- Interactive comparison HTML

### Simple Batch Script

```python
#!/usr/bin/env python3
import mne
from pathlib import Path
from eegcpm.modules.preprocessing import PreprocessingModule

config = {
    'bad_channels': {
        'check_clustering': True,
        'clustering_action': 'adaptive',
    }
}

for subject_id in subjects:
    raw = mne.io.read_raw_fif(f"{subject_id}_eeg.fif", preload=True)
    module = PreprocessingModule(config, output_dir / subject_id)
    result = module.process(raw, subject_id=subject_id)

    # Check result
    clustering = result.metadata['bad_channels']['clustering']
    print(f"{subject_id}: {clustering['severity']} clustering")
```

---

## Edge Cases

### Minimum Channels Safeguard

If dropping clustered channels would leave <10 channels:
- Falls back to interpolation
- Adds warning: "Too few channels remaining, interpolating despite clustering"
- Sets `fallback_to_interpolation: true` in result

### No Montage

If channel positions are unavailable:
- Clustering analysis automatically skipped
- Standard interpolation used
- Note added to QC report

### All Channels Clustered

If all bad channels are clustered (100%):
- Severity = 'severe'
- Adaptive action: drop all
- May trigger exclusion based on total % bad

---

## Testing

### Test Scripts

**`scripts/test_clustering_qc.py`**
- Tests clustering integration in QC reports
- Validates HTML generation with clustering visualization

**`scripts/test_clustering_preprocessing.py`**
- Tests all clustering actions (adaptive, interpolate_all, drop_clustered, fail_on_severe)
- Compares channel counts across strategies
- Validates adaptive behavior

**`scripts/check_bad_channel_clustering.py`**
- Interactive tool for analyzing clustering in existing datasets

**`scripts/run_clustering_comparison_batch.py`**
- Batch multi-pipeline comparison

---

## References

**K-Nearest Neighbors**
Altman, N. S. (1992). An introduction to kernel and nearest-neighbor nonparametric regression. *The American Statistician*, 46(3), 175-185.

**Spline Interpolation**
Perrin, F., et al. (1989). Spherical splines for scalp potential and current density mapping. *Electroencephalography and Clinical Neurophysiology*, 72(2), 184-187.

**PREP Pipeline** (bad channel detection)
Bigdely-Shamlo, N., et al. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. *Frontiers in Neuroinformatics*, 9, 16.

---

**See also**:
- Implementation: `eegcpm/modules/preprocessing/channel_clustering.py`
- QC integration: `eegcpm/modules/qc/preprocessed_qc.py`
- Quality control tiers: `planning/QC_SYSTEM.md`
