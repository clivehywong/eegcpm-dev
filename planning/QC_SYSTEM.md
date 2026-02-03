# Quality Control System

**Status**: Implemented
**Location**: `eegcpm/modules/qc/`

---

## Overview

Multi-tier quality control system for EEG preprocessing and analysis. Provides automated quality metrics, diagnostic visualizations, and HTML reports to ensure data integrity and identify subjects/runs requiring exclusion.

### QC Modules

| Module | Purpose | Output | Location |
|--------|---------|--------|----------|
| **RawDataQC** | Assess unprocessed data quality | PSD, channel variance, correlations | `qc/raw_qc.py` |
| **PreprocessedQC** | Validate preprocessing effectiveness | Before/after comparison, ICA diagnostics, ERP waveforms | `qc/preprocessed_qc.py` |
| **EpochsQC** | Evaluate epoch quality | Rejection stats, trial counts, ERPs by condition | `qc/epochs_qc.py` |
| **ERPQC** | Event-related potential analysis | Condition-based ERPs, topomaps, peak detection | `qc/erp_qc.py` |
| **ComparisonQC** | Compare preprocessing methods | Side-by-side visualizations, statistical comparisons | `qc/preprocessing_comparison_qc.py` |

---

## Quality Control Tiers

### Tier 1: Channel-Level QC (Automatic Removal)

**Flatline Channels**
- **Detection**: Variance < 1e-12 V²
- **Cause**: Disconnected electrodes, hardware failure
- **Action**: **DROP** (cannot be interpolated)
- **Rationale**: Zero variance breaks ICA decomposition and source localization

**Bridged Channels**
- **Detection**: Correlation > 0.98 between channel pairs
- **Cause**: Electrode gel creating conductive bridge
- **Action**: **DROP** second channel in each pair
- **Rationale**: Linear dependency reduces rank, biases ICA and connectivity estimates

**High Variance Channels**
- **Detection**: Variance > 5× the 99th percentile
- **Cause**: Line noise (60 Hz), movement artifacts, electrical interference
- **Action**: **DROP** before RANSAC
- **Rationale**: Corrupts RANSAC correlation estimates, dominates ICA components
- **Critical**: Must be removed BEFORE RANSAC to prevent false detections

### Tier 2: RANSAC Bad Channel Detection

- **Method**: Random sample consensus with spatial correlation
- **Threshold**: Correlation < 0.75 with predicted signal
- **Action**: **INTERPOLATE** (not drop)
- **Rationale**: Preserves consistent channel count across subjects for connectivity analysis

### Tier 3: Subject-Level QC

**20% Bad Channel Threshold**
- **Threshold**: 20% (community standard from PREP pipeline)
- **Calculation**: `(Flatline + Bridged + High Variance + RANSAC) / Original Channels`
- **Status Levels**:
  - **< 10%**: OK (high quality)
  - **10-20%**: WARNING (acceptable, use with caution)
  - **> 20%**: EXCLUDE (interpolation unreliable)
- **Rationale**: Excessive interpolation introduces spatial blur, reduces statistical power

**Bad Channel Clustering**
- **Detection**: DBSCAN clustering of bad channels based on spatial proximity
- **Severity Levels**:
  - **None**: No clustering (scattered bad channels)
  - **Mild**: Small isolated clusters
  - **Moderate**: Larger clusters or multiple small clusters
  - **Severe**: Very large clusters indicating regional failure
- **Recommendation**: Severe clustering → consider exclusion even if below 20% threshold

### Tier 4: Epoch-Level QC

- **Minimum trial count**: Task-specific (e.g., 20 trials/condition)
- **Maximum rejection rate**: 50% (after artifact rejection)
- **Status**: Implemented in EpochsQC module

---

## Pipeline Processing Order (Critical)

```
1. Load raw data
2. Detect flatline channels → DROP
3. Detect bridged channels → DROP
4. Detect high variance channels → DROP
5. RANSAC bad channel detection (on clean data) → INTERPOLATE
6. Bad channel clustering analysis → FLAG
7. Subject quality check (20% threshold) → FLAG for exclusion
8. Continue preprocessing (re-reference, filter, ICA)
```

**Why this order matters**: High variance channels must be removed BEFORE RANSAC to prevent corruption of spatial correlation estimates. RANSAC assumes stationary Gaussian noise - line noise (pure sine waves) violates this assumption.

---

## HTML Report System

### Report Structure

QC reports are co-located with data outputs at each processing stage:

```
derivatives/
├── preprocessing/{pipeline}/sub-{ID}/ses-{session}/task-{task}/run-{run}/
│   ├── *_preprocessed_raw.fif
│   ├── *_ica.fif
│   ├── *_preprocessed_qc.html        # QC here!
│   └── *_qc_metrics.json
│
├── epochs/{preprocessing}/{task}/sub-{ID}/ses-{session}/
│   ├── *_epo.fif
│   ├── *_epochs_qc.html              # QC here!
│   └── *_qc_metrics.json
```

### HTML Components

**Self-contained reports** with embedded base64 images:
- **Metrics table**: Numeric QC metrics with status badges (ok/warning/bad)
- **Notes section**: Processing warnings and recommendations
- **Diagnostic figures**: Embedded PNG images (100 DPI for web)
- **Index page**: Subject navigation sidebar with iframe content viewer

**Example HTML builder usage**:
```python
from eegcpm.modules.qc.html_report import HTMLReportBuilder

builder = HTMLReportBuilder(title="Preprocessed QC: sub-001")
builder.add_header("Overall Status: OK", level=2)
builder.add_metrics_table(result.metrics)
builder.add_figure("psd", fig_bytes, caption="Power Spectral Density")
builder.add_notes(result.notes)
html = builder.build()
```

---

## Multi-Pipeline Comparison

Compare different preprocessing approaches on the same dataset.

### Supported Pipelines

**MNE-based**:
- `mne_standard`: Standard MNE preprocessing (0.5-40 Hz, PICARD ICA, ICLabel)
- `mne_extended_ica`: Extended ICA (30 components vs 20)
- `mne_strict_filter`: Strict filtering (1 Hz highpass, 60 Hz notch)

**EEGPrep-based** (EEGLAB-compatible):
- `eegprep_standard`: 3-step pipeline (Mild ASR → ICA → Aggressive ASR)
- `eegprep_very_aggressive`: Maximum artifact removal (BurstCriterion=10)

### Comparison Outputs

```
derivatives/multi-pipeline-comparison/
├── summary_report.html              # Interactive comparison
├── overall_statistics.json          # Aggregate stats
├── summary_statistics.csv           # Per-pipeline metrics
│
└── sub-{ID}/
    ├── mne_standard/
    ├── eegprep_standard/
    ├── qc/
    │   └── *_multi_pipeline_comparison.html
    └── *_statistics.json
```

### Diagnostic Visualizations

**Side-by-side comparison plots**:
1. **Channel locations**: Excluded channels marked in red
2. **Time series**: 10s segments from first/middle/last thirds
3. **Statistical comparison**: Violin plots + statistics table
4. **Per-channel variance**: Log-scale bar plots
5. **Artifact segments**: RMS amplitude with bad segments marked

---

## Metadata Structure

### Preprocessing Metadata

```python
result.metadata = {
    'data_quality': {
        'flatline_channels': ['E1', 'E2'],
        'bridged_channels': ['E5', 'E6'],
        'high_variance_channels': ['E127'],
        'rank': {'eeg': 121},
        'ica_feasible': True,
        'recommended_ica_components': 20,
    },
    'bad_channels': {
        'detected': ['E10', 'E15'],      # RANSAC bad
        'n_bad': 2,                       # RANSAC count
        'n_dropped': 3,                   # Flatline + bridged + high_var
        'n_total_bad': 5,                 # Total bad
        'n_total': 129,                   # Original
        'percent_bad': 3.9,
        'quality_status': 'ok',           # ok / warning / exclude
        'quality_message': 'High quality data',
    },
    'clustering': {
        'n_clustered_channels': 0,
        'pct_clustered': 0.0,
        'severity': 'none',
        'warning': None,
    },
    'iclabel': {
        'components': [
            {
                'index': 0,
                'label': 'brain',
                'probability': 0.95,
                'variance_explained': 12.5,
                'rejected': False,
                'reject_reason': '-'
            },
            # ... more components
        ]
    }
}
```

---

## Usage Examples

### Basic QC Generation

```python
from eegcpm.modules.qc import PreprocessedQC

qc = PreprocessedQC(output_dir=Path("output/qc"), dpi=100)

result = qc.compute(
    data=raw_preprocessed,
    subject_id="sub-001",
    ica=ica,
    raw_before=raw_original,
    session_id="01",
    task_name="contdet"
)

html_path = Path("output/qc/sub-001_preprocessed_qc.html")
qc.generate_html_report(result, save_path=html_path)
```

### Quality Check

```python
# Check quality status
if result.metadata['bad_channels']['quality_status'] == 'exclude':
    print("⛔ EXCLUDE: Subject has too many bad channels")
elif result.metadata['bad_channels']['quality_status'] == 'warning':
    print("⚠️  WARNING: Monitor data quality closely")
else:
    print("✓ OK: High quality data")

# Check clustering
if result.metadata['clustering']['severity'] in ['moderate', 'severe']:
    print(f"⚠️ Clustering warning: {result.metadata['clustering']['warning']}")
```

### Pipeline Comparison

```python
from eegcpm.modules.qc import MultiPipelineComparison

comparator = MultiPipelineComparison(
    config_file="config/multi_pipeline_comparison.yaml",
    output_dir=Path("output/comparison")
)

results = comparator.run_batch(
    bids_root=Path("data/bids"),
    subjects=["sub-001", "sub-002"],
    task="rest"
)

# Results include:
# - Per-subject comparison reports
# - Per-pipeline statistics
# - Summary report across all subjects
```

---

## References

**PREP Pipeline**
Bigdely-Shamlo, N., et al. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. *Frontiers in Neuroinformatics*, 9, 16.

**RANSAC Bad Channel Detection**
Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: a paradigm for model fitting. *Communications of the ACM*, 24(6), 381-395.

**ICLabel**
Pion-Tonachini, L., et al. (2019). ICLabel: An automated electroencephalographic independent component classifier. *NeuroImage*, 198, 181-197.

**Autoreject**
Jas, M., et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage*, 159, 417-429.

---

**See also**:
- Implementation: `eegcpm/modules/qc/`
- Bad channel detection: `eegcpm/modules/preprocessing/channel_clustering.py`
- HTML report builder: `eegcpm/modules/qc/html_report.py`
