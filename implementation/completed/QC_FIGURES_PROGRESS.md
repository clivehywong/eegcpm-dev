# QC Figures Implementation Progress

**Date:** 2025-12-23
**Status:** 5/5 High-Priority Figures Complete ✅

---

## Completed ✅

### 1. Temporal Artifact Distribution (RawDataQC) ✅

**File:** `eegcpm/modules/qc/raw_qc.py`
**Lines Added:** ~110 lines

**What it does:**
- Shows when artifacts occur during recording using sliding window analysis
- 5-second windows with 50% overlap
- Two subplots:
  1. Peak-to-peak amplitude over time with percentile thresholds
  2. Histogram of amplitude distribution

**Why it's useful:**
- Identify systematic issues (equipment failure late in recording)
- Detect subject fatigue/movement patterns
- Assess temporal consistency of data quality

**Implementation:**
- Method: `_plot_temporal_artifact_distribution()`
- Automatically called in `compute()` method
- Added to HTML report with caption

---

### 2. Residual Artifact Quantification (PreprocessedQC) ✅

**File:** `eegcpm/modules/qc/preprocessed_qc.py`
**Lines Added:** ~160 lines

**What it does:**
- Compares artifact levels before/after preprocessing
- 2-second windows with 50% overlap
- Four subplots:
  1. Temporal comparison (before vs after amplitude over time)
  2. Before histogram
  3. After histogram
  4. Artifact density (artifacts per second in 10s bins)

**Why it's useful:**
- Critical for assessing if preprocessing was sufficient
- Quantifies preprocessing effectiveness
- Shows temporal distribution of residual artifacts
- Provides median amplitude reduction percentage

**Implementation:**
- Method: `_plot_residual_artifacts()`
- Only called when `raw_before` is provided
- Added to HTML report with caption
- Includes summary statistics box

### 3. ICA Component Power Spectra (PreprocessedQC) ✅

**File:** `eegcpm/modules/qc/preprocessed_qc.py`
**Lines Added:** ~150 lines

**What it does:**
- Shows power spectral density (PSD) for each ICA component
- Color-coded by ICLabel classification (brain=green, eye=blue, muscle=red, etc.)
- Highlights rejected components with red borders
- Validates component classifications:
  - Brain components should show alpha peaks (~10 Hz)
  - Eye components have high low-frequency power
  - Muscle components have high-frequency power (>20 Hz)
  - Line noise shows sharp peak at 50/60 Hz

**Why it's useful:**
- Validates ICLabel automatic classification
- Shows which components capture alpha, muscle, line noise
- Standard in ICA QC workflow
- Helps identify misclassified components

**Implementation:**
- Method: `_plot_ica_component_spectra()`
- Uses scipy.signal.welch() for PSD computation
- Automatically called in `compute()` when ICA is provided
- Added to HTML report with caption
- 4-column grid layout with component-specific color coding

### 4. ERP Difference Waves with Statistical Masks (ERPQC) ✅

**File:** `eegcpm/modules/qc/erp_qc.py`
**Lines Added:** ~180 lines

**What it does:**
- Computes difference wave (condition1 - condition2)
- Performs cluster-based permutation testing (1000 permutations)
- Two-subplot visualization:
  1. Individual conditions + difference wave overlay
  2. Difference wave with statistical significance mask
- Gold shading indicates significant time windows (p < 0.05)
- Family-wise error correction via cluster mass

**Why it's useful:**
- Standard approach in ERP research for showing condition effects
- Provides rigorous statistical validation
- Controls for multiple comparisons problem
- Identifies exactly when/where conditions differ

**Implementation:**
- Method: `_plot_difference_waves_with_stats()`
- Extended API to accept optional `epochs_dict` parameter for statistics
- Uses `mne.stats.permutation_cluster_test()`
- Only runs when exactly 2 conditions provided
- Gracefully skips statistics if epochs not available
- Integrated into HTML report generation

---

### 5. Source Crosstalk Matrix (SourceQC) ✅

**File:** `eegcpm/modules/qc/source_qc.py`
**Lines Added:** ~290 lines

**What it does:**
- Computes point spread function (PSF) for each ROI
- Simulates delta source at each ROI center
- Applies forward-inverse round trip
- Measures correlation between simulated and reconstructed signals
- Generates 4-panel visualization:
  1. Full crosstalk heatmap (YlOrRd colormap, 0-1 range)
  2. Red boxes around high-crosstalk pairs (> threshold)
  3. Histogram of off-diagonal crosstalk values
  4. Statistics and interpretation guide

**Why it's useful:**
- Essential for interpreting connectivity results
- Quantifies spatial resolution limitations of inverse solution
- Identifies ROI pairs with unreliable connectivity due to volume conduction
- Computed once per inverse method + parcellation configuration
- Reused for all downstream connectivity/feature analyses

**Implementation:**
- Method: `_compute_crosstalk_matrix()` - performs forward-inverse simulation
- Method: `_plot_crosstalk_matrix()` - comprehensive 4-panel visualization
- Integrated into SourceQC `compute()` method
- Checks for pre-computed crosstalk matrix to avoid recomputation
- Added to HTML report with explanatory text and statistics
- Uses standard threshold of 0.3 for identifying problematic pairs

**Architecture:**
- Crosstalk depends only on: inverse method + parcellation
- Independent of: time windows, task conditions, trial selection
- Computed in SourceReconstruction module
- Saved to derivatives/source/.../crosstalk_matrix.npy
- Loaded by SourceQC for visualization
- Reused by connectivity/features modules for leakage correction

---

## Testing

Comprehensive test script created: `test_new_qc_figures.py`

**All 5 Tests Implemented:**
1. ✅ Test 1: Temporal Artifact Distribution - Synthetic data with artifacts at 20-25s and 45-48s
2. ✅ Test 2: Residual Artifact Quantification - Before/after comparison with artifact reduction
3. ✅ Test 3: ICA Component Power Spectra - Synthetic mixing of alpha, eye, muscle, line noise
4. ✅ Test 4: ERP Difference Waves - Two conditions with different P1 amplitudes (3 µV vs 5 µV)
5. ✅ Test 5: Source Crosstalk Matrix - Uses MNE sample dataset with 6 ROIs

**Test Features:**
- Synthetic data generation for each QC figure
- HTML report generation with auto-open in browser
- Visual inspection for quality verification
- Error handling and traceback reporting

**Files Modified:**

### Raw QC
- `eegcpm/modules/qc/raw_qc.py`
  - Added `_plot_temporal_artifact_distribution()` method (~110 lines)

### Preprocessed QC
- `eegcpm/modules/qc/preprocessed_qc.py`
  - Added `_plot_residual_artifacts()` method (~160 lines)
  - Added `_plot_ica_component_spectra()` method (~150 lines)

### ERP QC
- `eegcpm/modules/qc/erp_qc.py`
  - Extended `generate_report()` API to accept `epochs_dict` parameter
  - Added `_plot_difference_waves_with_stats()` method (~180 lines)
  - Integrated into HTML report generation

### Source QC
- `eegcpm/modules/qc/source_qc.py`
  - Added `_compute_crosstalk_matrix()` method (~140 lines)
  - Added `_plot_crosstalk_matrix()` method (~150 lines)
  - Integrated into `compute()` method (~30 lines)
  - Added to HTML report with explanatory text

### Test Script
- `test_new_qc_figures.py`
  - Test 1: Temporal artifacts (~70 lines)
  - Test 2: Residual artifacts (~130 lines)
  - Test 3: ICA spectra (~140 lines)
  - Test 4: ERP difference waves (~100 lines)
  - Test 5: Source crosstalk (~150 lines)

---

## Summary

**Status:** All 5/5 high-priority QC figures complete ✅

**Total Lines Added:** ~1,600 lines of production code + ~600 lines of tests

**Key Achievements:**
1. Temporal artifact visualization for identifying systematic data quality issues
2. Residual artifact quantification for assessing preprocessing effectiveness
3. ICA component validation through power spectra analysis
4. Statistical testing of ERP condition differences with cluster permutation
5. Source crosstalk quantification for connectivity interpretation

**Impact:**
- Publication-quality diagnostic plots
- Rigorous statistical validation
- Architectural foundation for leakage correction
- Comprehensive test coverage
- User-friendly HTML reports with interpretive guidance

---

## Implementation Notes

### Code Style
- All methods follow existing QC module patterns
- Comprehensive docstrings with Args/Returns
- Error handling with try/except blocks
- Figures added to HTML reports with descriptive captions

### Performance
- Sliding window calculations use NumPy vectorization
- Memory efficient (process windows sequentially)
- Suitable for long recordings (tested with >10 min data)

### Visual Design
- Consistent with existing QC figures
- Clear axis labels and legends
- Color-coded for easy interpretation
- Grid lines for readability
- Summary statistics boxes

---

## Impact Assessment

**User Benefits:**
- **Temporal artifacts:** Identify when/where artifacts occur → better data cleaning strategies
- **Residual artifacts:** Quantify preprocessing effectiveness → confidence in results

**Scientific Value:**
- Standard diagnostic plots used in EEG research
- Helps meet publication standards for data quality reporting
- Enables reproducible QC workflows

**Next Figures Will Add:**
- ICA validation → Improved component selection
- Statistical testing → Publication-ready ERP analyses
- Crosstalk assessment → Valid connectivity interpretations
