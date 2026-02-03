# ERP Feature Extraction - Implementation Complete

**Date**: 2025-12-22
**Status**: ✅ COMPLETE

---

## Overview

Implemented a comprehensive ERP feature extraction system that addresses the three key requirements:

1. ✅ **Task definitions at project initialization** - Task type and primary events defined upfront
2. ✅ **ERP observation for both sensor and source-level** - Separate modules for each
3. ✅ **ERP as Feature Extraction with QC** - Not just QC, full feature extraction with accompanying reports

---

## Architecture

### The Correct Flow

```
Project Init
     ↓
Task Config Definition (task_type, primary_event, erp_components)
     ↓
Preprocessing (with basic ERP QC using primary_event)
     ↓
Epoching
     ↓
     ├─→ Sensor-Level ERP Feature Extraction → features.csv + ERP QC
     └─→ Source Reconstruction
              ↓
         Source-Level ERP Feature Extraction → features.csv + ERP QC
              ↓
         Prediction (combine features from both)
```

### QC vs Feature Extraction

| Stage | Type | Purpose | Uses ERP Components |
|-------|------|---------|---------------------|
| **Preprocessing QC** | Diagnostic QC | Sanity check signal quality | ❌ No (uses primary_event only) |
| **Epochs QC** | Diagnostic QC | Check epoch quality, rejection stats | ❌ No (basic ERP plot) |
| **Source QC** | Diagnostic QC | Check ROI coverage, SNR | ❌ No (ROI time courses ≠ ERPs) |
| **ERP Feature Extraction** | **Feature Extraction** | Extract P1, N1, P3, etc. for prediction | ✅ YES (full component analysis) |
| **Source ERP Feature Extraction** | **Feature Extraction** | Extract ERPs at ROI level | ✅ YES (full component analysis) |

**Key Insight**: ERP analysis belongs in the **Features Stage**, not QC stage. QC just checks data quality.

---

## What Was Implemented

### 1. Enhanced Task Configuration Schema

**File**: `eegcpm/core/task_config.py`

**Added**:
- `task_type`: 'event-related' or 'continuous'
- `primary_event`: Primary event codes for basic QC (used in preprocessing QC)
- `erp_components`: List of ERP components to extract (used in features stage)
- `ERPComponentSpec` model: name, search_window, channels, polarity

**Templates Created**:
- `eegcpm/config/tasks/contdet_example.yaml`
- `eegcpm/config/tasks/rest_example.yaml`

**Example Config**:
```yaml
task_name: contdet
task_type: event-related

# For basic QC in preprocessing stage
primary_event:
  name: all_stimuli
  codes: [1, 2]

# For feature extraction stage
erp_components:
  - name: P1
    search_window: [0.08, 0.12]
    channels: [Oz, O1, O2]
    polarity: positive

  - name: N1
    search_window: [0.14, 0.18]
    channels: [Oz, O1, O2]
    polarity: negative

  - name: P3
    search_window: [0.3, 0.5]
    channels: [Pz, CPz, Cz]
    polarity: positive
```

---

### 2. Sensor-Level ERP Feature Extraction

**File**: `eegcpm/modules/features/erp_features.py`

**Class**: `ERPFeatureModule`

**Extracts**:
- Peak amplitudes and latencies for each component
- Mean amplitudes in time windows
- Area under curve
- Per-channel or averaged features

**Output**:
```
derivatives/features/
└── {preprocessing}/           # e.g., standard
    └── {task}/                # e.g., contdet
        └── epochs/            # Sensor-level
            └── erp/           # Feature type
                └── {variant}/ # e.g., standard_components
                    ├── features.csv          # P1_amplitude, P1_latency, N1_amplitude, ...
                    ├── config.yaml           # Config used
                    └── {subject}_erp_qc.html # QC report (uses existing ERPQC module)
```

**Features Extracted** (per subject row):
- `subject_id`, `session_id`
- `P1_amplitude`, `P1_latency`
- `N1_amplitude`, `N1_latency`
- `P3_amplitude`, `P3_latency`
- `mean_amp_early`, `mean_amp_late` (if windows defined)

**QC Report**:
- Uses existing `eegcpm.modules.qc.erp_qc.ERPQC` module
- Condition-based waveforms
- Topographic maps
- Peak detection visualization

---

### 3. Source-Level ERP Feature Extraction

**File**: `eegcpm/modules/features/source_erp_features.py`

**Class**: `SourceERPFeatureModule`

**Extracts**:
- ERP features **per ROI** (e.g., MPFC, PCC, LPFC_L, etc.)
- Same metrics as sensor-level but localized to source space
- Network-level summaries

**Output**:
```
derivatives/features/
└── {preprocessing}/
    └── {task}/
        └── source-{method}-{template}/  # e.g., source-dSPM-CONN32
            └── erp/
                └── {variant}/
                    ├── features.csv                  # ROI_P1_amplitude, ROI_P1_latency, ...
                    ├── config.yaml
                    └── {subject}_source_erp_qc.html  # ROI-level ERP plots
```

**Features Extracted** (per subject row):
- `subject_id`, `session_id`
- `MPFC_P1_amplitude`, `MPFC_P1_latency`
- `MPFC_N1_amplitude`, `MPFC_N1_latency`
- `PCC_P1_amplitude`, `PCC_P1_latency`
- ... (for each ROI × component combination)

**QC Report**:
- ROI-level ERP waveforms with component windows marked
- Top 12 ROIs by signal strength
- Grid layout for easy comparison

---

## Directory Structure

### Complete Feature Organization

```
derivatives/features/
├── {preprocessing}/           # e.g., standard
│   └── {task}/                # e.g., contdet
│       ├── epochs/            # SENSOR-LEVEL FEATURES
│       │   ├── erp/           # ERP features
│       │   │   ├── early_components/
│       │   │   │   ├── features.csv
│       │   │   │   ├── config.yaml
│       │   │   │   └── sub-001_erp_qc.html
│       │   │   └── late_components/
│       │   │       └── ...
│       │   ├── bandpower/     # Spectral features
│       │   └── timefreq/      # Time-frequency features
│       │
│       └── source-dSPM-CONN32/  # SOURCE-LEVEL FEATURES
│           ├── erp/           # Source ERP features
│           │   ├── standard_components/
│           │   │   ├── features.csv
│           │   │   ├── config.yaml
│           │   │   └── sub-001_source_erp_qc.html
│           │   └── late_only/
│           │       └── ...
│           ├── connectivity/  # Connectivity features
│           └── bandpower/     # Source bandpower features
```

---

## Usage Examples

### 1. Define Task at Project Init

Create task config:

```yaml
# project/eegcpm/configs/tasks/contdet.yaml
task_name: contdet
task_type: event-related

primary_event:
  name: all_stimuli
  codes: [1, 2]

erp_components:
  - name: P1
    search_window: [0.08, 0.12]
    channels: [Oz, O1, O2]
    polarity: positive

  - name: P3
    search_window: [0.3, 0.5]
    channels: [Pz, CPz, Cz]
    polarity: positive
```

### 2. Extract Sensor-Level ERP Features

**Via Python**:
```python
from eegcpm.modules.features import ERPFeatureModule
from eegcpm.core.paths import EEGCPMPaths
import mne

# Load epochs
epochs = mne.read_epochs("derivatives/epochs/standard/contdet/sub-001/sub-001_epo.fif")

# Setup paths
paths = EEGCPMPaths(project_root)
output_dir = paths.get_features_dir(
    preprocessing="standard",
    task="contdet",
    dependency="epochs",
    feature_type="erp",
    variant="standard_components"
)

# Configure
config = {
    "variant": "standard_components",
    "depends_on": {"preprocessing": "standard", "task": "contdet", "epochs": "standard"},
    "components": {
        "P1": {"search_window": [0.08, 0.12], "channels": ["Oz", "O1", "O2"], "polarity": "positive"},
        "P3": {"search_window": [0.3, 0.5], "channels": ["Pz", "CPz", "Cz"], "polarity": "positive"}
    },
    "metrics": ["peak_amplitude", "peak_latency", "mean_amplitude"],
    "generate_qc": True
}

# Extract
module = ERPFeatureModule(config, output_dir)
result = module.process(epochs, subject=subject)
```

**Via CLI** (future):
```bash
eegcpm extract-features \
  --project /path/to/project \
  --config configs/features/erp_standard_components.yaml
```

### 3. Extract Source-Level ERP Features

```python
from eegcpm.modules.features import SourceERPFeatureModule
import numpy as np

# Load ROI time courses
roi_data = np.load("derivatives/source/standard/contdet/variant-dSPM-CONN32/sub-001/sub-001_roi_timecourses.npz")

# Setup paths
output_dir = paths.get_features_dir(
    preprocessing="standard",
    task="contdet",
    dependency="source-dSPM-CONN32",
    feature_type="erp",
    variant="standard_components"
)

# Configure (same component definitions as sensor-level)
config = {
    "variant": "standard_components",
    "depends_on": {"preprocessing": "standard", "task": "contdet", "source": "dSPM-CONN32"},
    "components": {
        "P1": {"search_window": [0.08, 0.12], "polarity": "positive"},
        "P3": {"search_window": [0.3, 0.5], "polarity": "positive"}
    },
    "rois": "all",  # Or specify: ["MPFC", "PCC", "LPFC_L", "LPFC_R"]
    "metrics": ["peak_amplitude", "peak_latency"],
    "generate_qc": True
}

# Extract
module = SourceERPFeatureModule(config, output_dir)
result = module.process(roi_data, subject=subject, sfreq=500)
```

### 4. Combine for Prediction

```python
# Prediction config combines sensor + source ERP features
feature_sources = [
    {
        "path": "features/standard/contdet/epochs/erp/standard_components",
        "prefix": "sensor_erp_"
    },
    {
        "path": "features/standard/contdet/source-dSPM-CONN32/erp/standard_components",
        "prefix": "source_erp_"
    }
]

# Results in combined feature matrix with:
# - sensor_erp_P1_amplitude
# - sensor_erp_P3_amplitude
# - source_erp_MPFC_P1_amplitude
# - source_erp_MPFC_P3_amplitude
# - source_erp_PCC_P1_amplitude
# - ... etc.
```

---

## How This Solves the Three Requirements

### ✅ 1. Task Definitions at Project Init

- Users create task configs **before** any processing
- Task type (`event-related` vs `continuous`) determines workflow
- Primary event codes used for basic QC in preprocessing stage
- ERP component definitions stored in task config for later use

### ✅ 2. ERP Observation for Source-Reconstructed Data

- `SourceERPFeatureModule` extracts ERPs at ROI level
- Same component definitions applied to source space
- Allows comparison: sensor P1 vs MPFC P1 vs V1 P1
- Network-level ERP analysis possible

### ✅ 3. ERP as Feature Extraction with QC

- **Feature Extraction**: Primary purpose is creating features for prediction
- **QC Reports**: Accompanying quality control using existing `ERPQC` module
- Features saved to `features.csv` for later combination
- Clear separation: QC checks quality, Features extract predictors

---

## Benefits

1. **Consistent Workflow**: Task config defines everything upfront
2. **Flexibility**: Multiple variants (early vs late components, different channels)
3. **Source-Level ERPs**: Novel capability to observe ERPs at ROI level
4. **Prediction Ready**: Features formatted for CPM and other models
5. **Reproducible**: Config files track all parameters
6. **QC Integrated**: Automatic quality reports using proven ERP QC module

---

## Next Steps

### Immediate (Required for UI)
- Update UI task configuration page to include:
  - Task type selection (event-related vs continuous)
  - Primary event definition
  - ERP component builder

### Future Enhancements
- CLI commands for ERP feature extraction
- Batch processing across subjects
- Template configs for common paradigms (oddball, N-back, etc.)
- Network-level ERP summaries (DMN average P3, etc.)

---

## Files Created/Modified

### New Files
1. `eegcpm/config/tasks/contdet_example.yaml` - Event-related task example
2. `eegcpm/config/tasks/rest_example.yaml` - Continuous task example
3. `eegcpm/modules/features/erp_features.py` - Sensor-level ERP extraction
4. `eegcpm/modules/features/source_erp_features.py` - Source-level ERP extraction

### Modified Files
1. `eegcpm/core/task_config.py` - Added task_type, primary_event, erp_components
2. `eegcpm/modules/features/__init__.py` - Export new modules

---

## Status: COMPLETE ✅

All three requirements implemented:
1. ✅ Task config at init with primary events
2. ✅ Source-level ERP observation
3. ✅ ERP as feature extraction (not just QC)

Ready for UI integration and testing!
