# EEGCPM Development Changelog

This document chronicles the major development phases and architectural decisions in EEGCPM's evolution.

---

## Phase 2: Data Quality Detection System (Nov 2025)

### Motivation

50% of preprocessing failures were caused by LAPACK errors from rank-deficient data. Needed proactive quality detection before analysis.

### Implementation

**Created:** `eegcpm/modules/preprocessing/data_quality.py`

**Four Detection Methods:**

1. **Flatline Channels**: Variance < 1e-12 V² (disconnected electrodes)
2. **Bridged Channels**: Correlation > 0.98 (gel creates conductive paths)
3. **Data Rank**: SVD-based dimensionality estimation
4. **Comprehensive Assessment**: Combined detection with ICA recommendations

**Preprocessing Integration:**
- Added Step 1.5: Data quality assessment (after montage, before bad channels)
- Modified `_run_ica()` to use rank-based component selection
- Graceful ICA skip if rank too low

**Test Results (sub-NDARZN501NFF):**
- 3 flatline channels detected and removed
- 37 bridged channels detected and removed
- Data rank: 127/129 channels
- ICA: 20 components (rank-based)
- Zero LAPACK errors (previously crashed)

### Impact

- **Before**: 30% success rate (3/10 subjects)
- **After**: 100% success rate on available subjects (8/8)
- **Trade-off**: +3s processing time per subject to eliminate 50% failures

### Key Learnings

1. **Data quality trumps algorithms** - Best methods fail on poor data
2. **Proactive > Reactive** - Check quality BEFORE analysis
3. **Real-world data is messy** - 60% flatlines, 80% bridging in clinical EEG
4. **Rank deficiency is universal** - Fixed ICA components inappropriate
5. **Clear warnings matter** - "3 flatline channels" > "LAPACK error -5"

---

## Stage-First Architecture Refactor (Nov 2025)

### Motivation

Original pipeline-centric architecture caused data duplication when testing analysis variants. Needed flexible mix-and-match of processing stages.

### Architecture Change

**Before (Pipeline-Centric):**
```
eegcpm/pipelines/standard/{subject}/...
derivatives/pipeline-standard/
```

**After (Stage-First):**
```
derivatives/
├── preprocessing/{pipeline}/{subject}/ses-{session}/task-{task}/run-{run}/
├── epochs/{preprocessing}/{task}/{subject}/ses-{session}/
├── source/{preprocessing}/{task}/variant-{method}-{template}/{subject}/...
├── features/{preprocessing}/{task}/{source-variant}/{feature-type}/
└── prediction/{model-name}/
```

### Benefits

1. **Prevents Waste**: Each preprocessing variant reusable across analyses
2. **Flexible Combinations**: Mix preprocessing → epochs → source
3. **Cleaner Organization**: Stage-based directories more intuitive
4. **Single Source of Truth**: EEGCPMPaths eliminates path bugs

### Implementation

**Created:**
- `eegcpm/core/paths.py` (453 lines) - Centralized path management
- `eegcpm/core/dependencies.py` (354 lines) - Dependency resolver
- `tools/migrate_to_stage_first.py` (383 lines) - Migration script

**Modified:**
- `eegcpm/workflow/state.py` - Added stage tracking (current_stage, metadata)
- `eegcpm/core/config.py` - Added BaseStageConfig with dependencies
- `eegcpm/cli/preprocess.py` - Single `--project` argument
- `eegcpm/ui/pages/2_run_selection.py` - Simplified to single project root

### Breaking Changes

1. **Path Arguments**: `--bids-root`, `--derivatives` → `--project`
2. **State DB Location**: `derivatives/.eegcpm/` → `eegcpm/.eegcpm/`
3. **Output Structure**: `eegcpm/pipelines/` → `derivatives/preprocessing/`
4. **QC Reports**: Separate `derivatives/qc/` → Co-located with data

### Migration

```bash
python tools/migrate_to_stage_first.py --project /path --execute --symlinks
```

---

## Naming Convention Standardization (Nov 2025)

### Problem

Inconsistent naming across outputs:
- Mixed case formats (kebab vs snake)
- Redundant information in filenames
- Subject ID inconsistency (with/without `sub-` prefix)
- No version control for pipeline outputs

### Proposed Solution

**Principles:**
1. **Strict BIDS** for data files (`.fif`, `.epo.fif`)
2. **BIDS-inspired** for reports (`.html`, `.json`)
3. **Hierarchical context** - Less repetition
4. **Pipeline versioning** - Track parameters

**Examples:**

SAIIT Task:
```
OLD: task-saiit2afcblock1, task-saiit2afcblock2
NEW: task-saiit_run-01, task-saiit_run-02
```

QC Reports:
```
OLD: NDARZN501NFF_task-saiit2afcblock1_raw-qc.html
NEW: sub-NDARZN501NFF_ses-01_task-saiit_run-01_stage-raw_qc.html
```

### Implementation Priority

**High Priority (Must Have):**
1. Consistent BIDS naming for all data files
2. `task-{name}_run-{N}` instead of custom block names
3. Uniform QC naming (`stage-{name}_qc.html`)
4. Always include `sub-` prefix

**Medium Priority (Should Have):**
5. Pipeline versioning (`eegcpm-v0.1.0/`)
6. `config.yaml` alongside outputs
7. Subject-centric folder organization

**Low Priority (Nice to Have):**
8. BIDS validator compliance
9. Migration of existing data

**Status**: Proposal stage - not yet implemented

---

## QC System Enhancement (Dec 2025)

### Critical Fixes

#### 1. ASR Unit Conversion Bug ⚠️

**Problem**: eegprep stores data in µV, MNE expects Volts. Missing conversion caused 449x scaling error.

**Fix** (`eegcpm/modules/preprocessing/steps/asr.py:177`):
```python
# IMPORTANT: eegprep stores in µV, MNE expects V
data = eeg_cleaned['data'] * 1e-6  # µV → V
```

**Impact**: All preprocessed data amplitudes were wrong by 449x

#### 2. Event Preservation

**Problem**: ASR creates new Raw object without preserving annotations, losing all events.

**Fix** (`eegcpm/modules/preprocessing/steps/asr.py:226-228`):
```python
# Preserve annotations - CRITICAL for epoching
if raw.annotations is not None:
    raw_cleaned.set_annotations(raw.annotations)
```

### QC Report Enhancements

1. **Bad Channels Topography**: Color-coded by reason (dropped=red, interpolated=blue)
2. **PSD Before/After**: Show all channels including interpolated
3. **ICA Component Plots**: Red highlighting for rejected, ICLabel classifications
4. **Show ALL ICA Components**: Dynamic grid, no 20-component limit
5. **Before/After Metrics**: Added Min/Max amplitude rows
6. **Complete ICA Table**: All components with variance explained
7. **ERP Waveforms**: Auto-generated if events available

### ICLabel Migration

**Changed from**: mne-icalabel
**Changed to**: eegprep ICLabel (EEGLAB port)

**Why**: More reliable, better EEGLAB compatibility

**Requires**: PyTorch (CPU version installed)

**Workflow**:
1. Convert MNE Raw → EEGLAB via `eegprep.eeg_mne2eeg()`
2. Add ICA matrices
3. Run `eegprep.iclabel()`
4. Extract 7-class probabilities

### Other Fixes

- **Workflow State JSON**: Fixed serialization error from ICA objects
- **UI Default**: QC reports shown inline by default
- **Montage Restoration**: After ASR to prevent loss

---

## Current Development (Dec 2025)

### Documentation Consolidation

**Goal**: Reduce 18 markdown files → 10 focused documents

**New Structure:**
```
/Users/clive/eegcpm/
├── docs/                  # User/developer documentation
│   └── WORKFLOWS.md       # Multi-run workflow guide
├── planning/              # Architecture documentation
│   ├── QC_SYSTEM.md       # Quality control system
│   └── BAD_CHANNEL_DETECTION.md  # Clustering analysis
└── implementation/        # Historical records
    └── CHANGELOG.md       # This file
```

**Merges Completed:**
- ✅ QC docs (3 → 1): QUALITY_CONTROL_SYSTEM + comparison + multi-pipeline
- ✅ Clustering docs (3 → 1): bad_channel_clustering + implementations + batch
- ✅ Workflow docs (2 → 1): MULTIRUN_WORKFLOW + QUICK_REFERENCE
- ✅ Historical docs (4 → 1): This changelog

**Pending:**
- Move existing docs to new structure
- Create new planning docs (ARCHITECTURE, TASK_SYSTEM, UI_ARCHITECTURE)
- Trim CLAUDE.md (32K → 15-20K)
- Delete obsolete files

---

## References

### PREP Pipeline
Bigdely-Shamlo, N., et al. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. *Frontiers in Neuroinformatics*, 9, 16.

### BIDS Specification
https://bids-specification.readthedocs.io/

### MNE-Python
https://mne.tools/stable/
