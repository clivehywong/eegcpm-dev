# EEGCPM Scripts

Utility scripts for development, testing, and batch processing.

---

## Utilities

### `regenerate_qc.py`
**Purpose**: Regenerate QC reports without reprocessing data

**Usage**:
```bash
python scripts/regenerate_qc.py <subject_id> [pipeline] [session] [task] [run]
```

**Example**:
```bash
python scripts/regenerate_qc.py NDARAA306NT2 standard 01 contdet 1
```

**When to use**: After updating QC visualization code, regenerate reports from existing preprocessed data.

---

### `check_bad_channel_clustering.py`
**Purpose**: Analyze bad channel clustering in existing datasets

**Usage**:
```bash
python scripts/check_bad_channel_clustering.py
```

**Output**: Interactive analysis of spatial clustering patterns in bad channels.

---

### `launch_run_selection_ui.py`
**Purpose**: Launch Streamlit run selection interface

**Usage**:
```bash
python scripts/launch_run_selection_ui.py
```

**Opens**: http://localhost:8503

**Alternative**: `streamlit run eegcpm/ui/pages/4_run_selection.py`

---

### `generate_aggregate_report.py`
**Purpose**: Generate aggregate QC summary across all subjects

**Usage**:
```bash
python scripts/generate_aggregate_report.py
```

**Output**: `derivatives/{pipeline}/qc/index.html` - Interactive index with subject navigation.

---

## Batch Processing

### `run_clustering_comparison_batch.py`
**Purpose**: Compare different clustering strategies on same subjects

**Strategies compared**:
- `standard_interpolate_all`: Ignore clustering (baseline)
- `clustering_adaptive`: Adaptive strategy (recommended)
- `clustering_drop_all`: Conservative (always drop)
- `clustering_strict_qc`: Fail on severe clustering

**Usage**:
```bash
python scripts/run_clustering_comparison_batch.py
```

**Output**: Individual QC reports + summary statistics + comparison HTML.

---

### `run_multi_pipeline_comparison_batch.py`
**Purpose**: Compare different preprocessing pipelines on same subjects

**Pipelines compared**:
- `mne_standard`: Standard MNE (0.5-40 Hz, PICARD ICA, ICLabel)
- `mne_extended_ica`: Extended ICA (30 components)
- `mne_strict_filter`: Strict filtering (1 Hz highpass, 60 Hz notch)
- `eegprep_standard`: 3-step pipeline (ASR → ICA → ASR)
- `eegprep_very_aggressive`: Maximum artifact removal

**Usage**:
```bash
python scripts/run_multi_pipeline_comparison_batch.py
```

**Output**: Side-by-side visualizations + statistical comparisons.

---

## Development Tests

### `test_clustering_preprocessing.py`
**Purpose**: Test clustering integration in preprocessing

**Tests**:
- All clustering actions (adaptive, interpolate_all, drop_clustered, fail_on_severe)
- Channel count validation
- Adaptive behavior verification

**Usage**:
```bash
python scripts/test_clustering_preprocessing.py
```

---

### `test_clustering_qc.py`
**Purpose**: Test clustering visualization in QC reports

**Tests**:
- Clustering analysis in QC
- HTML generation with clustering plots
- Metric display

**Usage**:
```bash
python scripts/test_clustering_qc.py
```

---

### `test_epoch_combination.py`
**Purpose**: Test multi-run epoch combination

**Tests**:
- Run-level preprocessing
- Quality assessment
- Epoch combination
- Channel harmonization

**Usage**:
```bash
python scripts/test_epoch_combination.py
```

---

### `test_run_processor.py`
**Purpose**: Test RunProcessor for multi-run workflows

**Tests**:
- Processing multiple runs for same subject
- Quality metrics extraction
- Run selection recommendations

**Usage**:
```bash
python scripts/test_run_processor.py
```

---

## Archived Scripts

### `cleanup_and_archive.sh`
**Purpose**: Archive old scripts (historical)

**Note**: This script was used during development cleanup. Not needed for regular use.

---

## Script Organization

**Keep scripts in this directory for**:
- Utilities that users/developers might run manually
- Batch processing tools
- Development testing (integration tests not in `tests/`)

**Do NOT add**:
- Production code (goes in `eegcpm/`)
- Unit tests (go in `tests/unit/`)
- Obsolete scripts (delete them)

---

## See Also

- `eegcpm/cli/` - Production CLI commands (`eegcpm preprocess`, etc.)
- `tests/` - Unit test suite
- `eegcpm/ui/` - Streamlit interface
