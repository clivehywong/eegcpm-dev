# EEGCPM Workflows Guide

**Status**: Implemented
**Location**: `eegcpm/pipeline/`

---

## Overview

This guide covers EEGCPM's workflow systems for processing EEG data from raw files through epoch combination.

---

## Quick Start

### Processing Commands

```bash
# Single subject preprocessing
eegcpm preprocess \
  --config config.yaml \
  --project /path/to/project \
  --pipeline standard \
  --subject SUBJECTID \
  --task taskname

# Launch Streamlit UI
streamlit run eegcpm/ui/app.py --server.port 8502
```

### Key File Locations

```
project_root/
├── bids/                           # BIDS input data
├── derivatives/
│   ├── preprocessing/{pipeline}/   # Preprocessed outputs
│   │   └── sub-{ID}/ses-{session}/task-{task}/run-{run}/
│   │       ├── *_preprocessed_raw.fif
│   │       ├── *_ica.fif
│   │       └── *_preprocessed_qc.html
│   └── epochs/{preprocessing}/{task}/
│       └── sub-{ID}/ses-{session}/
│           ├── *_epo.fif
│           └── *_epochs_qc.html
└── eegcpm/
    ├── configs/                    # Configuration files
    └── .eegcpm/state.db           # Workflow state tracking
```

---

## Multi-Run Workflow

Process multiple runs per subject, assess quality, and combine epochs.

### Workflow Steps

```
1. Run-level preprocessing → Quality metrics
2. Automatic quality assessment → Accept/Review/Reject recommendations
3. Manual run selection (optional) → Override recommendations
4. Epoch combination → Harmonize channels → Create combined epochs
5. QC reporting → Individual + combined + aggregate reports
```

### Quality Indicators

| Symbol | Quality | Bad Channels | Action |
|--------|---------|--------------|--------|
| 🟢 | Excellent | <10% | Auto-accept |
| 🔵 | Good | 10-20% | Auto-accept |
| 🟡 | Acceptable | 20-30% | Review |
| 🔴 | Poor | >30% | Auto-reject |

### Quality Metrics Tracked

- `n_bad_channels`: Number of channels marked as bad
- `pct_bad_channels`: Percentage of bad channels
- `n_clustered_bad`: Spatially clustered bad channels
- `clustering_severity`: None, mild, moderate, severe
- `ica_success`: Whether ICA artifact removal succeeded
- `quality_status`: Overall rating
- `recommended_action`: Accept, review, or reject

### Code Example

```python
from pathlib import Path
from eegcpm.pipeline.run_processor import RunProcessor
from eegcpm.pipeline.epoch_combiner import EpochCombiner
from eegcpm.workflow.state import WorkflowStateManager

# Configuration
bids_root = Path("/path/to/bids")
output_root = Path("/path/to/derivatives/preprocessing/standard")
config = {...}  # Load from YAML

# Setup state manager
state_db = output_root.parent / ".eegcpm" / "state.db"
state_manager = WorkflowStateManager(state_db)

# Process all runs for subject
processor = RunProcessor(
    bids_root=bids_root,
    output_root=output_root,
    config=config,
    state_manager=state_manager
)

results = processor.process_subject_task(
    subject_id="sub-001",
    task="taskname",
    session="01",
    pipeline="standard"
)

# Get automatic recommendations
recommendations = processor.get_run_selection_recommendations(results)

# Combine accepted runs
combined_output = output_root / "sub-001" / "ses-01" / "task-taskname" / "combined"
combiner = EpochCombiner(
    config=config,
    output_dir=combined_output,
    state_manager=state_manager
)

result = combiner.combine_with_selection(
    run_results=results,
    selection=recommendations,
    subject_id="sub-001",
    session="01",
    task="taskname",
    pipeline="standard",
    generate_qc=True
)

print(f"Combined {result.n_total_epochs} epochs from {result.n_runs_combined} runs")
```

### Manual Run Selection

Override automatic recommendations:

```python
# Get automatic recommendations
auto_recommendations = processor.get_run_selection_recommendations(results)

# Manual override
manual_selection = {
    '01': True,   # Include run 01
    '02': True,   # Include run 02
    '03': False,  # Exclude run 03 (override automatic accept)
}

# Combine with manual selection
result = combiner.combine_with_selection(
    run_results=results,
    selection=manual_selection,
    subject_id="sub-001",
    session="01",
    task="taskname",
    pipeline="standard"
)
```

---

## Epoch Combination

### Channel Harmonization

When runs have different bad channels:

1. Finds **common channels** across all selected runs
2. Drops unique bad channels from each run
3. Ensures all runs have the same channel set

**Example:**
```
Run 1: 37 channels (3 bad: Fp1, F3, T7)
Run 2: 36 channels (4 bad: Fp1, F4, C4, T8)

Common channels: 33 (intersection of good channels)
  - Dropped from Run 1: Fp1, F3, T7 (unique bad)
  - Dropped from Run 2: Fp1, F4, C4, T8 (unique bad)
```

### Event Handling

- Excludes BAD, EDGE, BOUNDARY annotations
- Preserves task-related event codes
- Auto-generates event IDs if not in config

### Epoch Parameters

```yaml
epochs:
  tmin: -0.2           # Start time before event
  tmax: 0.5            # End time after event
  baseline: [null, 0]  # Baseline correction window
  reject: null         # Amplitude rejection (or null)
```

---

## Batch Processing

### Multiple Subjects

```python
subjects = ["sub-001", "sub-002", "sub-003"]

for subject_id in subjects:
    print(f"\nProcessing {subject_id}...")

    results = processor.process_subject_task(
        subject_id=subject_id,
        task="taskname",
        session="01",
        pipeline="standard"
    )

    recommendations = processor.get_run_selection_recommendations(results)

    if sum(recommendations.values()) > 0:
        combined_output = output_root / subject_id / "ses-01" / "task-taskname" / "combined"
        combiner = EpochCombiner(config=config, output_dir=combined_output)

        result = combiner.combine_with_selection(
            run_results=results,
            selection=recommendations,
            subject_id=subject_id,
            session="01",
            task="taskname",
            pipeline="standard"
        )

        print(f"  ✓ Combined {result.n_total_epochs} epochs")
    else:
        print(f"  ✗ No runs accepted for {subject_id}")
```

---

## Streamlit UI

### Launch

```bash
# Main app
streamlit run eegcpm/ui/app.py --server.port 8502

# Run selection page directly
streamlit run eegcpm/ui/pages/run_selection.py --server.port 8503
```

### UI Pages

| Page | Purpose |
|------|---------|
| **Home** | Project setup and status |
| **Pipeline Config** | Configure preprocessing steps |
| **Batch Preprocessing** | Process multiple subjects |
| **Task Config** | Configure epoch parameters |
| **Run Selection** | Review and select runs |
| **Preprocessing** | Single subject processing |

### Run Selection UI Features

- Visual quality assessment for all runs
- Color-coded quality indicators
- Interactive checkboxes to select/deselect
- Override automatic recommendations
- One-click epoch combination
- Direct links to QC reports

---

## QC Reports

### Combined QC Report

Location: `derivatives/preprocessing/{pipeline}/{subject}/ses-{session}/task-{task}/combined/{subject}_combined_qc.html`

**Contains:**
1. **Run Quality Assessment**: Table with metrics, bar charts
2. **Channel Harmonization**: Before/after channel counts
3. **Event Distribution**: Counts per event type, charts

### Aggregate Index

Location: `derivatives/preprocessing/{pipeline}/qc/index.html`

**Features:**
- Overview statistics (total subjects, runs, epochs)
- Organized by task
- Subject-level summary table
- Links to individual QC reports

---

## Troubleshooting

### No Events Found

**Problem:** `ValueError: No events found in data`

**Solution:**
- Check annotations exist in preprocessed files
- Verify event codes (not all BAD/EDGE)
- Specify `event_id` in epochs config

### All Epochs Dropped

**Problem:** `RuntimeWarning: All epochs were dropped!`

**Solution:**
- Check `reject` thresholds (may be too strict)
- Verify BAD annotations don't cover all trials
- Set `reject_by_annotation=False`

### Channel Mismatch

**Problem:** `ValueError: raws[1].info['nchan'] must match`

**Solution:**
- Channel harmonization should handle this automatically
- Check `_harmonize_channels()` is being called
- Verify common channels exist across runs

### SSP Projector Mismatch

**Problem:** `ValueError: SSP projectors in raws[1] must be the same`

**Solution:**
- Handled by removing projectors before combination
- Average reference applied after concatenation

---

## CLI Commands

```bash
# Run preprocessing
eegcpm preprocess --config config.yaml --project /path --subject ID

# Check status
eegcpm status --project /path

# Resume interrupted processing
eegcpm resume --project /path --pipeline standard

# Import QC from existing outputs
eegcpm import-qc --project /path --pipeline standard

# View combined epochs
python -c "import mne; e=mne.read_epochs('path/to/epo.fif'); print(len(e))"

# Count processed subjects
find derivatives/preprocessing -name "*_preprocessed_raw.fif" | wc -l
```

---

## Best Practices

1. **Review QC first**: Check raw data quality before preprocessing
2. **Trust auto-recommendations**: Start with accept/reject defaults
3. **Review borderline cases**: Manual check for "review" status
4. **Keep 2+ runs**: Combination improves SNR with multiple runs
5. **Check channel counts**: Ensure enough common channels remain
6. **Verify events**: Check event distribution in combined QC
7. **Document decisions**: Use manual selection for audit trail

---

**See also:**
- Implementation: `eegcpm/pipeline/`
- QC System: `planning/QC_SYSTEM.md`
- Bad Channel Detection: `planning/BAD_CHANNEL_DETECTION.md`
