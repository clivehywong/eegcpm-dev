# UI Architecture

**Status**: Implemented
**Framework**: Streamlit
**Location**: `eegcpm/ui/`

---

## Overview

EEGCPM provides a web-based Streamlit UI for interactive preprocessing, configuration, and quality control review.

---

## Page Structure

### Main App

**File**: `eegcpm/ui/app.py`

**Purpose**: Home page and project dashboard

**Features** (Updated 2025-12-22):
- Project management and configuration
- **Pipeline progress tracking** (5 stages with progress bars)
- **Recent activity log** (last 5 processing events)
- **Quick action buttons** (Continue, Reprocess, View QC)
- **Workflow guide** (stage-based pipeline explanation)
- Dataset summary (subjects, sessions, tasks)

### Page Organization (Stage-Based Navigation)

Pages are organized by processing stage with consistent icons:

```
eegcpm/ui/pages/
│
├─ 🔧 Configuration
│  ├── 0_preprocessing_config.py  # Preprocessing parameters
│  └── 1_task_config.py           # Task/epoch parameters
│
├─ ⚙️ Processing
│  ├── 1_batch_preprocessing.py   # Batch processing
│  ├── 3_preprocessing.py         # Single subject processing
│  └── 5_trial_sorting.py         # Trial organization
│
├─ 📊 Quality Control
│  ├── 6_run_selection.py         # Run selection & QC
│  ├── 7_qc_browser.py            # Browse QC reports
│  └── 8_group_summary.py         # Group-level QC
│
└─ _archived/                     # Orphaned pages (unused)
   ├── analysis.py
   ├── hpc.py
   ├── pipeline.py
   ├── project.py
   └── results.py
```

**Stage-based icons**: 🔧 Configuration, ⚙️ Processing, 📊 Quality Control, 📈 Results (future)

---

## Page Descriptions

### 0. Pipeline Config

**Purpose**: Configure preprocessing pipeline steps

**Features**:
- Template selection (standard, minimal, test_fastica)
- Copy templates to project configs
- Step-by-step configuration:
  - Montage (channel positions)
  - Filter (highpass, lowpass)
  - Drop Flat (variance threshold)
  - Bad Channels (PREP detection)
  - ASR (Artifact Subspace Reconstruction)
  - ICA (Independent Component Analysis)
  - ICLabel (component classification)
  - Reference (re-referencing)
- YAML preview and save
- Config deletion

**Layout**:
- **Left column**: Package templates (read-only)
- **Right column**: Project configs (editable)

**Key Components**:
```python
# Get step parameters from config
step_params = get_step_from_config(config, 'ica')

# Update step in config
update_step_in_config(config, 'ica', {
    'method': 'fastica',
    'n_components': 'rank-1',
    'l_freq_fit': 1.0
})
```

### 1. Batch Preprocessing

**Purpose**: Process multiple subjects in parallel

**Features**:
- Subject selection (multi-select)
- Pipeline selection
- Task and session filtering
- Cleanup options (derivatives, pipelines, logs)
- Batch execution with progress tracking
- Log streaming

**Cleanup Modes**:
- Clean derivatives: Remove all processed outputs
- Clean pipelines: Remove preprocessing folders
- Clean logs: Remove processing logs

**Key Components**:
```python
# Parallel processing
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(process_subject, subjects)
```

### 2. Task Config

**Purpose**: Configure task-specific epoch parameters

**Features**:
- Event code scanning from BIDS
- Condition builder (multi-select event types)
- Epoch timing configuration
- Response mapping
- Trial filtering (RT thresholds, accuracy)
- Epoch binning strategies
- YAML preview and save

**Tabs**:
1. **Epoch Timing**: tmin, tmax, baseline
2. **Conditions**: Event type selection for each condition
3. **Response Mapping**: Behavioral response categories
4. **Preview & Save**: YAML preview, save to configs/

**Key Components**:
```python
def scan_events_from_bids(bids_root, task, max_subjects=10):
    """Scan BIDS events.tsv files to discover event types."""
    event_names = Counter()

    for events_file in bids_root.rglob(f"*task-{task}*_events.tsv"):
        df = pd.read_csv(events_file, sep='\t')
        event_names.update(df['trial_type'].tolist())

    return {'event_names': event_names, ...}
```

### 3. Preprocessing (Single Subject)

**Purpose**: Interactive single-subject preprocessing

**Features**:
- Subject/session/task/run selection
- Pipeline selection
- Real-time progress updates
- Inline QC report display
- Processing logs

**Workflow**:
1. Select data identifiers
2. Choose preprocessing pipeline
3. Execute preprocessing
4. Review QC report inline or in new tab
5. Check processing warnings

**Key Components**:
```python
# Execute preprocessing
result = preprocessing_module.process(
    raw,
    subject_id=subject_id,
    session_id=session_id,
    task_name=task,
    run_id=run
)

# Display QC inline
if view_inline and qc_html_path.exists():
    st.components.v1.html(qc_html.read_text(), height=800, scrolling=True)
```

### 4. Run Selection

**Purpose**: Multi-run quality assessment and epoch combination

**Features**:
- Quality indicator badges (🟢🔵🟡🔴)
- Interactive run selection (checkboxes)
- Quality metrics table
- Override automatic recommendations
- One-click epoch combination
- Direct links to QC reports

**Quality Indicators**:
- 🟢 Excellent: <10% bad channels
- 🔵 Good: 10-20% bad channels
- 🟡 Acceptable: 20-30% bad channels
- 🔴 Poor: >30% bad channels

**Key Components**:
```python
# Get quality recommendations
recommendations = processor.get_run_selection_recommendations(results)

# Combine selected runs
result = combiner.combine_with_selection(
    run_results=results,
    selection=manual_selection,  # Override with user selection
    subject_id=subject_id,
    generate_qc=True
)
```

### 5. Trial Sorting

**Purpose**: Organize epochs by experimental conditions

**Features** (Planned):
- Condition-based trial counts
- Epoch binning visualization
- RT distribution plots
- Accuracy breakdown
- Export trial metadata

---

## Session State Management

### Global State

```python
# Project configuration
st.session_state.eegcpm_root = "/path/to/project"
st.session_state.current_project_name = "HBN Study"

# BIDS paths
st.session_state.bids_root = "/path/to/bids"
st.session_state.derivatives_root = "/path/to/derivatives"
```

### Page-Specific State

```python
# Preprocessing page
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

# Run selection page
if 'selected_runs' not in st.session_state:
    st.session_state.selected_runs = {}
```

---

## Layout Patterns

### Two-Column Layout

Used in Pipeline Config and Batch Preprocessing:

```python
col1, col2 = st.columns([1, 2])  # 1:2 ratio

with col1:
    st.header("Templates")
    # ...

with col2:
    st.header("Project Configs")
    # ...
```

### Sidebar Navigation

```python
st.sidebar.header("📂 Current Project")
st.sidebar.info(f"**{st.session_state.current_project_name}**")
st.sidebar.caption(f"EEGCPM: `{eegcpm_root}`")
```

### Expandable Sections

```python
with st.expander("2️⃣ Filter - Temporal Filtering", expanded=True):
    l_freq = st.number_input("Highpass (Hz)", ...)
    h_freq = st.number_input("Lowpass (Hz)", ...)
```

---

## Data Flow

### User Input → Config

1. User adjusts parameters in UI
2. Parameters stored in `config` dict
3. Config saved to YAML file
4. YAML loaded by CLI/batch scripts

### Processing → QC Display

1. Preprocessing generates QC HTML
2. QC HTML saved co-located with data
3. UI loads QC HTML from file
4. Display inline via `st.components.v1.html()`

### BIDS Scanning → UI Population

1. UI scans BIDS directory structure
2. Discovers subjects, sessions, tasks, runs
3. Populates dropdowns dynamically
4. User selects from available options

---

## Component Library

### Custom Components

**QC Report Viewer**:
```python
def display_qc_report(qc_path: Path, height: int = 800):
    """Display QC HTML report inline."""
    if qc_path.exists():
        html_content = qc_path.read_text()
        st.components.v1.html(html_content, height=height, scrolling=True)
    else:
        st.warning(f"QC report not found: {qc_path}")
```

**Pipeline Selector**:
```python
def select_pipeline(preprocessing_dir: Path) -> str:
    """Select preprocessing pipeline from available options."""
    excluded_dirs = {'logs', 'qc', 'reports', '__pycache__'}
    pipelines = [
        d.name for d in preprocessing_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.') and d.name not in excluded_dirs
    ]
    return st.selectbox("Pipeline", options=sorted(pipelines))
```

**Subject Selector**:
```python
def select_subjects(bids_root: Path, multi: bool = True) -> List[str]:
    """Select subject(s) from BIDS directory."""
    subjects = [d.name for d in bids_root.glob("sub-*") if d.is_dir()]
    if multi:
        return st.multiselect("Subjects", options=sorted(subjects))
    else:
        return [st.selectbox("Subject", options=sorted(subjects))]
```

---

## Styling

### Theme

Default Streamlit theme with custom tweaks:

```python
st.set_page_config(
    page_title="EEGCPM - Pipeline Config",
    page_icon="🔧",
    layout="wide"  # Use full width
)
```

### Status Badges

```python
def status_badge(status: str) -> str:
    """Generate colored status badge."""
    colors = {
        'ok': '🟢',
        'warning': '🟡',
        'exclude': '🔴',
    }
    return f"{colors.get(status, '⚪')} {status.upper()}"
```

### Quality Indicators

```python
def quality_indicator(pct_bad: float) -> str:
    """Generate quality indicator emoji."""
    if pct_bad < 10:
        return '🟢 Excellent'
    elif pct_bad < 20:
        return '🔵 Good'
    elif pct_bad < 30:
        return '🟡 Acceptable'
    else:
        return '🔴 Poor'
```

---

## Launch Commands

### Main App

```bash
streamlit run eegcpm/ui/app.py --server.port 8502
```

### Specific Page

```bash
streamlit run eegcpm/ui/pages/2_task_config.py --server.port 8503
```

### Development Mode

```bash
streamlit run eegcpm/ui/app.py --server.port 8502 --server.runOnSave true
```

---

## Future Enhancements

### Planned Features

1. **Real-time Processing**: WebSocket-based live updates
2. **Interactive Plots**: Plotly for interactive QC figures
3. **User Authentication**: Multi-user project management
4. **Database Integration**: Store QC metrics in database
5. **Export Reports**: PDF generation from QC HTML

### UI Improvements

1. **Drag-and-drop**: File upload for configs
2. **Keyboard shortcuts**: Quick navigation
3. **Dark mode**: Theme toggle
4. **Responsive design**: Mobile-friendly layouts

---

**See also:**
- Implementation: `eegcpm/ui/`
- Architecture: `planning/ARCHITECTURE.md`
- Task System: `planning/TASK_SYSTEM.md`
