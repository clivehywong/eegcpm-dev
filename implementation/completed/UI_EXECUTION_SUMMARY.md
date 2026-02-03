# UI Execution Integration - Completed

## Summary

The EEGCPM Streamlit UI now supports **direct execution** of processing stages instead of just showing commands to copy/paste.

## What Was Implemented

### 1. Execution Utilities (`eegcpm/ui/utils/executor.py`)

Created a new module with:
- **`ProcessExecutor`** - Subprocess management with real-time output streaming
- **`run_eegcpm_command()`** - Execute EEGCPM CLI commands with live log display in UI
- **`create_source_config()`** - Helper to generate source reconstruction configs

### 2. Updated UI Pages

#### **Page 8: Source Reconstruction** (`8_source_reconstruction.py`)
- ✅ Now executes `eegcpm source-reconstruct` directly
- ✅ Shows real-time logs during execution
- ✅ Displays success/failure status
- ✅ Shows QC reports in Results tab

#### **Page 9: Epochs Processing** (NEW - `9_epochs_processing.py`)
- ✅ New page for epochs extraction
- ✅ Executes `eegcpm epochs` directly
- ✅ Subject selection from preprocessed data
- ✅ Results viewer with QC reports

#### **Page 3: Single Subject Preprocessing** (`3_single_preprocessing.py`)
- ✅ Already had direct execution via `RunProcessor`
- ✅ No changes needed

## How to Use

### 1. Start the UI

```bash
cd /Users/clive/eegcpm/eegcpm-0.1
streamlit run eegcpm/ui/app.py --server.port 8502
```

### 2. Processing Workflow

**Typical workflow:**

1. **Configure Project** (Home page)
   - Select BIDS root directory
   - Set up EEGCPM workspace

2. **Configure Preprocessing** (Page 0)
   - Choose pipeline steps (filtering, ICA, ASR, etc.)
   - Save configuration YAML

3. **Run Preprocessing** (Page 3)
   - Select subject/session/task
   - Choose preprocessing config
   - Click "Run Preprocessing" → Executes directly!
   - View results and QC reports

4. **Configure Task** (Page 1)
   - Define epoch timing (tmin, tmax, baseline)
   - Map event codes to conditions
   - Save task configuration YAML

5. **Extract Epochs** (Page 9 - NEW!)
   - Select task configuration
   - Choose subjects to process
   - Click "Run Epochs Extraction" → Executes directly!
   - View epochs QC reports

6. **Source Reconstruction** (Page 8)
   - Configure inverse method (dSPM, sLORETA, etc.)
   - Set parcellation (CONN 32 ROIs)
   - Choose subjects
   - Click "Run Source Reconstruction" → Executes directly!
   - View source QC reports with sliding window correlation

## Technical Details

### Real-Time Log Streaming

The `ProcessExecutor` class:
```python
executor = ProcessExecutor()
executor.start(['eegcpm', 'source-reconstruct', '--project', '/path/to/project'])

while executor.is_running:
    new_logs = executor.get_output_lines()
    # Display in Streamlit
```

### Integration Pattern

All execution pages follow this pattern:

```python
from eegcpm.ui.utils.executor import run_eegcpm_command

# In button click handler:
args = {
    'project': str(project_root),
    'config': str(config_path),
    'subjects': selected_subjects
}

status_container = st.empty()
log_container = st.container()

success = run_eegcpm_command(
    command='source-reconstruct',  # or 'epochs', 'preprocess'
    args=args,
    log_container=log_container,
    status_container=status_container
)
```

## Testing

To test the complete workflow:

```bash
# 1. Start UI
streamlit run eegcpm/ui/app.py --server.port 8502

# 2. In browser (http://localhost:8502):
#    - Navigate to "Processing: Source Reconstruction" (Page 8)
#    - Configure and run on test data
#    - Check Results tab for QC reports

# 3. Test epochs:
#    - Navigate to "Processing: Epochs" (Page 9)
#    - Select task config
#    - Run extraction
#    - View QC reports
```

## Next Steps

**Completed:**
- ✅ Execution utilities
- ✅ Source reconstruction UI execution
- ✅ Epochs processing UI execution
- ✅ Preprocessing already working

**Future Enhancements:**
- Add progress bars showing % completion
- Add cancel/stop button for running processes
- Add email notifications when long jobs complete
- Save execution logs to files
- Add batch job management (queue multiple subjects)
- Connectivity module UI (when module is implemented)
- Features stage UI
- CPM prediction UI

## File Locations

```
eegcpm/ui/
├── utils/
│   └── executor.py              # NEW - Execution utilities
├── pages/
│   ├── 3_single_preprocessing.py  # Already has execution
│   ├── 8_source_reconstruction.py # UPDATED - Added execution
│   └── 9_epochs_processing.py     # NEW - Epochs execution
```

## Notes

- All execution happens via CLI commands (`eegcpm epochs`, `eegcpm source-reconstruct`)
- UI streams stdout/stderr in real-time
- Process runs in subprocess, doesn't block Streamlit
- Error handling shows tracebacks in UI
- QC reports embedded directly in Results tabs
