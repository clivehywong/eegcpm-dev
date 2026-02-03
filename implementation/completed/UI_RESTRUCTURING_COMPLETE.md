# UI Restructuring Summary

**Date**: 2025-12-22
**Status**: Phase 1 Core Restructuring - Partial Completion

---

## Completed Changes

### 1. Page Titles & Icons Standardization

All pages now follow consistent naming convention with stage-based titles:

**Configuration Pages** (🔧):
- `0_preprocessing_config.py`: "Configuration: Preprocessing"
- `1_task_config.py`: "Configuration: Task & Epochs"

**Processing Pages** (⚙️):
- `1_batch_preprocessing.py`: "Processing: Batch Preprocessing"
- `3_preprocessing.py`: "Processing: Single Subject Preprocessing"
- `5_trial_sorting.py`: "Processing: Trial Sorting"

**Quality Control Pages** (📊):
- `6_run_selection.py`: "Quality Control: Run Selection"
- `7_qc_browser.py`: "Quality Control: QC Browser"
- `8_group_summary.py`: "Quality Control: Group Summary"

### 2. Home Page Redesign (`eegcpm/ui/app.py`)

**New Features Added:**
- **Pipeline Progress Tracking**: Scans derivatives folder to show completion percentage for all 5 stages
  - Preprocessing
  - Epochs
  - Source
  - Features
  - Prediction
- **Recent Activity Log**: Shows last 5 processing events with timestamps
- **Quick Action Buttons**: Continue Last Stage, Reprocess Failed Subjects, View QC Reports
- **Workflow Guide**: Explains the stage-based pipeline approach

**Functions Added:**
```python
get_pipeline_progress(project_root: Path) -> dict
get_recent_activity(project_root: Path, limit: int) -> list
```

### 3. Page Renumbering

**New page structure:**
```
eegcpm/ui/pages/
├── 0_preprocessing_config.py    # Configuration: Preprocessing
├── 1_task_config.py              # Configuration: Task/Epoch parameters
├── 1_batch_preprocessing.py     # Processing: Batch (TO BE MERGED)
├── 3_preprocessing.py            # Processing: Single subject (TO BE MERGED)
├── 5_trial_sorting.py            # Processing: Trial organization (INCOMPLETE)
├── 6_run_selection.py            # QC: Run selection & quality assessment
├── 7_qc_browser.py               # QC: Browse QC reports
├── 8_group_summary.py            # QC: Group-level summary
└── _archived/                    # Orphaned pages moved here
    ├── analysis.py
    ├── hpc.py
    ├── pipeline.py
    ├── project.py
    └── results.py
```

### 4. Orphaned Pages Archived

Moved 5 unused/incomplete pages to `_archived/` subdirectory:
- `analysis.py` - Empty placeholder
- `hpc.py` - HPC job submission (incomplete)
- `pipeline.py` - Pipeline visualization (unused)
- `project.py` - Duplicate project management
- `results.py` - Empty placeholder

---

## Remaining Work

### Phase 1 Tasks (Still TODO)

1. **Merge Batch + Single Preprocessing Pages**
   - Current: Two separate pages (`1_batch_preprocessing.py`, `3_preprocessing.py`)
   - Target: Single unified page `2_preprocessing.py` with mode selector (Batch/Single)
   - Complexity: Both pages are ~300-500 lines, need careful merging

2. **Create Unified Configuration Page**
   - Combine `0_preprocessing_config.py` + `1_task_config.py`
   - Use tabs to separate preprocessing vs task configuration
   - Target filename: Keep as separate pages or merge?

3. **Update Page Titles**
   - Update `st.set_page_config()` and headers in all pages
   - Ensure consistency with new stage-based navigation

4. **Create Processing Pipeline Page** (New)
   - Stage selector for Preprocessing → Epochs → Source → Connectivity
   - Dependency validation
   - Progress tracking per stage
   - Target: `2_processing_pipeline.py` (or keep existing pages?)

### Missing Stages (Phase 2+)

5. **Create Epochs Processing Page** (New)
   - Currently `5_trial_sorting.py` exists but is incomplete
   - Needs: Event selection, epoch creation, rejection parameters
   - Target: `3_epochs.py`

6. **Create Source Reconstruction Page** (New)
   - Method selection (dSPM, sLORETA)
   - Template/parcellation selection
   - ROI specification
   - Target: `4_source.py`

7. **Create Connectivity Analysis Page** (New)
   - Method selection (PLV, wPLI, coherence, correlation)
   - Frequency band configuration
   - Network analysis options
   - Target: `5_connectivity.py`

8. **Create Features Extraction Page** (New - Phase 3)
   - Feature type selection (ERP, bandpower, TFA, connectivity, source)
   - Multi-feature combination
   - Cross-task feature integration
   - Target: `9_features.py`

9. **Create Prediction/Results Page** (New - Phase 3)
   - Model selection (CPM, Ridge, SVM, RF)
   - Feature selection options
   - Cross-validation configuration
   - Results visualization
   - Target: Use existing `_archived/results.py` as starting point?

---

## Design Decision Needed

**Question**: Should we merge pages or keep them separate?

**Option A: Keep Separate Pages** (Current partial implementation)
- Pros: Easier incremental updates, clear separation of concerns
- Cons: More navigation clicking, context switching

**Option B: Unified Processing Pipeline Page**
- Pros: All stages in one view, clear workflow progression
- Cons: Single page complexity, harder to maintain

**Recommendation**:
- **Short-term**: Keep separate pages for Phase 1 completion
- **Long-term**: Consider unified view after all stages implemented (Phase 4)

---

## File Locations

**Modified:**
- `/Users/clive/eegcpm/eegcpm-0.1/eegcpm/ui/app.py` - Home page redesign
- `/Users/clive/eegcpm/planning/UI_REDESIGN_PROPOSAL.md` - Updated with implementation status

**Renamed:**
- `pages/0_pipeline_config.py` → `pages/0_preprocessing_config.py`
- `pages/2_task_config.py` → `pages/1_task_config.py`
- `pages/4_run_selection.py` → `pages/6_run_selection.py`
- `pages/6_qc_browser.py` → `pages/7_qc_browser.py`
- `pages/7_group_qc_summary.py` → `pages/8_group_summary.py`

**Archived:**
- `pages/analysis.py` → `pages/_archived/analysis.py`
- `pages/hpc.py` → `pages/_archived/hpc.py`
- `pages/pipeline.py` → `pages/_archived/pipeline.py`
- `pages/project.py` → `pages/_archived/project.py`
- `pages/results.py` → `pages/_archived/results.py`

---

## Next Actions

1. **Test the redesigned home page**:
   ```bash
   streamlit run eegcpm/ui/app.py --server.port 8502
   ```
   - Verify pipeline progress tracking works
   - Check recent activity log
   - Test quick action buttons

2. **Update page titles and icons**:
   - Update all renamed pages with consistent `st.set_page_config()`
   - Use stage-based icons: 🔧 Configuration, ⚙️ Processing, 📊 QC, 📈 Results

3. **Decision on page merging strategy**:
   - User input needed: Merge preprocessing pages or keep separate?
   - Should Configuration pages be merged or kept as tabs?

4. **Continue with Phase 2** (after Phase 1 completion):
   - Implement missing stages (Source, Connectivity, Features, Prediction)
   - Create unified Processing Pipeline page with stage selector

---

## Testing Notes

**Before testing**, ensure:
1. Project has `derivatives/` folder with some processed data
2. BIDS dataset has subjects in `bids/sub-*/`
3. Some QC reports exist in `derivatives/*/qc/`

**Expected behavior**:
- Home page shows progress bars for each stage
- Recent activity shows last 5 QC report generations
- Quick actions navigate to appropriate pages

**Known limitations**:
- Pipeline progress is estimated from directory structure (not from workflow state DB)
- Recent activity only tracks QC report timestamps, not all processing events
- Quick action buttons mostly placeholders (need hookup to actual processing functions)
