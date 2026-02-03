# EEGCPM Architecture

**Status**: Implemented
**Version**: 0.1.0

---

## Overview

EEGCPM uses a **stage-first architecture** that organizes outputs by processing stage rather than pipeline variant. This enables flexible mix-and-match combinations and eliminates data duplication.

---

## Core Principles

### 1. Stage-First Organization

Derivatives are organized by processing stage:

```
derivatives/
├── preprocessing/{pipeline}/    # Stage 1: Multiple preprocessing variants
├── epochs/{preprocessing}/      # Stage 2: Depends on preprocessing
├── source/{preprocessing}/      # Stage 3: Depends on epochs
├── features/{preprocessing}/    # Stage 4: Depends on source
└── prediction/{model}/          # Stage 5: Depends on features
```

**Why Stage-First?**
- **Prevents waste**: One preprocessing → many source methods
- **Flexible combinations**: Mix preprocessing A + source B + features C
- **Clear dependencies**: Each stage explicitly depends on previous
- **Disk efficient**: No duplication when testing variants

### 2. Dependency Tracking

Each configuration declares dependencies:

```yaml
# epochs/standard.yaml
stage: epochs
depends_on:
  preprocessing: standard
  task: rest
```

Dependencies are validated before processing.

### 3. Centralized Path Management

`EEGCPMPaths` class provides single source of truth:

```python
from eegcpm.core.paths import EEGCPMPaths

paths = EEGCPMPaths(project_root="/data/study")

# All paths derived from single root
preproc_dir = paths.get_preprocessing_dir(
    pipeline="standard",
    subject="sub-001",
    session="01",
    task="rest",
    run="01"
)
```

---

## Directory Structure

### Project Layout

```
project_root/
├── bids/                        # BIDS input data
│   ├── dataset_description.json
│   ├── participants.tsv
│   └── sub-{ID}/
│       └── ses-{session}/
│           └── eeg/
│               ├── *_eeg.fif
│               ├── *_eeg.json
│               └── *_events.tsv
│
├── derivatives/                 # Stage-first processed outputs
│   ├── preprocessing/
│   ├── epochs/
│   ├── source/
│   ├── features/
│   ├── prediction/
│   └── behavioral/
│
└── eegcpm/                      # Workspace (not in derivatives)
    ├── configs/
    │   ├── preprocessing/
    │   ├── tasks/
    │   ├── source/
    │   └── features/
    └── .eegcpm/
        └── state.db             # Workflow tracking
```

### Stage-Specific Paths

**Preprocessing** (Stage 1):
```
derivatives/preprocessing/{pipeline}/sub-{ID}/ses-{session}/task-{task}/run-{run}/
├── *_preprocessed_raw.fif
├── *_ica.fif
├── *_preprocessed_qc.html
└── *_qc_metrics.json
```

**Epochs** (Stage 2):
```
derivatives/epochs/{preprocessing}/{task}/sub-{ID}/ses-{session}/
├── *_epo.fif
├── *_epochs_qc.html
└── *_qc_metrics.json
```

**Source** (Stage 3):
```
derivatives/source/{preprocessing}/{task}/variant-{method}-{template}/sub-{ID}/ses-{session}/
├── *_stc-lh.stc
├── *_stc-rh.stc
└── *_source_qc.html
```

**Features** (Stage 4):
```
derivatives/features/{preprocessing}/{task}/{source-variant}/{feature-type}/
├── features.csv
└── features_qc.html
```

**Prediction** (Stage 5):
```
derivatives/prediction/{model-name}/
├── config.yaml
├── features_combined.csv
├── predictions.csv
└── model_qc.html
```

---

## Module System

### BaseModule Pattern

All processing steps inherit from `BaseModule`:

```python
from eegcpm.pipeline.base import BaseModule, ModuleResult

class PreprocessingModule(BaseModule):
    name: str = "preprocessing"
    version: str = "0.1.0"

    def validate_input(self, data) -> bool:
        """Validate input data before processing."""
        pass

    def process(self, data, **kwargs) -> ModuleResult:
        """Execute processing logic."""
        pass

    def get_checkpoint_data(self) -> Dict:
        """Return data for checkpointing."""
        pass
```

### Specialized Base Classes

- `RawDataModule`: For modules operating on continuous EEG
- `EpochsModule`: For modules operating on epoched data
- `SourceModule`: For modules operating on source-reconstructed data

### Module Result

All modules return `ModuleResult`:

```python
@dataclass
class ModuleResult:
    success: bool
    data: Any                      # Processed output
    metadata: Dict[str, Any]       # Processing metrics
    warnings: List[str]            # User-facing warnings
    figures: Dict[str, bytes]      # QC plots (base64 PNG)
```

---

## Configuration System

### Pydantic Models

All configuration via Pydantic for validation:

```python
from eegcpm.core.config import PreprocessingConfig

config = PreprocessingConfig(
    steps=[
        {"name": "filter", "params": {"l_freq": 0.5, "h_freq": 40.0}},
        {"name": "bad_channels", "params": {"method": "prep"}},
        {"name": "ica", "params": {"method": "fastica", "n_components": "rank-1"}},
    ]
)
```

### Configuration Files

YAML configs in `eegcpm/configs/`:

```yaml
# preprocessing/standard.yaml
stage: preprocessing
pipeline: standard

steps:
  - name: montage
    params:
      type: GSN-HydroCel-129

  - name: filter
    params:
      l_freq: 0.5
      h_freq: 40.0
      method: fir

  - name: ica
    params:
      method: fastica
      n_components: rank-1
      l_freq_fit: 1.0
```

---

## Workflow State Tracking

### WorkflowState

Tracks processing progress in SQLite:

```python
@dataclass
class WorkflowState:
    subject_id: str
    session_id: Optional[str]
    task_name: Optional[str]
    run_id: Optional[str]
    pipeline: str
    current_stage: str              # NEW: preprocessing, epochs, source, ...
    status: str                     # pending, in_progress, completed, failed
    metadata: Dict[str, Any]        # Stage-specific data
```

### State Manager

```python
from eegcpm.workflow.state import WorkflowStateManager

state_db = eegcpm_root / ".eegcpm" / "state.db"
state_manager = WorkflowStateManager(state_db)

# Update state
state_manager.update_state(
    subject_id="sub-001",
    session_id="01",
    pipeline="standard",
    current_stage="preprocessing",
    status="completed",
    metadata={"n_channels": 129, "duration": 300.0}
)
```

---

## Pipeline Execution

### PipelineExecutor

Orchestrates module sequence:

```python
from eegcpm.pipeline.executor import PipelineExecutor

executor = PipelineExecutor(
    config=config,
    output_dir=output_dir,
    state_manager=state_manager
)

result = executor.execute(
    data=raw,
    subject_id="sub-001",
    session_id="01",
    task="rest"
)
```

### Checkpoint/Resume

```python
from eegcpm.pipeline.checkpoint import CheckpointManager

checkpoint = CheckpointManager(output_dir / "checkpoints")

# Save checkpoint
checkpoint.save(
    subject_id="sub-001",
    module_name="preprocessing",
    data=result.data,
    metadata=result.metadata
)

# Resume from checkpoint
data, metadata = checkpoint.load("sub-001", "preprocessing")
```

---

## Data Models

### Core Models

Defined in `eegcpm/core/models.py`:

```python
@dataclass
class Subject:
    id: str
    sessions: List[Session] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    id: str
    runs: List[Run] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Run:
    id: str
    task: str
    events: List[Event] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Event:
    onset: float
    duration: float
    trial_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Quality Control Integration

QC reports co-located with outputs at each stage:

```
derivatives/preprocessing/{pipeline}/sub-{ID}/ses-{session}/task-{task}/run-{run}/
├── *_preprocessed_raw.fif         # Data
├── *_preprocessed_qc.html         # QC here!
└── *_qc_metrics.json
```

**No separate `qc/` folder** - reports live with data they describe.

---

## Extensibility

### Adding New Modules

1. Inherit from appropriate base class
2. Implement required methods
3. Register in module registry
4. Add configuration schema

### Adding New Stages

1. Add stage-specific path method to `EEGCPMPaths`
2. Create base configuration class
3. Update dependency resolver
4. Add to workflow state schema

---

**See also:**
- Implementation: `eegcpm/core/`, `eegcpm/pipeline/`
- QC System: `planning/QC_SYSTEM.md`
- Workflows: `docs/WORKFLOWS.md`
