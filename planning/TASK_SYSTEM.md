# Task Configuration System

**Status**: In Development
**Location**: `eegcpm/ui/pages/2_task_config.py`

---

## Overview

Task-specific configuration system for defining how EEG data should be epoched, filtered, and analyzed based on experimental paradigms.

---

## Configuration Structure

### YAML Format

```yaml
# configs/tasks/saiit.yaml
task: saiit
description: "Selective Attention and Inhibition of Irrelevant Targets (2AFC)"

# Epoch timing
tmin: -0.2
tmax: 0.8
baseline: [-0.2, 0.0]

# Event conditions
conditions:
  target_left:
    event_codes: [11, 13]
    description: "Target appears on left"

  target_right:
    event_codes: [12, 14]
    description: "Target appears on right"

# Response mapping
responses:
  correct:
    response_codes: [201, 202]
    description: "Correct button press"

  incorrect:
    response_codes: [203, 204]
    description: "Incorrect button press"

  missing:
    response_codes: [0]
    description: "No response"

# Trial filtering
trial_filters:
  rt_min: 0.15          # Reject RT < 150ms (premature)
  rt_max: 2.0           # Reject RT > 2s (too slow)
  accuracy_filter: null # null = keep all, "correct" = only correct trials

# Epoch binning
binning:
  - type: stimulus_x_accuracy
    description: "Stimulus type × Response accuracy"

  - type: rt_quartiles
    description: "Reaction time quartiles"
```

---

## Epoch Timing

### Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `tmin` | float | Start time before event (s) | -0.2 |
| `tmax` | float | End time after event (s) | 0.8 |
| `baseline` | [float, float] | Baseline correction window | [-0.2, 0.0] |
| `reject` | dict \| null | Amplitude rejection thresholds | `{eeg: 150e-6}` |

### Baseline Correction Options

- `[null, 0]`: Use pre-stimulus period (tmin → 0)
- `[-0.2, 0.0]`: Specific window
- `null`: No baseline correction

---

## Experimental Conditions

### Condition Definition

```yaml
conditions:
  condition_name:
    event_codes: [1, 2, 3]       # Trigger codes
    description: "Human-readable"
    metadata:
      stimulus_type: "visual"
      difficulty: "easy"
```

### Event Code Scanning

UI can scan BIDS events to discover available triggers:

```python
from eegcpm.ui.pages.task_config import scan_events_from_bids

events_info = scan_events_from_bids(
    bids_root=Path("/data/bids"),
    task="saiit",
    max_subjects=10
)

# Returns:
# {
#   'event_names': Counter({'stimulus_left': 120, 'stimulus_right': 118}),
#   'columns': ['onset', 'duration', 'trial_type', 'response_time'],
#   'sampling_info': {'total_files': 30, 'total_events': 2400}
# }
```

---

## Response Mapping

### Response Categories

```yaml
responses:
  response_category:
    response_codes: [201, 202]
    rt_field: "response_time"        # Column in events.tsv
    description: "Category description"
```

### Linking Stimuli to Responses

```yaml
# Example: Match stimulus → response
conditions:
  correct_left:
    event_codes: [11]              # Left stimulus
    response_match:
      category: correct
      response_code: 201           # Left button
```

---

## Trial Filtering

### Rejection Criteria

#### 1. Reaction Time Filters

```yaml
trial_filters:
  rt_min: 0.15           # Reject premature responses (< 150ms)
  rt_max: 2.0            # Reject late responses (> 2s)
  rt_field: "response_time"
```

**Rationale**:
- RT < 150ms: Too fast to be genuine stimulus-driven response
- RT > 2000ms: Attention lapse, off-task behavior

#### 2. Accuracy Filter

```yaml
trial_filters:
  accuracy_filter: "correct"    # Only keep correct trials
  # Options: null (keep all), "correct", "incorrect"
```

#### 3. Missing Response Handling

```yaml
trial_filters:
  drop_missing_responses: true   # Drop trials with no response
```

#### 4. Sequential Position

```yaml
trial_filters:
  drop_first_n: 5               # Drop first 5 trials (practice)
  drop_last_n: 2                # Drop last 2 trials (fatigue)
```

---

## Epoch Binning Strategies

### 1. Stimulus × Accuracy

Cross stimulus type with response accuracy:

```yaml
binning:
  - type: stimulus_x_accuracy
    conditions: [target_left, target_right]
    responses: [correct, incorrect]
```

**Creates bins**:
- `target_left_correct`
- `target_left_incorrect`
- `target_right_correct`
- `target_right_incorrect`

### 2. Reaction Time Quartiles

Bin trials by RT distribution:

```yaml
binning:
  - type: rt_quartiles
    rt_field: "response_time"
    labels: [fast, medium_fast, medium_slow, slow]
```

**Creates bins**:
- Q1 (fastest 25%)
- Q2 (25-50%)
- Q3 (50-75%)
- Q4 (slowest 25%)

### 3. Stimulus × RT Interaction

Cross stimulus conditions with RT bins:

```yaml
binning:
  - type: stimulus_x_rt
    conditions: [target_left, target_right]
    rt_bins: 4
    rt_field: "response_time"
```

### 4. Sequential Position

Bin by trial position (e.g., learning effects):

```yaml
binning:
  - type: sequential_position
    bins: 5                       # 5 equal-sized blocks
    labels: [early, mid_early, middle, mid_late, late]
```

### 5. Custom Metadata

Bin by arbitrary metadata field:

```yaml
binning:
  - type: metadata_field
    field: "difficulty"
    values: [easy, medium, hard]
```

---

## Task-Specific Examples

### Resting State

```yaml
task: rest
description: "Eyes-open resting state"

tmin: 0.0
tmax: 2.0
baseline: null               # No baseline for continuous segments

# No event conditions - continuous data segmentation
segment_duration: 2.0        # 2-second windows
segment_overlap: 0.0         # No overlap
```

### Go/No-Go Task

```yaml
task: gonogo
description: "Response inhibition task"

tmin: -0.2
tmax: 1.0
baseline: [-0.2, 0.0]

conditions:
  go:
    event_codes: [10]
    description: "Go trials (respond)"

  nogo:
    event_codes: [20]
    description: "No-go trials (inhibit)"

responses:
  commission:
    event_codes: [30]
    description: "False alarm on no-go"

trial_filters:
  rt_min: 0.15
  rt_max: 1.0

binning:
  - type: trial_type
    bins: [go_correct, go_miss, nogo_correct, nogo_commission]
```

### N-Back Working Memory

```yaml
task: nback
description: "N-back working memory task"

tmin: -0.2
tmax: 2.0
baseline: [-0.2, 0.0]

conditions:
  target:
    event_codes: [1]
    description: "Target (match)"

  non_target:
    event_codes: [2]
    description: "Non-target (no match)"

binning:
  - type: stimulus_x_accuracy_x_load
    stimuli: [target, non_target]
    accuracy: [correct, incorrect]
    loads: [1, 2, 3]               # N-back load levels
```

---

## Implementation Plan

### Phase 1: Basic Epoch Configuration ✅

- [x] YAML config loading
- [x] Epoch timing parameters
- [x] Condition definitions
- [x] UI for basic configuration

### Phase 2: Response Mapping (In Progress)

- [ ] Response category definitions
- [ ] Stimulus-response linking
- [ ] RT field specification
- [ ] UI for response configuration

### Phase 3: Trial Filtering (Planned)

- [ ] RT min/max filters
- [ ] Accuracy filtering
- [ ] Missing response handling
- [ ] Sequential position filters
- [ ] UI filter configuration

### Phase 4: Epoch Binning (Planned)

- [ ] Stimulus × Accuracy binning
- [ ] RT quartile binning
- [ ] Sequential position binning
- [ ] Custom metadata binning
- [ ] UI binning specification

---

## API Reference

### Loading Task Config

```python
from pathlib import Path
import yaml

def load_task_config(task_name: str, config_dir: Path) -> dict:
    """Load task configuration from YAML."""
    config_path = config_dir / "tasks" / f"{task_name}.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

### Creating Epochs from Config

```python
import mne

def create_epochs_from_config(raw, config: dict):
    """Create MNE Epochs object from task config."""
    # Extract events
    events, event_id = mne.events_from_annotations(raw)

    # Create condition-specific event_id
    event_mapping = {}
    for cond_name, cond_def in config['conditions'].items():
        for code in cond_def['event_codes']:
            event_mapping[cond_name] = code

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_mapping,
        tmin=config['tmin'],
        tmax=config['tmax'],
        baseline=tuple(config['baseline']) if config['baseline'] else None,
        reject=config.get('reject'),
        preload=True
    )

    return epochs
```

---

**See also:**
- UI Implementation: `eegcpm/ui/pages/2_task_config.py`
- Architecture: `planning/ARCHITECTURE.md`
- Workflows: `docs/WORKFLOWS.md`
