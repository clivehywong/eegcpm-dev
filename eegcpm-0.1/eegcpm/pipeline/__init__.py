"""Pipeline orchestration for EEGCPM."""

from eegcpm.pipeline.base import BaseModule, ModuleResult
from eegcpm.pipeline.executor import PipelineExecutor
from eegcpm.pipeline.checkpoint import CheckpointManager
# Avoid circular import: run_processor imports from modules.preprocessing
# from eegcpm.pipeline.run_processor import RunProcessor, RunProcessingResult, RunQualityMetrics
from eegcpm.pipeline.processing_state import ProcessingState, create_processing_state

__all__ = [
    "BaseModule",
    "ModuleResult",
    "PipelineExecutor",
    "CheckpointManager",
    # "RunProcessor",
    # "RunProcessingResult",
    # "RunQualityMetrics",
    "ProcessingState",
    "create_processing_state",
]
