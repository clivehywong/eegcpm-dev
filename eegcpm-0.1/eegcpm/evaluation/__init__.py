"""Evaluation module for EEGCPM prediction."""

from eegcpm.evaluation.prediction import CPMPredictor
from eegcpm.evaluation.models import ModelFactory
from eegcpm.evaluation.metrics import compute_metrics, compare_models

__all__ = [
    "CPMPredictor",
    "ModelFactory",
    "compute_metrics",
    "compare_models",
]
