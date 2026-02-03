"""
EEGCPM: EEG Connectome Predictive Modeling Toolbox

A comprehensive toolbox for EEG preprocessing, source reconstruction,
connectivity analysis, and predictive modeling.
"""

__version__ = "0.1.0"
__author__ = "EEGCPM Team"

from eegcpm.core.project import Project
from eegcpm.core.config import Config

__all__ = ["Project", "Config", "__version__"]
