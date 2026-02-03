"""Analysis modules for EEGCPM pipeline."""

from eegcpm.modules.preprocessing import PreprocessingPipeline
from eegcpm.modules.epochs import EpochExtractionModule
from eegcpm.modules.source import SourceReconstructionModule
from eegcpm.modules.connectivity import ConnectivityModule
from eegcpm.modules.timefreq import TimeFrequencyModule
from eegcpm.modules.features import FeatureExtractionModule

__all__ = [
    "PreprocessingPipeline",
    "EpochExtractionModule",
    "SourceReconstructionModule",
    "ConnectivityModule",
    "TimeFrequencyModule",
    "FeatureExtractionModule",
]
