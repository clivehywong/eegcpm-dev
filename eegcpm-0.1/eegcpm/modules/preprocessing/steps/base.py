"""
Base class for preprocessing steps.

Author: EEGCPM Development Team
Created: 2025-12
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import mne


class ProcessingStep(ABC):
    """
    Base class for a single preprocessing step.

    All preprocessing steps inherit from this class and implement
    the process() method. Steps are composable and can be chained
    together to form complete preprocessing pipelines.

    Attributes
    ----------
    name : str
        Step name (used in config and metadata)
    version : str
        Step version for reproducibility
    enabled : bool
        Whether this step is enabled (default: True)

    Examples
    --------
    Create a custom step:
    >>> class MyCustomStep(ProcessingStep):
    ...     name = "my_step"
    ...     version = "1.0"
    ...
    ...     def __init__(self, param1=10):
    ...         self.param1 = param1
    ...
    ...     def process(self, raw, metadata):
    ...         # Do something with raw
    ...         raw_processed = raw.copy()
    ...         step_meta = {'param1': self.param1}
    ...         return raw_processed, step_meta
    """

    name: str = "processing_step"
    version: str = "1.0"

    def __init__(self, enabled: bool = True, **kwargs):
        """
        Initialize processing step.

        Parameters
        ----------
        enabled : bool
            Whether this step is enabled (default: True)
        **kwargs : dict
            Step-specific parameters
        """
        self.enabled = enabled

    @abstractmethod
    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Process raw EEG data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw EEG data
        metadata : dict
            Accumulated metadata from previous steps.
            This allows steps to access information from earlier processing
            (e.g., ICLabel can access ICA components from ICAStep)

        Returns
        -------
        raw : mne.io.BaseRaw
            Processed raw EEG data
        step_metadata : dict
            Metadata specific to this step (will be stored under metadata[step.name])

        Raises
        ------
        ValueError
            If input data is invalid
        RuntimeError
            If processing fails
        """
        pass

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """
        Validate that input data is suitable for this step.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input data to validate

        Returns
        -------
        valid : bool
            True if data is valid for this step

        Raises
        ------
        ValueError
            If validation fails with specific reason
        """
        # Default: accept all data
        return True

    def skip_step(self, raw: mne.io.BaseRaw, metadata: Dict[str, Any]) -> bool:
        """
        Determine if this step should be skipped based on current state.

        Useful for conditional execution (e.g., skip ICA if too few channels).

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Current raw data
        metadata : dict
            Current metadata

        Returns
        -------
        skip : bool
            True if step should be skipped
        """
        return not self.enabled

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration as dictionary.

        Returns
        -------
        config : dict
            Configuration dictionary that can reproduce this step
        """
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
