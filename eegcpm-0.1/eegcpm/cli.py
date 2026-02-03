"""
DEPRECATED: Legacy preprocessing-only CLI

This file has been moved to scripts/legacy_cli.py for reference.
The current CLI is implemented in eegcpm/cli/ directory.

Use the new CLI:
    eegcpm --help
    eegcpm preprocess --help
    eegcpm epochs --help
    eegcpm source-reconstruct --help
    eegcpm connectivity --help
"""

import warnings

warnings.warn(
    "eegcpm.cli module is deprecated. Use 'eegcpm' command or import from eegcpm.cli.main",
    DeprecationWarning,
    stacklevel=2
)

# Re-export main entry point for backward compatibility
from eegcpm.cli.main import main

__all__ = ['main']
