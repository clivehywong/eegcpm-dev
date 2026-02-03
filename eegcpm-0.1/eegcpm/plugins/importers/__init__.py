"""Data importer plugins for converting various formats to BIDS."""

from eegcpm.plugins.importers.base import BaseImporter, ImportResult, ImportSummary
from eegcpm.plugins.importers.hbn import HBNImporter, convert_hbn_to_bids
from eegcpm.plugins.importers.hbn_behavioral import (
    BaseBehavioralParser,
    BehavioralResult,
    ParadigmSummary,
    TrialData,
    SAIIT2AFCParser,
    SurroundSuppParser,
    VisLearnParser,
    WISCProcSpeedParser,
    VideoParser,
    parse_behavioral_file,
    discover_behavioral_files,
)
from eegcpm.plugins.importers.hbn_events import (
    BIDSEvent,
    BIDSEventsWriter,
    trials_to_bids_events,
    write_events_tsv,
    write_events_json,
    write_trials_tsv,
)
from eegcpm.plugins.importers.hbn_summary import (
    BehavioralSummaryGenerator,
    PreprocessingQCGenerator,
    PreprocessingQC,
    generate_all_summaries,
)

__all__ = [
    # Base classes
    "BaseImporter",
    "ImportResult",
    "ImportSummary",
    # HBN importer
    "HBNImporter",
    "convert_hbn_to_bids",
    # Behavioral parsing
    "BaseBehavioralParser",
    "BehavioralResult",
    "ParadigmSummary",
    "TrialData",
    "SAIIT2AFCParser",
    "SurroundSuppParser",
    "VisLearnParser",
    "WISCProcSpeedParser",
    "VideoParser",
    "parse_behavioral_file",
    "discover_behavioral_files",
    # BIDS events
    "BIDSEvent",
    "BIDSEventsWriter",
    "trials_to_bids_events",
    "write_events_tsv",
    "write_events_json",
    "write_trials_tsv",
    # Summary generators
    "BehavioralSummaryGenerator",
    "PreprocessingQCGenerator",
    "PreprocessingQC",
    "generate_all_summaries",
]
