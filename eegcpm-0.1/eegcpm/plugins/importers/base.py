"""Base importer interface for data format conversion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of importing a single subject/session."""

    subject_id: str
    success: bool
    output_files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class ImportSummary:
    """Summary of batch import operation."""

    total_subjects: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[ImportResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_subjects": self.total_subjects,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.successful / self.total_subjects if self.total_subjects > 0 else 0,
            "failed_subjects": [r.subject_id for r in self.results if not r.success and not r.skipped],
            "skipped_subjects": [r.subject_id for r in self.results if r.skipped],
        }

    def save(self, path: Path):
        """Save summary to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseImporter(ABC):
    """
    Abstract base class for data importers.

    Importers convert various data formats to BIDS-compatible structure.
    """

    name: str = "base"
    description: str = "Base importer"
    supported_formats: List[str] = []

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool = True,
        overwrite: bool = False,
    ):
        """
        Initialize importer.

        Args:
            source_dir: Directory containing source data
            output_dir: Output directory for BIDS data
            skip_existing: Skip subjects that are already converted
            overwrite: Overwrite existing files (only if skip_existing=False)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.skip_existing = skip_existing
        self.overwrite = overwrite

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def discover_subjects(self) -> List[str]:
        """
        Discover all subjects in the source directory.

        Returns:
            List of subject IDs
        """
        pass

    @abstractmethod
    def import_subject(
        self,
        subject_id: str,
        **kwargs,
    ) -> ImportResult:
        """
        Import a single subject's data.

        Args:
            subject_id: Subject identifier

        Returns:
            ImportResult with status and output files
        """
        pass

    def is_subject_complete(self, subject_id: str) -> bool:
        """
        Check if a subject has already been converted.

        Override in subclass for format-specific checks.

        Args:
            subject_id: Subject identifier

        Returns:
            True if subject data already exists in output
        """
        bids_id = self._to_bids_id(subject_id)
        subject_dir = self.output_dir / bids_id
        return subject_dir.exists() and any(subject_dir.glob("**/*_eeg.fif"))

    def _to_bids_id(self, subject_id: str) -> str:
        """Convert source subject ID to BIDS-compliant ID."""
        # Remove any existing prefix
        clean_id = subject_id.replace("sub-", "")
        return f"sub-{clean_id}"

    def import_all(
        self,
        subject_ids: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
        progress_callback=None,
    ) -> ImportSummary:
        """
        Import all subjects.

        Args:
            subject_ids: Specific subjects to import (default: all)
            max_subjects: Maximum number to import (for testing)
            progress_callback: Optional callback(current, total, subject_id)

        Returns:
            ImportSummary with results
        """
        if subject_ids is None:
            subject_ids = self.discover_subjects()

        if max_subjects:
            subject_ids = subject_ids[:max_subjects]

        summary = ImportSummary(total_subjects=len(subject_ids))

        for i, subject_id in enumerate(subject_ids):
            if progress_callback:
                progress_callback(i + 1, len(subject_ids), subject_id)

            # Check if should skip
            if self.skip_existing and self.is_subject_complete(subject_id):
                result = ImportResult(
                    subject_id=subject_id,
                    success=True,
                    skipped=True,
                    skip_reason="Already converted",
                )
                summary.skipped += 1
                summary.results.append(result)
                logger.info(f"Skipping {subject_id}: already converted")
                continue

            # Import subject
            try:
                result = self.import_subject(subject_id)
                if result.success:
                    summary.successful += 1
                    logger.info(f"Successfully imported {subject_id}")
                else:
                    summary.failed += 1
                    logger.warning(f"Failed to import {subject_id}: {result.errors}")
            except Exception as e:
                result = ImportResult(
                    subject_id=subject_id,
                    success=False,
                    errors=[str(e)],
                )
                summary.failed += 1
                logger.error(f"Error importing {subject_id}: {e}")

            summary.results.append(result)

        return summary

    def create_dataset_description(self, **kwargs) -> Dict[str, Any]:
        """
        Create BIDS dataset_description.json content.

        Override to customize for specific datasets.
        """
        return {
            "Name": kwargs.get("name", "Imported Dataset"),
            "BIDSVersion": "1.8.0",
            "DatasetType": "raw",
            "Authors": kwargs.get("authors", []),
            "License": kwargs.get("license", ""),
            "Acknowledgements": kwargs.get("acknowledgements", ""),
            "HowToAcknowledge": kwargs.get("how_to_acknowledge", ""),
            "SourceDatasets": [
                {
                    "URL": kwargs.get("source_url", ""),
                    "DOI": kwargs.get("source_doi", ""),
                }
            ],
            "GeneratedBy": [
                {
                    "Name": "EEGCPM",
                    "Version": "0.1.0",
                    "Description": f"Converted using {self.name} importer",
                }
            ],
        }

    def save_dataset_description(self, **kwargs):
        """Save dataset_description.json to output directory."""
        desc = self.create_dataset_description(**kwargs)
        desc_path = self.output_dir / "dataset_description.json"
        with open(desc_path, "w") as f:
            json.dump(desc, f, indent=2)
