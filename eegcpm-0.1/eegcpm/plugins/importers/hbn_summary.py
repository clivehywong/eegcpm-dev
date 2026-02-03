"""
Summary table generators for HBN behavioral and QC data.

Generates:
- Per-paradigm group summaries (all subjects × metrics)
- Cross-paradigm usability matrix
- Preprocessing QC tables
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eegcpm.plugins.importers.hbn_behavioral import (
    BehavioralResult,
    ParadigmSummary,
    discover_behavioral_files,
    parse_behavioral_file,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingQC:
    """QC metrics for preprocessing."""

    subject_id: str
    paradigm: str
    block: Optional[str] = None

    # Raw data QC
    n_channels_total: int = 0
    n_bad_channels: int = 0
    bad_channels: List[str] = field(default_factory=list)

    # Recording duration
    duration_seconds: float = 0.0
    sfreq: float = 500.0

    # Epochs QC (if epoched)
    n_epochs_total: Optional[int] = None
    n_epochs_rejected: Optional[int] = None
    n_epochs_kept: Optional[int] = None
    rejection_rate: Optional[float] = None

    # ICA QC (if applied)
    n_ica_components: Optional[int] = None
    n_ica_excluded: Optional[int] = None
    ica_excluded: List[int] = field(default_factory=list)

    # Overall usability
    usable: bool = True
    exclusion_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject_id": self.subject_id,
            "paradigm": self.paradigm,
            "block": self.block,
            "n_channels_total": self.n_channels_total,
            "n_bad_channels": self.n_bad_channels,
            "bad_channels": ";".join(self.bad_channels) if self.bad_channels else None,
            "duration_seconds": self.duration_seconds,
            "sfreq": self.sfreq,
            "n_epochs_total": self.n_epochs_total,
            "n_epochs_rejected": self.n_epochs_rejected,
            "n_epochs_kept": self.n_epochs_kept,
            "rejection_rate": self.rejection_rate,
            "n_ica_components": self.n_ica_components,
            "n_ica_excluded": self.n_ica_excluded,
            "ica_excluded": ";".join(map(str, self.ica_excluded)) if self.ica_excluded else None,
            "usable": self.usable,
            "exclusion_reasons": ";".join(self.exclusion_reasons) if self.exclusion_reasons else None,
        }


class BehavioralSummaryGenerator:
    """Generate behavioral summary tables across subjects."""

    def __init__(self, source_dir: Path, output_dir: Path):
        """
        Initialize generator.

        Args:
            source_dir: HBN raw data directory
            output_dir: Output directory for summaries
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_all_behavioral_data(
        self,
        subjects: Optional[List[str]] = None,
        paradigms: Optional[List[str]] = None,
    ) -> Dict[str, List[BehavioralResult]]:
        """
        Collect behavioral data for all subjects/paradigms.

        Returns:
            Dictionary mapping paradigm -> list of BehavioralResult
        """
        results_by_paradigm: Dict[str, List[BehavioralResult]] = {}

        # Discover subjects if not provided
        if subjects is None:
            raw_dir = self.source_dir / "raw"
            if not raw_dir.exists():
                raw_dir = self.source_dir
            subjects = [d.name for d in sorted(raw_dir.glob("NDAR*")) if d.is_dir()]

        for subject_id in subjects:
            subject_dir = self._get_subject_dir(subject_id)
            if not subject_dir.exists():
                continue

            # Discover behavioral files
            beh_files = discover_behavioral_files(subject_dir, subject_id)

            for mat_file, paradigm, block in beh_files:
                # Filter by paradigm if specified
                if paradigms and paradigm not in paradigms:
                    continue

                # Parse behavioral data
                result = parse_behavioral_file(mat_file, subject_id, paradigm, block)

                # Store by paradigm
                if paradigm not in results_by_paradigm:
                    results_by_paradigm[paradigm] = []
                results_by_paradigm[paradigm].append(result)

        return results_by_paradigm

    def _get_subject_dir(self, subject_id: str) -> Path:
        """Get subject source directory."""
        raw_dir = self.source_dir / "raw"
        if not raw_dir.exists():
            raw_dir = self.source_dir
        return raw_dir / subject_id

    def generate_paradigm_summary(
        self,
        paradigm: str,
        results: List[BehavioralResult],
    ) -> pd.DataFrame:
        """
        Generate summary table for a single paradigm.

        Args:
            paradigm: Paradigm name
            results: List of behavioral results for this paradigm

        Returns:
            DataFrame with one row per subject/block
        """
        records = []
        for result in results:
            if result.summary:
                records.append(result.summary.to_dict())
            else:
                # Create minimal record for failed parsing
                records.append({
                    "subject_id": result.subject_id,
                    "paradigm": paradigm,
                    "block": None,
                    "n_trials": 0,
                    "usable": False,
                    "exclusion_reasons": "; ".join(result.errors),
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Sort by subject_id and block
        if "block" in df.columns:
            df = df.sort_values(["subject_id", "block"]).reset_index(drop=True)
        else:
            df = df.sort_values("subject_id").reset_index(drop=True)

        return df

    def generate_usability_matrix(
        self,
        results_by_paradigm: Dict[str, List[BehavioralResult]],
    ) -> pd.DataFrame:
        """
        Generate cross-paradigm usability matrix.

        Returns:
            DataFrame with subjects as rows, paradigms as columns,
            values indicating usability (1/0/NaN)
        """
        # Collect unique subjects and paradigms
        subjects = set()
        paradigms = set()
        usability = {}  # (subject, paradigm) -> usable

        for paradigm, results in results_by_paradigm.items():
            paradigms.add(paradigm)
            for result in results:
                subjects.add(result.subject_id)
                key = (result.subject_id, paradigm)
                # If multiple blocks, use AND logic
                if key in usability:
                    if result.summary:
                        usability[key] = usability[key] and result.summary.usable
                else:
                    usability[key] = result.summary.usable if result.summary else False

        # Build matrix
        subjects = sorted(subjects)
        paradigms = sorted(paradigms)

        data = []
        for subj in subjects:
            row = {"subject_id": subj}
            for para in paradigms:
                key = (subj, para)
                row[para] = usability.get(key, np.nan)
            data.append(row)

        df = pd.DataFrame(data)

        # Add summary columns
        paradigm_cols = [c for c in df.columns if c != "subject_id"]
        df["n_paradigms_available"] = df[paradigm_cols].notna().sum(axis=1)
        df["n_paradigms_usable"] = (df[paradigm_cols] == True).sum(axis=1)  # noqa: E712

        return df

    def generate_all_summaries(
        self,
        subjects: Optional[List[str]] = None,
        paradigms: Optional[List[str]] = None,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all summary tables.

        Args:
            subjects: List of subject IDs (optional, discovers all)
            paradigms: List of paradigms (optional, discovers all)
            save: Save to files

        Returns:
            Dictionary of DataFrames
        """
        logger.info("Collecting behavioral data...")
        results_by_paradigm = self.collect_all_behavioral_data(subjects, paradigms)

        outputs = {}

        # Per-paradigm summaries
        for paradigm, results in results_by_paradigm.items():
            logger.info(f"Generating summary for {paradigm} ({len(results)} entries)")
            df = self.generate_paradigm_summary(paradigm, results)
            outputs[f"paradigm_{paradigm}"] = df

            if save and len(df) > 0:
                path = self.output_dir / f"behavioral_summary_{paradigm}.csv"
                df.to_csv(path, index=False)
                logger.info(f"Saved {path}")

        # Usability matrix
        logger.info("Generating usability matrix...")
        matrix_df = self.generate_usability_matrix(results_by_paradigm)
        outputs["usability_matrix"] = matrix_df

        if save and len(matrix_df) > 0:
            path = self.output_dir / "behavioral_usability_matrix.csv"
            matrix_df.to_csv(path, index=False)
            logger.info(f"Saved {path}")

        # Overall statistics
        stats = self._compute_overall_stats(results_by_paradigm, matrix_df)
        outputs["statistics"] = stats

        if save:
            path = self.output_dir / "behavioral_statistics.json"
            with open(path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved {path}")

        return outputs

    def _compute_overall_stats(
        self,
        results_by_paradigm: Dict[str, List[BehavioralResult]],
        matrix_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute overall statistics."""
        stats = {
            "n_subjects": len(matrix_df),
            "paradigms": {},
        }

        for paradigm, results in results_by_paradigm.items():
            n_total = len(results)
            n_usable = sum(1 for r in results if r.summary and r.summary.usable)
            n_failed = sum(1 for r in results if not r.success)

            # Aggregate metrics
            accuracies = [r.summary.accuracy for r in results if r.summary and r.summary.accuracy is not None]
            rts = [r.summary.mean_rt for r in results if r.summary and r.summary.mean_rt is not None]

            stats["paradigms"][paradigm] = {
                "n_total": n_total,
                "n_usable": n_usable,
                "n_failed": n_failed,
                "usability_rate": n_usable / n_total if n_total > 0 else 0,
                "mean_accuracy": float(np.mean(accuracies)) if accuracies else None,
                "std_accuracy": float(np.std(accuracies)) if accuracies else None,
                "mean_rt": float(np.mean(rts)) if rts else None,
                "std_rt": float(np.std(rts)) if rts else None,
            }

        # Cross-paradigm stats
        if "n_paradigms_usable" in matrix_df.columns:
            stats["cross_paradigm"] = {
                "mean_paradigms_per_subject": float(matrix_df["n_paradigms_available"].mean()),
                "mean_usable_paradigms_per_subject": float(matrix_df["n_paradigms_usable"].mean()),
                "subjects_with_all_usable": int((matrix_df["n_paradigms_usable"] == matrix_df["n_paradigms_available"]).sum()),
            }

        return stats


class PreprocessingQCGenerator:
    """Generate preprocessing QC tables."""

    def __init__(self, bids_dir: Path, output_dir: Path):
        """
        Initialize generator.

        Args:
            bids_dir: BIDS root directory
            output_dir: Output directory for QC reports
        """
        self.bids_dir = Path(bids_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_qc_from_fif(
        self,
        fif_file: Path,
        subject_id: str,
        paradigm: str,
        block: Optional[str] = None,
    ) -> PreprocessingQC:
        """
        Extract QC metrics from a FIF file.

        Args:
            fif_file: Path to FIF file
            subject_id: Subject ID
            paradigm: Paradigm name
            block: Block identifier

        Returns:
            PreprocessingQC object
        """
        import mne

        qc = PreprocessingQC(
            subject_id=subject_id,
            paradigm=paradigm,
            block=block,
        )

        try:
            raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)

            qc.n_channels_total = len(raw.ch_names)
            qc.sfreq = raw.info["sfreq"]
            qc.duration_seconds = raw.times[-1]

            # Get bad channels
            qc.bad_channels = list(raw.info["bads"])
            qc.n_bad_channels = len(qc.bad_channels)

            # Check for excessive bad channels
            bad_rate = qc.n_bad_channels / qc.n_channels_total
            if bad_rate > 0.1:  # >10% bad channels
                qc.exclusion_reasons.append(f"High bad channel rate: {bad_rate:.1%}")

            # Check duration
            if qc.duration_seconds < 60:  # Less than 1 minute
                qc.exclusion_reasons.append(f"Short recording: {qc.duration_seconds:.1f}s")

            qc.usable = len(qc.exclusion_reasons) == 0

        except Exception as e:
            qc.usable = False
            qc.exclusion_reasons.append(f"Error reading file: {e}")
            logger.warning(f"Error reading {fif_file}: {e}")

        return qc

    def collect_all_qc(
        self,
        subjects: Optional[List[str]] = None,
    ) -> List[PreprocessingQC]:
        """
        Collect QC metrics for all subjects.

        Returns:
            List of PreprocessingQC objects
        """
        qc_list = []

        # Find all subject directories
        if subjects is None:
            subject_dirs = sorted(self.bids_dir.glob("sub-*"))
        else:
            subject_dirs = [self.bids_dir / f"sub-{s.replace('sub-', '')}" for s in subjects]

        for subject_dir in subject_dirs:
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            # Find all FIF files
            for fif_file in subject_dir.glob("**/eeg/*_eeg.fif"):
                # Parse paradigm from filename
                # Format: sub-*_ses-*_task-*_eeg.fif
                fname = fif_file.stem
                parts = fname.split("_")
                paradigm = None
                for part in parts:
                    if part.startswith("task-"):
                        paradigm = part.replace("task-", "")
                        break

                if paradigm is None:
                    paradigm = "unknown"

                # Check for block in paradigm name
                block = None
                for suffix in ["1", "2", "3", "4", "block1", "block2", "block3", "block4"]:
                    if paradigm.endswith(suffix):
                        block = suffix
                        paradigm = paradigm[: -len(suffix)]
                        break

                qc = self.collect_qc_from_fif(fif_file, subject_id, paradigm, block)
                qc_list.append(qc)

        return qc_list

    def generate_qc_summary(
        self,
        qc_list: List[PreprocessingQC],
    ) -> pd.DataFrame:
        """
        Generate QC summary table.

        Args:
            qc_list: List of PreprocessingQC objects

        Returns:
            DataFrame with QC metrics
        """
        records = [qc.to_dict() for qc in qc_list]
        df = pd.DataFrame(records)

        if len(df) > 0:
            df = df.sort_values(["subject_id", "paradigm", "block"]).reset_index(drop=True)

        return df

    def generate_qc_report(
        self,
        subjects: Optional[List[str]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate full QC report.

        Args:
            subjects: List of subjects (optional)
            save: Save to files

        Returns:
            Dictionary with QC data and statistics
        """
        logger.info("Collecting preprocessing QC metrics...")
        qc_list = self.collect_all_qc(subjects)

        # Generate summary table
        df = self.generate_qc_summary(qc_list)

        # Compute statistics
        stats = {}
        if len(df) > 0:
            stats = {
                "n_recordings": len(df),
                "n_usable": int(df["usable"].sum()),
                "n_excluded": int((~df["usable"]).sum()),
                "usability_rate": float(df["usable"].mean()),
                "mean_bad_channels": float(df["n_bad_channels"].mean()),
                "mean_duration": float(df["duration_seconds"].mean()),
                "paradigm_counts": df["paradigm"].value_counts().to_dict(),
            }

        outputs = {
            "qc_table": df,
            "statistics": stats,
        }

        if save:
            # Save table
            table_path = self.output_dir / "preprocessing_qc.csv"
            df.to_csv(table_path, index=False)
            logger.info(f"Saved {table_path}")

            # Save statistics
            stats_path = self.output_dir / "preprocessing_qc_stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved {stats_path}")

        return outputs


def generate_all_summaries(
    source_dir: str,
    bids_dir: str,
    output_dir: str,
    subjects: Optional[List[str]] = None,
    paradigms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to generate all summary tables.

    Args:
        source_dir: HBN raw data directory
        bids_dir: BIDS output directory
        output_dir: Summary output directory
        subjects: Optional list of subjects
        paradigms: Optional list of paradigms

    Returns:
        Dictionary with all outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Behavioral summaries
    beh_gen = BehavioralSummaryGenerator(
        source_dir=Path(source_dir),
        output_dir=output_dir / "behavioral",
    )
    outputs["behavioral"] = beh_gen.generate_all_summaries(subjects, paradigms)

    # Preprocessing QC
    qc_gen = PreprocessingQCGenerator(
        bids_dir=Path(bids_dir),
        output_dir=output_dir / "qc",
    )
    outputs["preprocessing_qc"] = qc_gen.generate_qc_report(subjects)

    logger.info("Summary generation complete")
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HBN summary tables")
    parser.add_argument("source_dir", help="HBN raw data directory")
    parser.add_argument("bids_dir", help="BIDS output directory")
    parser.add_argument("output_dir", help="Summary output directory")
    parser.add_argument("--subjects", nargs="+", help="Subjects to process")
    parser.add_argument("--paradigms", nargs="+", help="Paradigms to include")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generate_all_summaries(
        source_dir=args.source_dir,
        bids_dir=args.bids_dir,
        output_dir=args.output_dir,
        subjects=args.subjects,
        paradigms=args.paradigms,
    )
