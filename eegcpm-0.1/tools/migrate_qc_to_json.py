#!/usr/bin/env python3
"""Migrate existing QC data to standardized JSON format.

This script scans derivatives directories for existing preprocessed runs
and generates QC metrics JSON files from workflow state database.

Usage:
    python tools/migrate_qc_to_json.py --derivatives /path/to/derivatives --state-db /path/to/state.db
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eegcpm.workflow.state import WorkflowStateManager, ProcessingStatus
from eegcpm.modules.qc.quality_assessment import assess_quality_status, get_recommended_action
from eegcpm.modules.qc.metrics_io import save_qc_metrics_json
from datetime import datetime


def migrate_qc_data(derivatives_path: Path, state_db_path: Path, dry_run: bool = False):
    """
    Migrate QC data from workflow state to JSON files.

    Parameters
    ----------
    derivatives_path : Path
        Path to derivatives directory
    state_db_path : Path
        Path to workflow state database
    dry_run : bool
        If True, print what would be done without writing files
    """
    derivatives_path = Path(derivatives_path)
    state_db_path = Path(state_db_path)

    if not state_db_path.exists():
        print(f"Error: State database not found: {state_db_path}")
        return

    print(f"Loading workflow state from: {state_db_path}")
    state_manager = WorkflowStateManager(state_db_path)

    # Get all completed workflows
    summary = state_manager.get_summary()
    print(f"\nDatabase summary:")
    print(f"  Total workflows: {summary.get('total_workflows', 0)}")
    print(f"  Completed: {summary['status_counts'].get('completed', 0)}")
    print(f"  Failed: {summary['status_counts'].get('failed', 0)}")

    # Scan for all QC HTML reports in derivatives
    qc_reports = sorted(derivatives_path.glob("**/ses-*/task-*/run-*/*_preprocessed_qc.html"))
    print(f"\nFound {len(qc_reports)} QC reports in derivatives")

    migrated = 0
    skipped = 0
    failed = 0

    for qc_report in qc_reports:
        # Parse path components
        # Format: {subject}/ses-{session}/task-{task}/run-{run}/{subject}_ses-{session}_task-{task}_run-{run}_preprocessed_qc.html
        parts = qc_report.parts
        run_dir = qc_report.parent
        task_dir = run_dir.parent
        session_dir = task_dir.parent
        subject_dir = session_dir.parent

        subject_id = subject_dir.name
        session = session_dir.name.replace('ses-', '')
        task = task_dir.name.replace('task-', '')
        run = run_dir.name.replace('run-', '')

        # Check if JSON already exists
        json_filename = f"{subject_id}_ses-{session}_task-{task}_run-{run}_qc_metrics.json"
        json_path = run_dir / json_filename

        if json_path.exists() and not dry_run:
            skipped += 1
            continue

        # Try to load workflow state
        state = state_manager.load_state(
            subject_id=subject_id,
            task=task,
            pipeline='standard',  # Assume standard pipeline
            session=session,
            run=run
        )

        if not state or state.status != ProcessingStatus.COMPLETED:
            print(f"  Warning: No completed workflow state for {subject_id} run-{run}")
            failed += 1
            continue

        # Extract preprocessing metadata from steps
        preproc_metadata = None
        for step in state.steps:
            if step.step_name == 'preprocessing' and step.metadata:
                preproc_metadata = step.metadata
                break

        if not preproc_metadata:
            print(f"  Warning: No preprocessing metadata for {subject_id} run-{run}")
            failed += 1
            continue

        # Extract metrics from preprocessing metadata
        bad_channels_step = preproc_metadata.get('bad_channels', {})
        n_bad = bad_channels_step.get('n_bad_channels', 0)
        n_original = 128  # Default assumption

        # Try to get n_original from montage step
        montage_step = preproc_metadata.get('montage', {})
        if montage_step.get('n_channels_before'):
            n_original = montage_step['n_channels_before']

        pct_bad = (n_bad / n_original * 100) if n_original > 0 else 0

        # ICA info
        ica_step = preproc_metadata.get('ica', {})
        ica_success = ica_step.get('applied', False)
        n_ica_components = ica_step.get('n_components', 0)

        iclabel_step = preproc_metadata.get('iclabel', {})
        n_components_rejected = iclabel_step.get('n_components_removed', 0)

        # Clustering (placeholder - not in current metadata)
        clustering_severity = 'none'
        n_clustered_bad = 0

        # Assess quality
        quality_status = assess_quality_status(pct_bad, clustering_severity)
        recommended_action = get_recommended_action(quality_status, ica_success)

        # Build JSON data
        metrics_data = {
            'subject_id': subject_id,
            'session': session,
            'task': task,
            'run': run,
            'pipeline': 'standard',
            'timestamp': datetime.now().isoformat(),
            'quality_metrics': {
                'quality_status': quality_status,
                'pct_bad_channels': round(pct_bad, 2),
                'n_bad_channels': n_bad,
                'n_eeg_channels': n_original,
                'clustering_severity': clustering_severity,
                'n_clustered_bad': n_clustered_bad,
                'ica_success': ica_success,
                'n_ica_components': n_ica_components,
                'n_components_rejected': n_components_rejected,
                'recommended_action': recommended_action,
            },
            'qc_report_path': qc_report.name
        }

        if dry_run:
            print(f"  Would create: {json_path}")
            print(f"    Quality: {quality_status}, Action: {recommended_action}")
        else:
            save_qc_metrics_json(metrics_data, json_path)
            print(f"  ✓ {subject_id} run-{run}: {quality_status}")
            migrated += 1

    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Migrate QC data to JSON format")
    parser.add_argument('--derivatives', type=Path, required=True,
                       help='Path to derivatives directory')
    parser.add_argument('--state-db', type=Path, required=True,
                       help='Path to workflow state database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without writing files')

    args = parser.parse_args()

    migrate_qc_data(args.derivatives, args.state_db, args.dry_run)


if __name__ == '__main__':
    main()
