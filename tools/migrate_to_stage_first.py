#!/usr/bin/env python3
"""Migration script for stage-first architecture.

This script migrates data from the old pipeline-centric structure to the new
stage-first architecture.

Old structure:
    eegcpm/pipelines/{pipeline}/{subject}/ses-{session}/task-{task}/run-{run}/
    derivatives/pipeline-{pipeline}/

New structure:
    derivatives/preprocessing/{pipeline}/{subject}/ses-{session}/task-{task}/run-{run}/
    eegcpm/.eegcpm/state.db (moved from derivatives/.eegcpm/state.db)

Usage:
    python tools/migrate_to_stage_first.py --project /path/to/project --dry-run
    python tools/migrate_to_stage_first.py --project /path/to/project --execute
"""

import argparse
import shutil
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def scan_old_structure(project_root: Path, eegcpm_root: Path) -> List[Dict[str, Any]]:
    """
    Scan for old pipeline structure.

    Returns list of dicts with:
        - old_path: Path to old data
        - new_path: Path where it should be moved
        - type: "preprocessing", "qc", etc.
    """
    migrations = []

    # Scan old eegcpm/pipelines/ structure
    pipelines_dir = eegcpm_root / "pipelines"
    if pipelines_dir.exists():
        for pipeline_dir in pipelines_dir.iterdir():
            if not pipeline_dir.is_dir() or pipeline_dir.name.startswith('.'):
                continue

            pipeline = pipeline_dir.name

            # Scan subjects
            for subject_dir in pipeline_dir.iterdir():
                if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
                    continue

                subject = subject_dir.name

                # Scan sessions
                for session_dir in subject_dir.glob("ses-*"):
                    if not session_dir.is_dir():
                        continue

                    session = session_dir.name.replace('ses-', '')

                    # Scan tasks
                    for task_dir in session_dir.glob("task-*"):
                        if not task_dir.is_dir():
                            continue

                        task = task_dir.name.replace('task-', '')

                        # Scan runs
                        for run_dir in task_dir.glob("run-*"):
                            if not run_dir.is_dir():
                                continue

                            run = run_dir.name.replace('run-', '')

                            # New path
                            new_path = (
                                project_root / "derivatives" / "preprocessing" / pipeline /
                                subject / f"ses-{session}" / f"task-{task}" / f"run-{run}"
                            )

                            migrations.append({
                                'old_path': run_dir,
                                'new_path': new_path,
                                'type': 'preprocessing',
                                'pipeline': pipeline,
                                'subject': subject,
                                'session': session,
                                'task': task,
                                'run': run
                            })

    # Scan old derivatives/pipeline-{name} structure
    derivatives_dir = project_root / "derivatives"
    if derivatives_dir.exists():
        for pipeline_dir in derivatives_dir.glob("pipeline-*"):
            if not pipeline_dir.is_dir():
                continue

            pipeline = pipeline_dir.name.replace('pipeline-', '')

            # Scan subjects
            for subject_dir in pipeline_dir.iterdir():
                if not subject_dir.is_dir() or not subject_dir.name.startswith('sub-'):
                    continue

                subject = subject_dir.name

                # Similar scanning as above...
                for session_dir in subject_dir.glob("ses-*"):
                    if not session_dir.is_dir():
                        continue

                    session = session_dir.name.replace('ses-', '')

                    for task_dir in session_dir.glob("task-*"):
                        if not task_dir.is_dir():
                            continue

                        task = task_dir.name.replace('task-', '')

                        for run_dir in task_dir.glob("run-*"):
                            if not run_dir.is_dir():
                                continue

                            run = run_dir.name.replace('run-', '')

                            new_path = (
                                project_root / "derivatives" / "preprocessing" / pipeline /
                                subject / f"ses-{session}" / f"task-{task}" / f"run-{run}"
                            )

                            migrations.append({
                                'old_path': run_dir,
                                'new_path': new_path,
                                'type': 'preprocessing',
                                'pipeline': pipeline,
                                'subject': subject,
                                'session': session,
                                'task': task,
                                'run': run
                            })

    return migrations


def migrate_state_db(
    old_db: Path,
    new_db: Path,
    dry_run: bool = False
) -> bool:
    """
    Migrate state database to new location.

    Returns True if successful.
    """
    if not old_db.exists():
        print(f"⚠️  Old state DB not found: {old_db}")
        return False

    if new_db.exists():
        print(f"✓ New state DB already exists: {new_db}")
        return True

    if dry_run:
        print(f"[DRY RUN] Would copy state DB:")
        print(f"  From: {old_db}")
        print(f"  To: {new_db}")
        return True

    # Copy database
    new_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_db, new_db)
    print(f"✓ Migrated state DB to: {new_db}")

    # Update schema for new columns (handled by WorkflowStateManager._init_db)
    return True


def execute_migration(
    migrations: List[Dict[str, Any]],
    dry_run: bool = False,
    create_symlinks: bool = False
) -> Dict[str, int]:
    """
    Execute the migration.

    Parameters
    ----------
    migrations : list
        List of migration operations
    dry_run : bool
        If True, only print what would be done
    create_symlinks : bool
        If True, create symlinks at old location pointing to new location

    Returns
    -------
    dict
        Summary with counts of success, skipped, failed
    """
    stats = {
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }

    for migration in migrations:
        old_path = migration['old_path']
        new_path = migration['new_path']

        # Check if already exists
        if new_path.exists():
            print(f"⏭  Skipped (exists): {new_path}")
            stats['skipped'] += 1
            continue

        if dry_run:
            print(f"[DRY RUN] Would move:")
            print(f"  From: {old_path}")
            print(f"  To: {new_path}")
            stats['success'] += 1
            continue

        try:
            # Create parent directory
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # Move data
            shutil.move(str(old_path), str(new_path))
            print(f"✓ Moved: {old_path.name} → {new_path}")

            # Create symlink for backward compatibility
            if create_symlinks:
                try:
                    old_path.symlink_to(new_path)
                    print(f"  ↪ Created symlink: {old_path} → {new_path}")
                except Exception as e:
                    print(f"  ⚠️  Could not create symlink: {e}")

            stats['success'] += 1

        except Exception as e:
            print(f"✗ Failed to move {old_path}: {e}")
            stats['failed'] += 1
            stats['errors'].append(str(e))

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate EEGCPM data to stage-first architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be migrated
  python tools/migrate_to_stage_first.py --project /data/study --dry-run

  # Execute migration
  python tools/migrate_to_stage_first.py --project /data/study --execute

  # Execute with backward-compatibility symlinks
  python tools/migrate_to_stage_first.py --project /data/study --execute --symlinks
        """
    )

    parser.add_argument(
        '--project',
        type=Path,
        required=True,
        help='Project root directory'
    )
    parser.add_argument(
        '--eegcpm-root',
        type=Path,
        help='EEGCPM workspace directory (default: project/eegcpm)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without executing'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute the migration'
    )
    parser.add_argument(
        '--symlinks',
        action='store_true',
        help='Create symlinks at old locations for backward compatibility'
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        parser.error("Must specify either --dry-run or --execute")

    project_root = Path(args.project)
    eegcpm_root = Path(args.eegcpm_root) if args.eegcpm_root else project_root / "eegcpm"

    if not project_root.exists():
        print(f"❌ Project root not found: {project_root}")
        return 1

    print("=" * 80)
    print("EEGCPM Stage-First Architecture Migration")
    print("=" * 80)
    print(f"Project: {project_root}")
    print(f"EEGCPM: {eegcpm_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print("=" * 80)
    print()

    # Scan for data to migrate
    print("📂 Scanning for old structure...")
    migrations = scan_old_structure(project_root, eegcpm_root)

    if not migrations:
        print("✓ No old structure found - nothing to migrate")
        return 0

    print(f"Found {len(migrations)} items to migrate\n")

    # Group by pipeline
    by_pipeline = {}
    for m in migrations:
        pipeline = m['pipeline']
        if pipeline not in by_pipeline:
            by_pipeline[pipeline] = []
        by_pipeline[pipeline].append(m)

    for pipeline, items in by_pipeline.items():
        subjects = set(m['subject'] for m in items)
        print(f"  Pipeline '{pipeline}': {len(items)} runs across {len(subjects)} subjects")

    print()

    # Migrate state database
    print("🗄️  Migrating state database...")
    old_state_db = project_root / "derivatives" / ".eegcpm" / "state.db"
    new_state_db = eegcpm_root / ".eegcpm" / "state.db"
    migrate_state_db(old_state_db, new_state_db, dry_run=args.dry_run)
    print()

    # Execute migration
    print("🚀 Migrating data...")
    stats = execute_migration(
        migrations,
        dry_run=args.dry_run,
        create_symlinks=args.symlinks
    )
    print()

    # Summary
    print("=" * 80)
    print("Migration Summary")
    print("=" * 80)
    print(f"Success: {stats['success']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")

    if stats['errors']:
        print("\nErrors:")
        for error in stats['errors'][:5]:  # Show first 5
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    if args.dry_run:
        print("\n⚠️  This was a DRY RUN - no changes were made")
        print("Run with --execute to perform the migration")
    else:
        print(f"\n✓ Migration complete!")
        print(f"\nNew structure:")
        print(f"  Data: {project_root}/derivatives/preprocessing/")
        print(f"  State DB: {new_state_db}")

    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
